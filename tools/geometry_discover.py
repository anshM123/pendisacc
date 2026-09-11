"""Does the estimator recover the known null direction WITHOUT being told?

This is the falsifiable check on dynamics/geometry.py. The uniform inertial
direction is an exact symmetry of the passive dynamics under prescribed base
motion, proved in dynamics/symmetry.py. The estimator is given a 12-parameter
space and no hint that the symmetry exists. If it is a discovery method rather
than a restatement, the uniform direction must come out among the smallest
eigenvalues of G.

Scored in advance:

  PASS if the analytic direction's alignment with the two most-null
  eigenvectors is >= 0.90, AND the ratio of the largest to the smallest
  eigenvalue exceeds 100 (i.e. the geometry really is anisotropic rather than
  the estimator reporting noise).

CPU only. Uses the standalone closed loop, so it does not touch the GPU.

  run.cmd tools/geometry_discover.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import IQ, NZ, SimCfg  # noqa: E402
from dynamics.geometry import (  # noqa: E402
    NAMES, metric, spectrum, subspace_alignment, uniform_inertial_direction,
)
from dynamics.policy import Actor  # noqa: E402

OUT = os.path.join(ROOT, "results", "geometry_discover.json")
CKPT = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                    "2026-09-05_21-02-09_rel1", "model_800.pt")
N_IC = 6
HORIZON_S = 1.2
ALIGN_MIN = 0.90
ANISO_MIN = 100.0


def main() -> int:
    if not os.path.exists(CKPT):
        print("missing frozen checkpoint:", CKPT)
        return 1
    actor = Actor(CKPT)
    base = SimCfg()
    n_steps = int(HORIZON_S / base.dt_ctrl)

    rng = np.random.default_rng(0)
    z0s = []
    for _ in range(N_IC):
        z = np.zeros(NZ)
        # near dead hang, the distribution every episode actually starts from
        z[IQ] = [rng.uniform(-0.05, 0.05), np.pi + rng.uniform(-0.25, 0.25),
                 rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)]
        z0s.append(z)

    print("Estimating G over %d initial conditions, %d-parameter space,"
          % (len(z0s), len(NAMES)))
    print("%.1f s window. The estimator is NOT told the symmetry exists.\n" % HORIZON_S)

    G, _ = metric(actor, base, z0s, n_steps)
    w, V = spectrum(G)

    print("  eigenvalues of G, ascending (null first):")
    for i, val in enumerate(w):
        top = np.argsort(-np.abs(V[:, i]))[:3]
        desc = ", ".join("%s %+.2f" % (NAMES[t], V[t, i]) for t in top)
        print("    %2d  %.3e   dominated by: %s" % (i, val, desc))

    u = uniform_inertial_direction()
    a1 = subspace_alignment(V, 1, u)
    a2 = subspace_alignment(V, 2, u)
    a3 = subspace_alignment(V, 3, u)
    aniso = float(w[-1] / max(w[0], 1e-300))

    print("\n  analytic null direction (uniform m and I, equal relative):")
    print("    alignment with the 1 most-null eigenvector : %.3f" % a1)
    print("    alignment with the 2 most-null eigenvectors: %.3f" % a2)
    print("    alignment with the 3 most-null eigenvectors: %.3f" % a3)
    print("    anisotropy lambda_max / lambda_min         : %.3e" % aniso)

    ok = (a2 >= ALIGN_MIN) and (aniso >= ANISO_MIN)
    print("\n  thresholds fixed in advance: alignment(2) >= %.2f and"
          % ALIGN_MIN)
    print("  anisotropy >= %.0f  ->  %s" % (ANISO_MIN, "PASS" if ok else "FAIL"))
    if ok:
        print("\n  The estimator recovered an exact dynamical symmetry from")
        print("  closed-loop trajectories alone, without being given it. The")
        print("  same procedure can therefore be pointed at parameters whose")
        print("  geometry is NOT known analytically.")
    else:
        print("\n  The estimator did not recover the known answer, so it cannot")
        print("  be trusted on parameters whose answer is unknown. Reported as")
        print("  a failure of the method, not of the symmetry.")

    # the most sensitive direction, which has no analytic counterpart
    print("\n  most SENSITIVE direction (largest eigenvalue):")
    for t in np.argsort(-np.abs(V[:, -1]))[:5]:
        print("    %-10s %+.3f" % (NAMES[t], V[t, -1]))

    json.dump({"params": NAMES, "eigenvalues": w.tolist(),
               "eigenvectors": V.tolist(), "n_initial_conditions": len(z0s),
               "horizon_s": HORIZON_S, "checkpoint": os.path.basename(CKPT),
               "alignment_1": a1, "alignment_2": a2, "alignment_3": a3,
               "anisotropy": aniso, "align_min": ALIGN_MIN,
               "aniso_min": ANISO_MIN, "pass": bool(ok)},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
