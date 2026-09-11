"""H6: do policies that are identical in the simulator see the same geometry?

Pre-registered in PREREGISTRATION_H6.md. Estimates the metric tensor G for
each of the three nominal velocity policies -- same simulator, same reward,
same architecture, same iteration count, differing only in random seed -- and
compares their null subspaces.

P2 is checked FIRST: the analytic uniform-inertial direction is an exact
symmetry of the passive dynamics and must be recovered by every policy. If it
is not, the estimator is reporting noise and P1 means nothing.

  run.cmd tools/geometry_policy.py
"""

from __future__ import annotations

import glob
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

OUT = os.path.join(ROOT, "results", "geometry_policy.json")
K = 3                 # dimension of the null subspace compared
D_MIN = 0.30          # P1 threshold, frozen
ALIGN_MIN = 0.85      # P2 threshold, frozen
N_IC = 6
HORIZON_S = 1.2

# selected in their own simulator by claim5_deploy, before H6 existed
SEEDS = {
    "s1": ("2026-09-09_23-13-48_C5_S_nominal_s1", "model_800.pt", 100.0),
    "s2": ("2026-09-09_23-53-13_C5_S_nominal_s2", "model_600.pt", 71.1),
    "s3": ("2026-09-10_00-35-47_C5_S_nominal_s3", "model_600.pt", 23.0),
}


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Principal-subspace distance between two orthonormal bases."""
    k = A.shape[1]
    return float(np.sqrt(max(k - np.linalg.norm(A.T @ B, "fro") ** 2, 0.0)))


def main() -> int:
    base = SimCfg()
    n_steps = int(HORIZON_S / base.dt_ctrl)
    rng = np.random.default_rng(0)
    z0s = []
    for _ in range(N_IC):
        z = np.zeros(NZ)
        z[IQ] = [rng.uniform(-0.05, 0.05), np.pi + rng.uniform(-0.25, 0.25),
                 rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)]
        z0s.append(z)

    u = uniform_inertial_direction()
    res = {}
    print("Estimating G per policy. Same simulator, reward, architecture and")
    print("iteration count; only the random seed differs.\n")
    for tag, (run, ck, _) in SEEDS.items():
        p = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup", run, ck)
        if not os.path.exists(p):
            print("missing checkpoint:", p)
            return 1
        G, _ = metric(Actor(p), base, z0s, n_steps)
        w, V = spectrum(G)
        res[tag] = {"eigenvalues": w.tolist(), "V": V.tolist(),
                    "align_u_k3": subspace_alignment(V, K, u),
                    "anisotropy": float(w[-1] / max(w[0], 1e-300)),
                    "checkpoint": ck, "run": run}
        top = np.argsort(-np.abs(V[:, 0]))[:3]
        print("  %-3s  lambda_min %.3e  aniso %.2e  most-null: %s"
              % (tag, w[0], res[tag]["anisotropy"],
                 ", ".join("%s %+.2f" % (NAMES[t], V[t, 0]) for t in top)))

    # ---- P2 first: the exact symmetry must be stable across seeds ---------
    print("\nP2 (control): the analytic symmetry must be recovered by all three")
    aligns = {t: res[t]["align_u_k3"] for t in res}
    for t, a in aligns.items():
        print("    %-3s alignment with its 3 most-null eigenvectors: %.3f" % (t, a))
    p2 = all(a >= ALIGN_MIN for a in aligns.values())
    print("    threshold %.2f  ->  %s" % (ALIGN_MIN, "SUPPORTED" if p2 else "FAIL"))

    # ---- P1: do the null subspaces differ? --------------------------------
    tags = list(SEEDS)
    ds = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A = np.array(res[tags[i]]["V"])[:, :K]
            B = np.array(res[tags[j]]["V"])[:, :K]
            ds["%s-%s" % (tags[i], tags[j])] = subspace_distance(A, B)
    dbar = float(np.mean(list(ds.values())))
    print("\nP1 (primary): pairwise distance between %d-dim null subspaces" % K)
    for k, v in ds.items():
        print("    %-8s %.3f" % (k, v))
    print("    mean %.3f  (threshold %.2f, max possible %.3f)"
          % (dbar, D_MIN, np.sqrt(K)))
    if not p2:
        p1 = "INVALIDATED -- P2 failed, so the estimator is not trustworthy here"
    elif dbar >= D_MIN:
        p1 = "SUPPORTED"
    else:
        p1 = "FAIL -- geometry is not policy dependent at this resolution"
    print("    -> %s" % p1)

    # ---- exploratory only -------------------------------------------------
    print("\nEXPLORATORY (n = 3, cannot support a correlation):")
    print("    policy   transfer in R*   lambda_min      anisotropy")
    for t in tags:
        print("    %-6s   %6.1f%%          %.3e   %.3e"
              % (t, SEEDS[t][2], res[t]["eigenvalues"][0], res[t]["anisotropy"]))

    json.dump({"K": K, "d_min": D_MIN, "align_min": ALIGN_MIN,
               "params": NAMES, "per_policy": res,
               "pairwise_subspace_distance": ds, "mean_distance": dbar,
               "P2": "SUPPORTED" if p2 else "FAIL", "P1": p1,
               "exploratory_transfer": {t: SEEDS[t][2] for t in tags},
               "note": "Estimated on the CPU reproduction of the simulator, "
                       "which RESULTS.md records as discovery-grade only."},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
