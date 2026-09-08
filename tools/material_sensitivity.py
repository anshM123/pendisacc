"""What changes if the links are not made of SolidWorks' default material?

Every link body in the CAD carries density exactly 1000.0 kg/m^3, which is
SolidWorks' default for a part with no material assigned -- not PETG (1270),
not aluminium (2700), not steel (7850). The link masses in
triple_pendulum_params.yaml are therefore placeholders, and lambda_max, the
divergence time that set the 250 Hz control rate, and the actuator sizing all
descend from them.

Geometry is fixed, so swapping material scales each link's mass and its inertia
about any axis by the same density ratio, and leaves lc and L untouched. This
sweeps that ratio and reports what the control designer actually needs:

  * lambda_max and its divergence time constant
  * peak cart force to hold a 5 degree lean, against the drive's 99.7 N rated
    and 349.5 N peak
  * peak commanded velocity, against 4.0 m/s

  run.cmd tools/material_sensitivity.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import yaml
from scipy.linalg import expm, solve_continuous_are

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.analytical.triple_pendulum import PendulumParams, TriplePendulumModel  # noqa: E402

HW = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot", "hardware.yaml"), encoding="utf-8"))["drive"]
KV = float(HW["velocity_loop_gain_Nsm"])
VMAX = float(HW["max_cart_speed_ms"])
FPEAK = float(HW["peak_cart_force_N"])
FRATED = float(HW["rated_cart_force_used_N"])

MATERIALS = [
    ("CAD default (water)", 1000.0),
    ("PETG",                1270.0),
    ("aluminium 6061",      2700.0),
    ("steel",               7850.0),
]


def scaled_model(ratio: float) -> TriplePendulumModel:
    p = PendulumParams.from_yaml()
    links = [replace(l, m=l.m * ratio, I=l.I * ratio) for l in p.links]
    return TriplePendulumModel(replace(p, links=links))


def augmented(model, tau=0.100):
    A8, B8, _ = model.linearize()
    b = B8[:, 0]
    A = np.zeros((9, 9))
    A[:8, :8] = A8
    A[:8, 4] -= b * KV
    A[:8, 8] += b * KV
    A[8, 8] = -1.0 / tau
    B = np.zeros((9, 1))
    B[8, 0] = 1.0 / tau
    return A, B


def demand(model):
    """Peak |v_cmd| and |F| catching a 5 deg lean, best over a sweep of R."""
    A, B = augmented(model)
    Q = np.diag([0.1, 100.0, 100.0, 100.0, 0.1, 10.0, 10.0, 10.0, 0.1])
    best = None
    for R in np.logspace(-5, 3, 60):
        try:
            P = solve_continuous_are(A, B, Q, np.array([[R]]))
        except Exception:
            continue
        K = np.linalg.solve(np.array([[R]]), B.T @ P)
        Acl = A - B @ K
        if max(np.linalg.eigvals(Acl).real) > 0:
            continue
        z = np.zeros(9)
        z[1:4] = np.deg2rad(5.0)
        M = expm(Acl * (1.0 / 2000))
        pv = pf = 0.0
        caught = False
        for k in range(2000):
            pv = max(pv, abs(float((-K @ z)[0])))
            pf = max(pf, abs(KV * (z[8] - z[4])))
            z = M @ z
            if not caught and float(np.max(np.abs(z[1:4]))) < 0.10 * np.deg2rad(5.0):
                caught = True
        if caught and (best is None or pv < best[0]):
            best = (pv, pf)
    return best or (float("nan"), float("nan"))


def main() -> int:
    print("Link material sensitivity. Geometry fixed; mass and inertia scale with density.\n")
    print("material               rho    link masses (kg)          lambda_max   1/lambda   peak F   peak v_cmd")
    print("-" * 108)
    for name, rho in MATERIALS:
        ratio = rho / 1000.0
        m = scaled_model(ratio)
        lam = float(max(np.linalg.eigvals(m.linearize()[0]).real))
        pv, pf = demand(m)
        masses = ", ".join("%.3f" % l.m for l in m.p.links)
        print("%-20s %6.0f   %-24s %8.4f    %5.1f ms  %6.1f N   %5.2f m/s"
              % (name, rho, masses, lam, 1000.0 / lam, pf, pv))
    print("\ndrive limits: rated %.1f N, peak %.1f N, max speed %.1f m/s" % (FRATED, FPEAK, VMAX))
    print("control rate is 250 Hz; it must resolve the divergence time constant above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
