"""Is the upright equilibrium holdable through a servo with lag tau?

The swing-up policy reaches upright in 100% of episodes and holds it in ~37%.
Before blaming PPO, ask whether the ACTUATOR can do it at all: the upright
equilibrium diverges at lambda_max = 15.5 rad/s, while a 100 ms first-order lag
is an actuator bandwidth of only 10 rad/s. An actuator slower than the
instability it must catch is a hardware problem, not a tuning problem.

Model. The RL action is a commanded cart velocity, not a force, so the plant
seen by the controller is

    v_ref_dot = (v_cmd - v_ref) / tau           first-order drive lag
    F         = k_v * (v_ref - x_dot)           stiff inner velocity loop
    M(q) qdd  = B F - C qd - G(q)               the machine

linearised about upright, giving a 9-state system [q(4), qd(4), v_ref] with
v_cmd as the single input. LQR is generous here on purpose: it is the best a
full-state linear controller can do, so whatever it cannot achieve, PPO cannot
achieve either.

Reported per tau:
  * closed-loop settling rate (slowest decaying mode)
  * the peak |v_cmd| a 5 degree lean demands, against the drive's 4.0 m/s
  * the largest lean that stays within 4.0 m/s -- a linear proxy for the catch
    basin the swing-up has to land inside

  run.cmd tools\actuator_bandwidth.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml
from scipy.linalg import expm, solve_continuous_are

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.analytical.triple_pendulum import load_default  # noqa: E402

HW = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot", "hardware.yaml"), encoding="utf-8"))["drive"]
KV = float(HW["velocity_loop_gain_Nsm"])
VMAX = float(HW["max_cart_speed_ms"])


def augmented(model, tau):
    """A, B for z = [q(4), qd(4), v_ref], input v_cmd."""
    A8, B8, _ = model.linearize()          # force input
    b = B8[:, 0]
    A = np.zeros((9, 9))
    A[:8, :8] = A8
    A[:8, 4] -= b * KV                     # F = k_v (v_ref - xdot); -k_v * xdot
    A[:8, 8] += b * KV                     # +k_v * v_ref
    A[8, 8] = -1.0 / tau
    B = np.zeros((9, 1))
    B[8, 0] = 1.0 / tau
    return A, B


def main() -> int:
    model = load_default()
    A8, _, _ = model.linearize()
    lam = np.linalg.eigvals(A8)
    lmax = float(max(lam.real))
    print("plant (force input): lambda_max = %.4f rad/s -> %.1f ms divergence time\n"
          % (lmax, 1000.0 / lmax))

    # Penalise the three link angles and their rates; the cart is left almost
    # free so the reported numbers describe BALANCE, not recentring.
    Q = np.diag([0.1, 100.0, 100.0, 100.0, 0.1, 10.0, 10.0, 10.0, 0.1])

    # Self-check: closing the velocity loop and adding the lag state must not
    # change the instability being caught. If the augmented A is built wrong,
    # this is where it shows.
    for tau in (0.002, 0.100):
        A, _ = augmented(model, tau)
        ol = float(max(np.linalg.eigvals(A).real))
        assert abs(ol - lmax) < 0.05 * lmax, (
            "augmented plant lambda_max %.4f != open-loop %.4f at tau=%.3f" % (ol, lmax, tau))
        print("  check: tau=%5.0f ms -> augmented lambda_max %.4f rad/s (plant %.4f)"
              % (1000 * tau, ol, lmax))
    print()

    print("For each tau: sweep the LQR control cost R and find the cheapest")
    print("controller that pulls a 5 deg lean back inside 10% within 1.0 s.")
    print()
    print("  tau     BW     BW/lam   catchable?   catch time   peak |v_cmd|   max lean")
    print("  [ms]  [rad/s]                            [s]       for 5 deg     in 4 m/s")
    print("  " + "-" * 82)
    for tau in (0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.400):
        A, B = augmented(model, tau)
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
            z0 = np.zeros(9)
            z0[1:4] = np.deg2rad(5.0)
            # Judge the LINK ANGLES only. Eigenvalue bookkeeping is unusable
            # here: the cart position is deliberately almost unpenalised, so the
            # slowest stable mode is always the cart drifting home (-0.62 /s
            # regardless of tau) and any criterion based on it reports failure
            # even at tau = 2 ms, where the actuator is 32x faster than the fall.
            peak, tcatch = response(Acl, K, z0)
            if tcatch is None:
                continue
            if best is None or peak < best[0]:
                best = (peak, tcatch)
        if best is None:
            print("  %5.0f  %6.1f    %5.2f      NO -- no R catches it inside 1.0 s"
                  % (1000 * tau, 1.0 / tau, (1.0 / tau) / lmax))
            continue
        peak, tcatch = best
        lean = np.rad2deg(np.deg2rad(5.0) * VMAX / peak) if peak > 0 else float("inf")
        print("  %5.0f  %6.1f    %5.2f       yes        %5.2f      %7.2f m/s   %6.2f deg"
              % (1000 * tau, 1.0 / tau, (1.0 / tau) / lmax, tcatch, peak, lean))

    print()
    print("The catch basin is what matters for swing-up: 'max lean' is how far off")
    print("vertical the policy may arrive and still be caught inside 4.0 m/s.")
    return 0


def response(Acl, K, z0, T=1.0, n=2000, frac=0.10):
    """(peak |v_cmd|, time for the link angles to fall inside `frac`) or None.

    Matrix exponential, not Euler: with tau = 2 ms the actuator pole sits at
    -500 /s and an explicit step overflows.
    """
    dt = T / n
    M = expm(Acl * dt)
    z = z0.copy()
    lean0 = float(np.max(np.abs(z0[1:4])))
    peak, tcatch = 0.0, None
    for k in range(n):
        peak = max(peak, abs(float((-K @ z)[0])))
        z = M @ z
        if tcatch is None and float(np.max(np.abs(z[1:4]))) < frac * lean0:
            tcatch = (k + 1) * dt
    return peak, tcatch


if __name__ == "__main__":
    raise SystemExit(main())
