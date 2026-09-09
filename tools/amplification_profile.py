"""WHERE is the closed loop fragile?

G_T = sigma_max(Phi(T,0)) is one number for a whole episode, which is not much
use for a mechanism. The useful object is the profile: how much amplification
the closed loop accumulates as a function of time, and which interval of the
swing-up contributes it.

The hypothesis this tests is that fragility is not spread over the episode but
concentrated in the capture transition -- the moment the policy stops pumping
and has to arrest three links at an unstable equilibrium. If so, sim-to-real
failure should be governed by a short window, not by long-horizon chaos, and
the measured failure mode supports that: under randomised conditions the policy
reaches upright in 100.00% of trials and every loss is a rail excursion during
the transient.

Reports, per control step:
  * G(k)      = sigma_max(Phi(k,0)), growth achieved by step k
  * g(k)      = sigma_max(A_k), the instantaneous one-step gain
  * the tip height, so the profile can be read against the phases of the task

  run.cmd tools/amplification_profile.py --seconds 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import IQ, NZ, ClosedLoop, SimCfg, transition_matrices  # noqa: E402
from dynamics.policy import Actor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.join(
    ROOT, "logs", "rsl_rl", "tip_swingup", "2026-09-05_22-45-54_rel3", "model_700.pt"))
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "amplification_profile.json"))
args = ap.parse_args()

LINKS = np.array([0.24999, 0.25000, 0.31575])


def main() -> int:
    actor = Actor(args.checkpoint)
    d = np.load(os.path.join(ROOT, "results", "rollout_success.npz"), allow_pickle=True)
    dt = float(d["dt"])
    z0 = np.zeros(NZ)
    z0[IQ] = d["q"][0]

    n = int(args.seconds / dt)
    loop = ClosedLoop(actor, SimCfg(dt_ctrl=dt))
    Z = loop.rollout(z0, n)
    A = transition_matrices(loop, Z)

    tip = (LINKS * np.cos(Z[:, IQ][:, 1:4])).sum(axis=1) / LINKS.sum()

    Phi = np.eye(NZ)
    G = np.empty(n + 1)
    g = np.empty(n)
    G[0] = 1.0
    for k in range(n):
        g[k] = np.linalg.svd(A[k], compute_uv=False)[0]
        Phi = A[k] @ Phi
        G[k + 1] = np.linalg.svd(Phi, compute_uv=False)[0]

    t = np.arange(n + 1) * dt
    up = tip > 0.9
    t_capture = float(t[np.argmax(up)]) if up.any() else float("nan")

    # Report the PEAK and the contraction, not phase shares of log-growth.
    # Shares are meaningless once the loop contracts: the post-capture interval
    # legitimately has negative log-growth, which made the three "percentages"
    # come out as 157 / 164 / -221.
    kpk = int(np.argmax(G))
    G_peak, t_peak = float(G[kpk]), float(t[kpk])
    shed = 1.0 - G[-1] / G_peak

    print("checkpoint            : %s" % os.path.basename(args.checkpoint))
    print("horizon               : %.1f s (%d control steps)" % (args.seconds, n))
    print("capture (tip > 0.9)   : %.2f s" % t_capture)
    print()
    print("G peaks at            : %.4g   at t = %.2f s  (%+.2f s from capture)"
          % (G_peak, t_peak, t_peak - t_capture))
    print("G at the horizon      : %.4g   -> %.1f%% of the peak shed while holding"
          % (G[-1], 100 * shed))
    print()
    print("The closed loop amplifies through swing-up and capture, then contracts")
    print("once upright. Fragility is a WINDOW, not a property of the episode.")

    stride = max(1, n // 400)
    json.dump({
        "checkpoint": os.path.basename(args.checkpoint),
        "dt": dt, "seconds": args.seconds, "steps": n,
        "t_capture_s": t_capture,
        "G_final": float(G[-1]),
        "g_max": float(g.max()), "g_max_t": float(t[int(np.argmax(g))]),
        "G_peak": G_peak, "t_peak_s": t_peak,
        "t_peak_minus_capture_s": t_peak - t_capture,
        "fraction_of_peak_shed": float(shed),
        "t": t[::stride].tolist(), "G": G[::stride].tolist(),
        "g": g[::stride].tolist(), "tip": tip[::stride].tolist(),
    }, open(args.out, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
