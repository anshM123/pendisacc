"""Counterfactual force demand along the trajectory the policy actually flies.

Two corrections to the earlier estimate, both of which mattered.

1. The demand is AFFINE in the mass scale, not linear. The cart and reflected
   drivetrain inertia do not scale, so for a prescribed trajectory

       F_req(t; c) = m_cart * xddot(t)  +  c * F_links(t)

   and the peak over t need not occur at the same instant for different c, so
   even F_peak(c) is not affine. Predicting c * F_peak(1) is wrong twice over.

2. Peak clipping is NOT the failure criterion. c = 10 succeeds at 100% while
   its demand exceeds the clamp, so saturation breaks the exact symmetry
   CONTINUOUSLY and the policy tolerates some of that before crossing a
   transfer boundary. What matters is how much authority is missing and WHEN.

Reported per scale, holding the nominal trajectory fixed (true counterfactual
inverse dynamics -- what the drive would have had to supply to reproduce the
successful motion):

    F_peak        peak demand
    duty          fraction of the horizon spent saturated
    deficit       integral of |F_req - clip(F_req)|, the missing impulse [N s]
    when          whether saturation lands in the pump or the capture

  run.cmd tools/force_counterfactual.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import IQ, IQD, NZ, ClosedLoop, SimCfg  # noqa: E402
from dynamics.policy import Actor  # noqa: E402
from dynamics.analytical.triple_pendulum import PendulumParams, TriplePendulumModel  # noqa: E402
from dataclasses import replace  # noqa: E402

CLAMP = 349.5
LINKS = np.array([0.24999, 0.25000, 0.31575])


def main() -> int:
    actor = Actor(os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                               "2026-09-05_22-45-54_rel3", "model_700.pt"))
    d = np.load(os.path.join(ROOT, "results", "rollout_success.npz"), allow_pickle=True)
    dt = float(d["dt"])
    z0 = np.zeros(NZ); z0[IQ] = d["q"][0]

    # the trajectory the policy actually flies, nominal
    loop = ClosedLoop(actor, SimCfg())
    n = int(3.0 / dt)
    Z = loop.rollout(z0, n)
    q, qd = Z[:, IQ], Z[:, IQD]
    qdd = np.gradient(qd, dt, axis=0)
    tip = (LINKS * np.cos(q[:, 1:4])).sum(axis=1) / LINKS.sum()
    t = np.arange(n + 1) * dt
    t_capture = float(t[np.argmax(tip > 0.9)]) if (tip > 0.9).any() else float("nan")

    p = PendulumParams.from_yaml()
    m1 = TriplePendulumModel(p)
    links_only = [replace(l, m=0.0, I=0.0) for l in p.links]
    m0 = TriplePendulumModel(replace(p, links=links_only))   # cart alone

    F1 = np.array([(m1.M(q[k]) @ qdd[k] + m1.C(q[k], qd[k]) @ qd[k] + m1.G(q[k]))[0]
                   for k in range(n + 1)])
    F0 = np.array([(m0.M(q[k]) @ qdd[k] + m0.C(q[k], qd[k]) @ qd[k] + m0.G(q[k]))[0]
                   for k in range(n + 1)])
    F_links = F1 - F0                    # the part that scales with c

    print("counterfactual demand along the FIXED nominal trajectory")
    print("  F_req(t; c) = F_cart(t) + c * F_links(t)")
    print("  cart part peaks at %.1f N, link part at %.1f N, total at %.1f N\n"
          % (np.abs(F0).max(), np.abs(F_links).max(), np.abs(F1).max()))
    print("    c    F_peak[N]  peak/clamp   duty    deficit[N s]   saturation lands in")
    print("  " + "-" * 74)

    out = []
    for c in (1, 2, 5, 10, 15, 20, 30):
        F = F0 + c * F_links
        sat = np.abs(F) > CLAMP
        duty = float(sat.mean())
        deficit = float(np.sum(np.maximum(np.abs(F) - CLAMP, 0.0)) * dt)
        if sat.any():
            ts = t[sat]
            frac_capture = float(((ts > t_capture - 0.4) & (ts < t_capture + 0.4)).mean())
            where = "capture %.0f%%, pump %.0f%%" % (100 * frac_capture, 100 * (1 - frac_capture))
        else:
            where = "-"
        out.append({"c": c, "F_peak": float(np.abs(F).max()), "duty": duty,
                    "deficit_Ns": deficit, "where": where})
        print("  %4d   %8.1f   %7.2f    %5.1f%%   %10.2f    %s"
              % (c, np.abs(F).max(), np.abs(F).max() / CLAMP, 100 * duty, deficit, where))

    print("\n  measured: c=10 succeeds at 100%, c=20 fails at 0%.")
    print("  capture is at t = %.2f s." % t_capture)
    json.dump({"t_capture_s": t_capture, "clamp_N": CLAMP, "rows": out},
              open(os.path.join(ROOT, "results", "force_counterfactual.json"), "w"), indent=1)
    print("\n[out] results/force_counterfactual.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
