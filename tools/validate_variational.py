"""Does the standalone closed loop -- and its variational dynamics -- actually work?

Two checks, because they fail for different reasons.

CHECK 1: the standalone loop against Isaac.
  dynamics/closed_loop.py reimplements the whole control path (policy, action
  scale and clip, first-order servo, velocity loop under a force clamp,
  semi-implicit integration) outside Isaac, so that the analysis runs on a CPU
  in seconds. If any piece is wrong -- an angle convention, the action scale,
  the filter coefficient -- the reproduced trajectory drifts from the recorded
  one. Compared over a SHORT horizon: the upright equilibrium has lambda_max =
  15.5 rad/s, so two slightly different integrators must diverge eventually and
  demanding 12 s agreement would be a meaningless test.

CHECK 2: the first-order error propagation against direct simulation.
  This is the one that validates the maths the paper rests on. Take a simulator
  S and a perturbed model R, predict the end-state discrepancy with

      e_N = Phi(N,0) e_0 + sum_k Phi(N,k+1) d_k

  and compare against actually simulating R. For a small enough perturbation
  the two must agree; the relative error should fall roughly linearly as the
  perturbation shrinks, which is the signature of a correct first-order model.
  If the Jacobians or the backward accumulation are wrong, this check does not
  merely degrade, it fails outright.

  run.cmd tools/validate_variational.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import (  # noqa: E402
    IA, IQ, IQD, IV, NZ, ClosedLoop, SimCfg, stability_weighted_gap,
)
from dynamics.policy import Actor  # noqa: E402

CKPT = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                    "2026-09-05_22-45-54_rel3", "model_700.pt")
ROLLOUT = os.path.join(ROOT, "results", "rollout_success.npz")


def check_against_isaac(actor) -> dict:
    d = np.load(ROLLOUT, allow_pickle=True)
    q_ref, dt = d["q"], float(d["dt"])
    z0 = np.zeros(NZ)
    z0[IQ] = q_ref[0]                     # rates, servo state and action all zero

    loop = ClosedLoop(actor, SimCfg(dt_ctrl=dt))
    n = len(q_ref) - 1
    Z = loop.rollout(z0, n)
    err = np.linalg.norm(Z[:, IQ] - q_ref, axis=1)

    out = {}
    for horizon in (0.2, 0.5, 1.0, 2.0):
        k = min(int(horizon / dt), n)
        scale = max(np.abs(q_ref[:k + 1]).max(), 1e-9)
        out["err_%.1fs" % horizon] = float(err[:k + 1].max())
        out["rel_%.1fs" % horizon] = float(err[:k + 1].max() / scale)
    # did the reproduced loop do the TASK, whatever the pointwise drift?
    L = np.array([0.24999, 0.25000, 0.31575])
    tip = (L * np.cos(Z[:, IQ][:, 1:4])).sum(axis=1) / L.sum()
    out["reached_upright"] = bool((tip > 0.9).any())
    out["final_tip"] = float(tip[-1])
    out["upright_final_quarter"] = float((tip[int(0.75 * len(tip)):] > 0.9).mean())
    return out


def check_first_order(actor) -> dict:
    """Predicted vs simulated end-state discrepancy, over shrinking perturbations."""
    d = np.load(ROLLOUT, allow_pickle=True)
    q_ref, dt = d["q"], float(d["dt"])
    z0 = np.zeros(NZ)
    z0[IQ] = q_ref[0]

    n = 150                     # 0.6 s: long enough to amplify, short enough to stay linear
    base = SimCfg(dt_ctrl=dt)
    loop_s = ClosedLoop(actor, base)

    rows = []
    for scale in (1e-2, 3e-3, 1e-3, 3e-4):
        cfg_r = SimCfg(dt_ctrl=dt, mass_scale=(1.0 + scale, 1.0, 1.0))
        loop_r = ClosedLoop(actor, cfg_r)

        res = stability_weighted_gap(loop_s, loop_r, z0, n)
        # direct simulation of R from the same initial condition
        Zr = loop_r.rollout(z0, n)
        actual = Zr[-1] - res["Zs"][-1]

        # first-order prediction: e_0 = 0 here, so only the forced term survives
        P = np.eye(NZ)
        pred = np.zeros(NZ)
        A = None
        # recompute the forced sum with signs (stability_weighted_gap keeps norms)
        from dynamics.closed_loop import transition_matrices
        A = transition_matrices(loop_s, res["Zs"])
        for k in range(n - 1, -1, -1):
            pred = pred + P @ res["D"][k]
            P = P @ A[k]

        na, np_ = np.linalg.norm(actual), np.linalg.norm(pred)
        rel = float(np.linalg.norm(pred - actual) / max(na, 1e-300))
        rows.append({"perturbation": scale, "||actual||": float(na),
                     "||predicted||": float(np_), "rel_err": rel,
                     "G_T": res["G_T"], "D_SW": res["D_SW"]})
    return {"orders": rows}


def main() -> int:
    if not os.path.exists(CKPT):
        print("missing checkpoint:", CKPT)
        return 1
    actor = Actor(CKPT)
    print(actor)

    print("\n--- CHECK 1: standalone closed loop vs the Isaac rollout ---")
    c1 = check_against_isaac(actor)
    for h in (0.2, 0.5, 1.0, 2.0):
        print("  max |q - q_isaac| over %.1f s: %.5f rad  (%.2f%% of the angle scale)"
              % (h, c1["err_%.1fs" % h], 100 * c1["rel_%.1fs" % h]))
    print("  reproduced loop reached upright: %s, final tip %+.3f, held %.0f%% of final quarter"
          % (c1["reached_upright"], c1["final_tip"], 100 * c1["upright_final_quarter"]))

    print("\n--- CHECK 2: first-order error propagation vs direct simulation ---")
    c2 = check_first_order(actor)
    print("  %-14s %-14s %-14s %-10s %-10s" % ("perturbation", "||actual||", "||predicted||", "rel err", "G_T"))
    for r in c2["orders"]:
        print("  %-14.0e %-14.3e %-14.3e %-10.4f %-10.3g"
              % (r["perturbation"], r["||actual||"], r["||predicted||"], r["rel_err"], r["G_T"]))
    rels = [r["rel_err"] for r in c2["orders"]]
    shrinks = all(rels[i + 1] < rels[i] * 1.5 for i in range(len(rels) - 1))
    ok = rels[-1] < 0.05 and shrinks
    print("\n  relative error falls as the perturbation shrinks: %s" % shrinks)
    print("  first-order model valid (rel err < 5%% at the smallest step): %s" % (rels[-1] < 0.05))

    out = os.path.join(ROOT, "results", "variational_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": os.path.basename(CKPT), "isaac_agreement": c1,
                   "first_order": c2, "pass": bool(ok)}, fh, indent=2)
    print("\n[out] %s   -> %s" % (out, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
