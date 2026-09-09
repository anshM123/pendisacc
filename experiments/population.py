"""Does trajectory fidelity predict transfer? Does D_SW?

The experiment. One frozen policy, one hidden pseudo-reality R*, and a
population of candidate simulators. For each candidate S_i measure

  E_traj(S_i, R*)   conventional fidelity: how far the trajectories diverge
                    under the same policy from the same initial condition
  D_SW(S_i, R*)     the same model discrepancy, weighted by how much the
                    closed loop amplifies it before the horizon ends
  |J(S_i) - J(R*)|  the thing anyone actually cares about: how wrong the
                    simulator is about the OUTCOME

and ask which of the first two predicts the third.

R* is deliberately OUTSIDE the candidate family. It has Stribeck friction, a
second-order drive, a deadband and transport delay; no candidate has any of
those, they differ from nominal only in parameters and in viscous/Coulomb
friction. Without that the whole result would be circular -- a reviewer would
say reality was drawn from the randomisation distribution.

J is evaluated on the quantity that actually fails. Under randomised conditions
the frozen policy reaches upright in 100.00% of trials and every loss is a rail
excursion during the transient, so the outcome of interest is the peak cart
excursion and whether it stays inside the 0.60 m bound.

  run.cmd experiments/population.py --horizon 2.5 --episodes 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import (  # noqa: E402
    IQ, NZ, ClosedLoop, DriveCfg, FrictionCfg, SimCfg,
    stability_weighted_gap, transition_matrices,
)
from dynamics.policy import Actor  # noqa: E402

LINKS = np.array([0.24999, 0.25000, 0.31575])
RAIL = 0.60

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.join(
    ROOT, "logs", "rsl_rl", "tip_swingup", "2026-09-05_22-45-54_rel3", "model_700.pt"))
ap.add_argument("--horizon", type=float, default=2.5, help="seconds of variational analysis")
ap.add_argument("--task_seconds", type=float, default=12.0)
ap.add_argument("--episodes", type=int, default=12, help="initial conditions per simulator")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "population.json"))
args = ap.parse_args()


# --------------------------------------------------------------- the reality
def pseudo_reality() -> SimCfg:
    """Model-form effects that NO candidate contains."""
    return SimCfg(
        name="R*",
        drive=DriveCfg(order=2, tau=0.100, zeta=0.7, deadband=0.02, delay_steps=2),
        friction=FrictionCfg(model="stribeck", b_cart=1.2, fc_cart=1.8, fs_cart=3.2,
                             vs_cart=0.03, b_joint=2.0e-4, fc_joint=1.5e-3,
                             fs_joint=3.0e-3, vs_joint=0.05),
        mass_scale=(1.04, 0.97, 1.03), inertia_scale=(1.05, 0.96, 1.02),
        cart_mass_scale=1.03,
    )


# ---------------------------------------------------------- the candidates
def population(rng: np.random.Generator, n: int) -> list:
    """Parameter and viscous/Coulomb variation only -- no model-form effects."""
    sims = [SimCfg(name="nominal")]
    for i in range(n):
        u = rng.uniform
        sims.append(SimCfg(
            name="S%02d" % (i + 1),
            drive=DriveCfg(tau=float(u(0.06, 0.16)), kv=float(u(300.0, 520.0))),
            friction=FrictionCfg(
                model=rng.choice(["none", "viscous", "coulomb"]),
                b_cart=float(u(0.0, 2.5)), fc_cart=float(u(0.0, 3.0)),
                b_joint=float(u(0.0, 4.0e-4)), fc_joint=float(u(0.0, 2.5e-3))),
            mass_scale=tuple(u(0.90, 1.12, 3)),
            inertia_scale=tuple(u(0.90, 1.12, 3)),
            cart_mass_scale=float(u(0.92, 1.10)),
            gravity=float(u(9.79, 9.82)),
        ))
    return sims


def initial_conditions(rng, m: int, q0: np.ndarray) -> np.ndarray:
    Z = np.zeros((m, NZ))
    for i in range(m):
        Z[i, IQ] = q0 + np.concatenate([[rng.uniform(-0.05, 0.05)],
                                        rng.uniform(-0.08, 0.08, 3)])
        Z[i, 4:8] = np.concatenate([[rng.uniform(-0.05, 0.05)],
                                    rng.uniform(-0.15, 0.15, 3)])
    return Z


def outcome(loop: ClosedLoop, z0: np.ndarray, n: int) -> dict:
    Z = loop.rollout(z0, n)
    x = Z[:, 0]
    tip = (LINKS * np.cos(Z[:, IQ][:, 1:4])).sum(axis=1) / LINKS.sum()
    hit = int(np.argmax(np.abs(x) > RAIL)) if (np.abs(x) > RAIL).any() else -1
    if hit >= 0:
        return {"success": False, "peak_x": float(np.abs(x[:hit + 1]).max()),
                "hold": 0.0, "Z": Z}
    q = tip[int(0.75 * len(tip)):]
    return {"success": bool((tip > 0.9).any() and (q > 0.9).mean() > 0.95),
            "peak_x": float(np.abs(x).max()),
            "hold": float((q > 0.9).mean()), "Z": Z}


def main() -> int:
    rng = np.random.default_rng(7)
    actor = Actor(args.checkpoint)
    d = np.load(os.path.join(ROOT, "results", "rollout_success.npz"), allow_pickle=True)
    dt = float(d["dt"])
    n_var = int(args.horizon / dt)
    n_task = int(args.task_seconds / dt)

    Z0 = initial_conditions(rng, args.episodes, d["q"][0])
    sims = population(rng, 23)
    R = pseudo_reality()
    loop_r = ClosedLoop(actor, R)

    print("pseudo-reality R*: stribeck friction, 2nd-order drive (zeta 0.7),")
    print("                   %.0f mm/s deadband, %d-step delay, +-4%% masses" %
          (1000 * R.drive.deadband, R.drive.delay_steps))
    print("candidates       : %d, parameter + viscous/coulomb variation only\n" % len(sims))

    # reality's outcome, over the same initial conditions
    r_out = [outcome(loop_r, Z0[i], n_task) for i in range(args.episodes)]
    r_succ = float(np.mean([o["success"] for o in r_out]))
    r_peak = float(np.mean([o["peak_x"] for o in r_out]))
    print("R*: success %.1f%%, mean peak |x| %.3f m\n" % (100 * r_succ, r_peak))

    rows = []
    for si, S in enumerate(sims):
        loop_s = ClosedLoop(actor, S)
        s_out = [outcome(loop_s, Z0[i], n_task) for i in range(args.episodes)]
        s_succ = float(np.mean([o["success"] for o in s_out]))
        s_peak = float(np.mean([o["peak_x"] for o in s_out]))

        # conventional fidelity: state divergence under the same policy
        etraj = float(np.mean([
            np.linalg.norm(s_out[i]["Z"][:n_var + 1, :8] - r_out[i]["Z"][:n_var + 1, :8], axis=1).mean()
            for i in range(args.episodes)]))

        # stability-weighted gap, averaged over the same initial conditions
        dsw = graw = gT = 0.0
        for i in range(args.episodes):
            Zs = s_out[i]["Z"][:n_var + 1]
            A = transition_matrices(loop_s, Zs)
            g = stability_weighted_gap(loop_s, loop_r, Z0[i], n_var, A=A, Zs=Zs)
            dsw += g["D_SW"]; graw += g["raw_gap"]; gT += g["G_T"]
        m = args.episodes
        dsw, graw, gT = dsw / m, graw / m, gT / m

        rows.append({"name": S.name, "success": s_succ, "peak_x": s_peak,
                     "E_traj": etraj, "D_SW": dsw, "raw_gap": graw, "G_T": gT,
                     "err_success": abs(s_succ - r_succ),
                     "err_peak": abs(s_peak - r_peak)})
        print("  %-8s success %5.1f%%  peak|x| %.3f  E_traj %.4f  raw %.4f  D_SW %9.3f  G_T %8.1f"
              % (S.name, 100 * s_succ, s_peak, etraj, graw, dsw, gT))

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

    E = np.array([r["E_traj"] for r in rows])
    Draw = np.array([r["raw_gap"] for r in rows])
    Dsw = np.array([r["D_SW"] for r in rows])
    ep = np.array([r["err_peak"] for r in rows])
    es = np.array([r["err_success"] for r in rows])

    print("\nSpearman rank correlation with the OUTCOME error:")
    print("                       vs |peak|x| error|   vs |success error|")
    for label, v in (("E_traj (trajectory)", E), ("raw model gap", Draw), ("D_SW (stability-weighted)", Dsw)):
        print("  %-26s %8.3f            %8.3f" % (label, spearman(v, ep), spearman(v, es)))

    json.dump({
        "checkpoint": os.path.basename(args.checkpoint),
        "horizon_s": args.horizon, "episodes": args.episodes,
        "reality": {"success": r_succ, "peak_x": r_peak},
        "rows": rows,
        "spearman": {
            "E_traj_vs_peak": spearman(E, ep), "E_traj_vs_success": spearman(E, es),
            "raw_vs_peak": spearman(Draw, ep), "raw_vs_success": spearman(Draw, es),
            "D_SW_vs_peak": spearman(Dsw, ep), "D_SW_vs_success": spearman(Dsw, es),
        },
    }, open(args.out, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
