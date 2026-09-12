"""CPU proof, before any more Isaac time: what can similarity coordinates buy?

No Isaac. Frozen trained policies (CORR_s1..s3) run on the analytical closed
loop (dynamics/closed_loop.py, 10 substeps; agrees with Isaac on nominal).

  A  sanity      CPU success of each frozen policy on the corrected nominal plant
  B  exact gauge scale masses, inertias, cart, k_v AND F_clamp by c -> trajectory
                 difference vs nominal (should be round-off, clamp binding or not)
  C  raw == lite a raw log-box draw r and its quotient projection P r give the
                 same plant, so "raw DR" and "quotient DR without rescale" are
                 the same training distribution
  D  full        what "quotient DR at equal raw budget" actually is: the same
                 distribution with the quotient part widened by E|r| / E|Pr|
  E  sensitivity success of a frozen policy along each dimensionless group
                 m1/mc, m2/mc, m3/mc, k_v/mc, F_clamp/mc, one at a time

  run.cmd tools/pi_prove.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import yaml

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import NZ, ClosedLoop, DriveCfg, SimCfg  # noqa: E402
from dynamics.policy import Actor  # noqa: E402

KV0, F0 = 400.0, 349.5
STEPS = 3000                       # 12 s at 250 Hz, as the task
BOUND, UPRIGHT, HOLD = 0.60, 0.9, 0.25
N_IC = 16
COORDS = ["rho1", "rho2", "rho3", "m_cart", "kv", "F_clamp"]
U = np.ones(6) / np.sqrt(6.0)
P = np.eye(6) - np.outer(U, U)

_p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"),
                         encoding="utf-8"))
L = np.array([float(_p["bodies"][n].get("length", 2.0 * _p["bodies"][n]["l_com"]))
              for n in ("link1", "link2", "link3")])

_ACT = {}


def ckpt(tag):
    run = sorted(d for d in glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup", "*_" + tag))
                 if "VOID" not in d)[-1]
    fs = sorted(glob.glob(os.path.join(run, "model_*.pt")),
                key=lambda f: int("".join(c for c in os.path.basename(f) if c.isdigit())))
    return fs[-1]


def cfg_of(d):
    e = np.exp(np.asarray(d, float))
    return SimCfg(mass_scale=tuple(e[0:3]), inertia_scale=tuple(e[0:3]),
                  cart_mass_scale=float(e[3]),
                  drive=DriveCfg(kv=KV0 * float(e[4]), f_clamp=F0 * float(e[5])))


def ic(k):
    rng = np.random.default_rng(500 + k)
    z = np.zeros(NZ)
    z[0] = rng.uniform(-0.02, 0.02)
    z[1:4] = np.pi + rng.uniform(-0.02, 0.02, 3)
    return z


def episode(job):
    """job = (tag, d, k, keep_traj) -> success record."""
    tag, d, k, keep = job
    if tag not in _ACT:
        _ACT[tag] = Actor(ckpt(tag))
    loop = ClosedLoop(_ACT[tag], cfg_of(d))
    z = ic(k)
    tr = [z[:8].copy()] if keep else None
    hold_from = int(STEPS * (1 - HOLD))
    ever_up, held, early, peakF = False, 0, False, 0.0
    for s in range(STEPS):
        z = loop.step(z)
        if keep:
            tr.append(z[:8].copy())
        tip = float(L @ np.cos(z[1:4]) / L.sum())
        ever_up |= tip > UPRIGHT
        if s >= hold_from:
            held += tip > UPRIGHT
        if abs(z[0]) > BOUND:
            early = s < STEPS - 1
            break
    ok = (not early) and ever_up and held / (STEPS - hold_from) > 0.95
    return {"success": bool(ok), "early": bool(early),
            "hold": held / (STEPS - hold_from), "traj": np.array(tr) if keep else None}


def run(pool, jobs):
    return list(pool.map(episode, jobs, chunksize=1))


def main() -> int:
    out = {}
    rng = np.random.default_rng(20260912)
    with ProcessPoolExecutor(max_workers=14) as pool:
        # A
        tags = ["CORR_s1", "CORR_s2", "CORR_s3"]
        A = {}
        for t in tags:
            r = run(pool, [(t, np.zeros(6), k, False) for k in range(N_IC)])
            A[t] = 100 * np.mean([x["success"] for x in r])
            print("A  %s CPU nominal success %.1f%%" % (t, A[t]), flush=True)
        out["A_nominal_success_pct"] = A
        pol = max(A, key=A.get)
        out["policy_used"] = pol

        # B: exact gauge, including c where the clamp binds harder
        base = run(pool, [(pol, np.zeros(6), k, True) for k in range(2)])
        B = []
        for c in (0.25, 0.5, 2.0, 4.0):
            g = np.log(c) * np.ones(6)
            r = run(pool, [(pol, g, k, True) for k in range(2)])
            dmax = max(float(np.abs(a["traj"] - b["traj"]).max()) for a, b in zip(r, base))
            # and the "orbit" people usually randomise: inertials only, drive fixed
            g2 = g.copy()
            g2[4:] = 0.0
            r2 = run(pool, [(pol, g2, k, False) for k in range(N_IC)])
            B.append({"c": c, "max_abs_traj_diff_exact_gauge": dmax,
                      "success_inertials_only_pct": 100 * np.mean([x["success"] for x in r2])})
            print("B  c=%-5s exact gauge max|dz| %.2e | inertials-only (drive fixed) success %.1f%%"
                  % (c, dmax, B[-1]["success_inertials_only_pct"]), flush=True)
        out["B"] = B

        # C: raw box draw vs its quotient projection
        R = rng.uniform(-0.25, 0.25, size=(24, 6))
        jr = [(pol, r, 0, True) for r in R]
        jp = [(pol, P @ r, 0, True) for r in R]
        rr, rp = run(pool, jr), run(pool, jp)
        dC = [float(np.abs(a["traj"] - b["traj"]).max()) for a, b in zip(rr, rp)]
        out["C_max_abs_traj_diff_raw_vs_projected"] = max(dC)
        print("C  raw draw vs quotient projection, 24 draws: max|dz| %.2e" % max(dC), flush=True)

        # D
        big = np.random.default_rng(1).uniform(-0.25, 0.25, size=(200000, 6))
        k = float(np.linalg.norm(big, axis=1).mean() / np.linalg.norm(big @ P, axis=1).mean())
        out["D_equal_budget_quotient_width_factor"] = k
        print("D  'quotient DR at equal raw budget' == raw DR with quotient widened x%.4f" % k,
              flush=True)

        # E: one dimensionless group at a time
        E = []
        names = ["m1/mc", "m2/mc", "m3/mc", "kv/mc", "Fclamp/mc"]
        for i, nm in enumerate(names):
            idx = i if i < 3 else i + 1          # skip m_cart: it is the reference
            for delta in (-0.7, -0.35, 0.35, 0.7):
                d = np.zeros(6)
                d[idx] = delta
                r = run(pool, [(pol, d, k2, False) for k2 in range(N_IC)])
                s = 100 * np.mean([x["success"] for x in r])
                E.append({"group": nm, "dlog": delta, "factor": float(np.exp(delta)),
                          "success_pct": s})
                print("E  %-10s x%.2f  success %5.1f%%" % (nm, np.exp(delta), s), flush=True)
        out["E_group_sensitivity"] = E

    os.makedirs(os.path.join(ROOT, "results", "PI_prove"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "PI_prove", "prove.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/PI_prove/prove.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
