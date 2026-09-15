"""R1: the uncertainty-exponent pair test in an INDEPENDENT dynamics implementation (no PhysX).

Same frozen CORR_s1 policy, same m1 x m3 parameter pairs as the Isaac G2 run (the RNG layout of
experiments/fractal_map.py is reproduced call for call), same success criterion, on the
corrected analytical closed loop (dynamics/closed_loop.py). Output schema matches
fractal_map.py, so tools/fractal_analyze.py reads it and outcomes can be compared pair by pair.

  run.cmd experiments/fractal_cpu.py --out results/fractal/R1_cpu_pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dynamics.closed_loop import NZ, ClosedLoop, SimCfg  # noqa: E402
from dynamics.policy import Actor  # noqa: E402
from pi_prove import L  # noqa: E402

CK = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup", "2026-09-11_22-27-54_CORR_s1", "model_800.pt")
STEPS = 3000
_A = {}


def layout(seed=20260914, pairs=600, half=0.6, eps_s="0.3,0.1,0.03,0.01,0.003,0.001"):
    rng = np.random.default_rng(seed)
    eps = [0.0] + [float(e) for e in eps_s.split(",")]
    rows = []
    for e in eps:
        base = np.stack([rng.uniform(-half, half, pairs), rng.uniform(-half, half, pairs)], 1)
        phi = rng.uniform(0, 2 * np.pi, pairs)
        rows += [base, base + e * np.stack([np.cos(phi), np.sin(phi)], 1)]
    return np.concatenate(rows, 0), eps


def episode(xy):
    if "a" not in _A:
        _A["a"] = Actor(CK)
    s1, s3 = float(np.exp(xy[0])), float(np.exp(xy[1]))
    cfg = SimCfg(mass_scale=(s1, 1.0, s3), inertia_scale=(s1, 1.0, s3))
    loop = ClosedLoop(_A["a"], cfg)
    z = np.zeros(NZ)
    z[1] = np.pi + 0.03
    z[2] = z[1] + 0.01
    z[3] = z[2] + 0.01
    up, held, hold_from = False, 0, int(0.75 * STEPS)
    for k in range(STEPS):
        z = loop.step(z)
        if abs(z[0]) > 0.6 or not np.all(np.isfinite(z)):
            return 0, 0.0, 1
        tip = float(L @ np.cos(z[1:4]) / L.sum())
        up |= tip > 0.9
        if k >= hold_from:
            held += tip > 0.9
    h = held / (STEPS - hold_from)
    return int(up and h > 0.95), round(h, 4), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    P, eps = layout()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(episode, P, chunksize=4))
    out = {"checkpoint": CK, "mode": "pairs", "implementation": "analytical CPU closed loop (no PhysX)",
           "params": P.tolist(), "meta": {"eps": eps, "pairs": 600},
           "success": [r[0] for r in res], "hold": [r[1] for r in res], "early": [r[2] for r in res]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("[cpu] %d episodes, success %.1f%% -> %s" % (len(res), 100 * np.mean(out["success"]), a.out))


if __name__ == "__main__":
    main()
