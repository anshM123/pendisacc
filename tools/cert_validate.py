"""Does the linear certificate hold in the nonlinear CPU closed loop?

Takes the common gain found for the widest box in results/dr_certificate/cert.json
and runs it, and the trained policy, from a perturbed upright on the 27 plants
{-w, 0, +w}^3 in log arm-mass. 12 s, with the real action clip. Also repeats with
8 ms transport delay and a 30% slower servo, which the certificate did not
include, to see how much margin it has.

  run.cmd tools/cert_validate.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dynamics.closed_loop import IA, IQ, IQD, NZ, ClosedLoop  # noqa: E402
from dynamics.policy import Actor, observation  # noqa: E402
from pi_prove import L, cfg_of, ckpt  # noqa: E402

STEPS, BOUND = 3000, 0.60
CERT = json.load(open(os.path.join(ROOT, "results", "dr_certificate", "cert.json"), encoding="utf-8"))
BOX = CERT["boxes"][-1]
W = BOX["half_width_log"]
_ACT = {}


class Linear:
    """a = K y on the same observation variables the policy uses (angles, not sin)."""

    def __init__(self, K):
        self.K = np.asarray(K)

    def __call__(self, obs):
        # obs = [x, xd, sin th(3), cos th(3), w(3), a_prev]; at upright sin th ~ th
        y = np.concatenate([[obs[0]], np.arcsin(np.clip(obs[2:5], -1, 1)), [obs[1]], obs[8:11], [obs[11]]])
        return np.array([[float(np.clip(self.K @ y, -1.0, 1.0))]])


def episode(job):
    who, g3, k, stress = job
    if who not in _ACT:
        _ACT[who] = Linear(BOX["K"]) if who == "certificate" else Actor(ckpt(CERT["policy"]))
    d = np.zeros(6)
    d[0:3] = g3
    cfg = cfg_of(d)
    if stress:
        cfg.drive.delay_steps = 2
        cfg.drive.tau *= 1.3
    loop = ClosedLoop(_ACT[who], cfg)
    rng = np.random.default_rng(900 + k)
    z = np.zeros(NZ)
    z[1:4] = rng.uniform(-0.03, 0.03, 3)
    z[5:8] = rng.uniform(-0.05, 0.05, 3)
    held = 0
    for s in range(STEPS):
        z = loop.step(z)
        if abs(z[0]) > BOUND or not np.all(np.isfinite(z)):
            return False
        if s >= 3 * STEPS // 4:
            held += float(L @ np.cos(z[1:4]) / L.sum()) > 0.9
    return held / (STEPS // 4) > 0.95


def main() -> int:
    pts = list(itertools.product((-W, 0.0, W), repeat=3))
    out = {"half_width_log": W}
    with ProcessPoolExecutor(max_workers=20) as pool:
        for stress in (False, True):
            for who in ("certificate", "policy"):
                jobs = [(who, p, k, stress) for p in pts for k in range(2)]
                ok = list(pool.map(episode, jobs, chunksize=2))
                pct = 100 * float(np.mean(ok))
                key = "%s_%s" % (who, "delay8ms_tau1.3" if stress else "as_certified")
                out[key] = pct
                print("%-12s %-18s holds upright on %5.1f%% of %d runs (27 plants x 2 ICs, arm masses x%.2f..x%.2f)"
                      % (who, "8ms delay, tau x1.3" if stress else "as certified", pct, len(jobs),
                         np.exp(-W), np.exp(W)), flush=True)
    with open(os.path.join(ROOT, "results", "dr_certificate", "validate.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
