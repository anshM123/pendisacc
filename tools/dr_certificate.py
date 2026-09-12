"""Can domain randomisation over the arm-mass ratios succeed AT ALL? Decide on CPU.

Argument. The policy is memoryless over y = [cart, cart_vel, link angles, link
rates, previous action] (sin/cos of an angle is linear in it at upright). Any
such policy, however nonlinear, has a local linearisation a = K y at the upright.
The upright is locally asymptotically stable on plant p iff rho(A_p + B_p K C) < 1.
So a policy can hold the upright on EVERY plant in a set S only if one static
gain K stabilises every linearisation in S simultaneously.

  * finding such a K is a certificate that DR over S is not ruled out;
  * failing to find one (multi-start + cutting-plane) is evidence -- not proof,
    the problem is nonconvex -- that no memoryless policy can be robust over S,
    so randomising over S must either fail to train or give up on some of it.

Similarity is what makes this computable. Plants depend only on the
dimensionless groups (tools/pi_prove.py: exact to 1e-11), and the drive groups
do not affect the frozen policy, so S is a 3-D box in log(m_i/m_cart) rather
than a 6-D box in raw parameters.

Also reports the trained policy's own local gain and which grid plants it
stabilises, checked against its nonlinear CPU success (results/PI_prove).

  run.cmd tools/dr_certificate.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dynamics.closed_loop import IA, IQ, IQD, NZ, ClosedLoop  # noqa: E402
from dynamics.policy import Actor, observation  # noqa: E402
from pi_prove import cfg_of, ckpt  # noqa: E402

YIDX = list(range(IQ.start, IQ.stop)) + list(range(IQD.start, IQD.stop)) + [IA]
C = np.zeros((len(YIDX), NZ))
C[np.arange(len(YIDX)), YIDX] = 1.0
GRID = np.linspace(-0.5, 0.5, 9)          # log(m_i / m_i,nominal), spacing 0.125
WIDTHS = (0.125, 0.25, 0.375, 0.5)
POLICY = "CORR_s1"


class Const:
    def __init__(self, a):
        self.a = a

    def __call__(self, _obs):
        return np.array([[self.a]])


def linearise(d):
    """Open-loop A (NZ x NZ) and B (NZ,) at the upright for plant d (6 log coords)."""
    cfg = cfg_of(d)
    z0 = np.zeros(NZ)
    loop = ClosedLoop(Const(0.0), cfg)
    eps = 1e-6
    A = np.empty((NZ, NZ))
    for i in range(NZ):
        dz = np.zeros(NZ)
        dz[i] = eps
        A[:, i] = (loop.step(z0 + dz) - loop.step(z0 - dz)) / (2 * eps)
    lp, lm = ClosedLoop(Const(eps), cfg, model=loop.model), ClosedLoop(Const(-eps), cfg, model=loop.model)
    B = (lp.step(z0) - lm.step(z0)) / (2 * eps)
    return A, B


def plant(g3):
    d = np.zeros(6)
    d[0:3] = g3
    A, B = linearise(d)
    return tuple(g3), A, B


def policy_gain():
    act = Actor(ckpt(POLICY))
    z0 = np.zeros(NZ)

    def a(z):
        return float(act(observation(z[IQ], z[IQD], z[IA]))[0, 0])

    K = np.empty(len(YIDX))
    for j, i in enumerate(YIDX):
        dz = np.zeros(NZ)
        dz[i] = 1e-5
        K[j] = (a(z0 + dz) - a(z0 - dz)) / 2e-5
    return K, a(z0)


def rho(A, B, K):
    return float(np.max(np.abs(np.linalg.eigvals(A + np.outer(B, K) @ C))))


def worst(K, plants):
    return max(rho(A, B, K) for _, A, B in plants)


def search(args):
    """Minimise max spectral radius over an active set; cutting-plane on the box."""
    K0, box, seed, w = args
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(K0), 0.05)
    x0 = K0 / scale * (1.0 + (0.3 * rng.standard_normal(len(K0)) if seed else 0.0))
    # start from the box vertices and centre; add violators one at a time
    active = [p for p in box if all(abs(abs(v) - w) < 1e-9 for v in p[0])
              or all(abs(v) < 1e-9 for v in p[0])]
    keys = {p[0] for p in active}
    for _ in range(6):
        f = lambda x: worst(x * scale, active)  # noqa: E731
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"maxfev": 6000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True})
        r = minimize(f, r.x, method="Powell", options={"maxfev": 6000, "xtol": 1e-6, "ftol": 1e-8})
        x0 = r.x
        rs = [rho(A, B, x0 * scale) for _, A, B in box]
        j = int(np.argmax(rs))
        if rs[j] <= f(x0) + 1e-9 or box[j][0] in keys:
            break
        active.append(box[j])
        keys.add(box[j][0])
    K = x0 * scale
    return float(worst(K, box)), K.tolist(), seed


def main() -> int:
    K_pol, a0 = policy_gain()
    print("policy %s: local gain K = %s, a(upright) = %+.4f" % (POLICY, np.round(K_pol, 3), a0),
          flush=True)
    pts = [np.array(p) for p in itertools.product(GRID, GRID, GRID)]
    with ProcessPoolExecutor(max_workers=20) as pool:
        plants = list(pool.map(plant, pts, chunksize=4))
        print("linearised %d plants" % len(plants), flush=True)

        # the trained policy's own local robustness, and a check against nonlinear sims
        pol_rho = {p[0]: rho(p[1], p[2], K_pol) for p in plants}
        nominal = pol_rho[(0.0, 0.0, 0.0)]
        print("policy rho at nominal %.5f (%s)" % (nominal, "stable" if nominal < 1 else "UNSTABLE"),
              flush=True)
        check = []
        for i in range(3):
            for dl in (-0.375, 0.375):
                k = [0.0, 0.0, 0.0]
                k[i] = dl
                check.append({"link": i + 1, "dlog": dl, "rho": pol_rho[tuple(k)]})
                print("  policy, link%d x%.2f alone: rho %.4f" % (i + 1, np.exp(dl), pol_rho[tuple(k)]),
                      flush=True)
        frac = float(np.mean([v < 1 for v in pol_rho.values()]))
        print("policy locally stabilises %.1f%% of the +-0.5 log grid" % (100 * frac), flush=True)

        out = {"policy": POLICY, "K_policy": K_pol.tolist(), "a_upright": a0,
               "policy_rho_nominal": nominal, "policy_single_link_checks": check,
               "policy_stable_fraction_grid": frac, "grid": GRID.tolist(), "boxes": []}

        for w in WIDTHS:
            box = [p for p in plants if max(abs(v) for v in p[0]) <= w + 1e-9]
            starts = [(K_pol, box, s, w) for s in range(10)]
            res = sorted(pool.map(search, starts), key=lambda r: r[0])
            best, K, _ = res[0]
            verdict = "FEASIBLE (certificate)" if best < 1.0 else "no common gain found"
            out["boxes"].append({"half_width_log": w, "factor_range": [float(np.exp(-w)), float(np.exp(w))],
                                 "n_plants": len(box), "best_worst_rho": best, "K": K,
                                 "policy_worst_rho": max(pol_rho[p[0]] for p in box),
                                 "all_start_values": [r[0] for r in res], "verdict": verdict})
            print("box +-%.3f (x%.2f..x%.2f, %d plants): best common-gain worst rho %.5f -> %s | "
                  "trained policy worst rho %.4f"
                  % (w, np.exp(-w), np.exp(w), len(box), best, verdict,
                     out["boxes"][-1]["policy_worst_rho"]), flush=True)

    os.makedirs(os.path.join(ROOT, "results", "dr_certificate"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "dr_certificate", "cert.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/dr_certificate/cert.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
