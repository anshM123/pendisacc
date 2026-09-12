"""System identification in raw vs similarity coordinates (PREREGISTRATION_PI.md, section 3).

CPU only, policy-free. The true plant is a random draw; excitation is an
open-loop commanded-velocity sequence; measurements are cart position and the
three absolute link angles with sensor-level noise.

Every parameter fitted here carries mass dimension (link masses and inertias,
cart mass, velocity-loop gain k_v, joint viscous damping), so the mass-unit
similarity generator is the all-ones direction in their log-coordinates. The
force clamp would break that, so it is removed and the largest force the
excitation demands of the TRUE plant is recorded against the real 349.5 N clamp.

Estimators, both nonlinear least squares on trajectory error:
  raw        all 9 log-parameters, started from a prior draw (a practitioner
             with roughly +-30% prior knowledge)
  similarity 8 coordinates, gauge fixed at log m_cart = 0, i.e. the plant
             modulo similarity, started from the same draw projected

Outputs results/PI_sysid/sysid.json.

  run.cmd experiments/pi_sysid.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import (IQD, IV, NZ, ClosedLoop, DriveCfg,  # noqa: E402
                                  FrictionCfg, SimCfg)

NAMES = ["m1", "m2", "m3", "I1", "I2", "I3", "m_cart", "kv", "b_joint"]
P = len(NAMES)
GAUGE = NAMES.index("m_cart")
FREE = [i for i in range(P) if i != GAUGE]
GEN = np.ones(P) / np.sqrt(P)          # mass-unit similarity generator
KV0, B0 = 400.0, 0.002
DT, STEPS = 1.0 / 250.0, 375           # 1.5 s at the control rate
SIG_TH, SIG_X = 3.8e-4, 1.0e-4         # MT6701 14-bit; cart encoder
NS = (1, 2, 4, 8, 16)
TRUTHS = 8
PRIOR = 0.30


class OpenLoop:
    """Plays a fixed commanded-velocity sequence; ignores the observation."""

    def __init__(self, v):
        self.a, self.k = v / DriveCfg().action_scale, 0

    def __call__(self, _obs):
        a = self.a[min(self.k, len(self.a) - 1)]
        self.k += 1
        return np.array([[a]])


def excitation(rng):
    # piecewise-constant holds of 40..120 ms, +-1.2 m/s, plus a slow sine
    v, k = np.zeros(STEPS), 0
    while k < STEPS:
        h = int(rng.integers(10, 31))
        v[k:k + h] = rng.uniform(-1.2, 1.2)
        k += h
    t = np.arange(STEPS) * DT
    return v + 0.4 * np.sin(2 * np.pi * rng.uniform(0.5, 2.5) * t)


def cfg_of(theta, clamp=1e9):
    e = np.exp(theta)
    return SimCfg(mass_scale=tuple(e[0:3]), inertia_scale=tuple(e[3:6]),
                  cart_mass_scale=float(e[6]),
                  drive=DriveCfg(kv=KV0 * float(e[7]), f_clamp=clamp),
                  friction=FrictionCfg(model="viscous", b_joint=B0 * float(e[8])))


def z0_hanging(rng):
    z = np.zeros(NZ)
    z[1:4] = np.pi + rng.uniform(-0.02, 0.02, 3)
    return z


def simulate(theta, exc, z0s):
    cfg = cfg_of(theta)
    return [ClosedLoop(OpenLoop(v), cfg).rollout(z0, STEPS)[1:, 0:4]
            for v, z0 in zip(exc, z0s)]


def peak_force(theta, exc, z0s):
    """Largest |F| the true plant's velocity loop demands, with no clamp."""
    cfg = cfg_of(theta)
    peak = 0.0
    for v, z0 in zip(exc, z0s):
        loop, z = ClosedLoop(OpenLoop(v), cfg), z0.copy()
        for _ in range(STEPS):
            z = loop.step(z)
            peak = max(peak, abs(cfg.drive.kv * float(z[IV] - np.atleast_1d(z[IQD])[0])))
    return peak


def residual(theta, exc, z0s, meas):
    r = []
    for s, m in zip(simulate(theta, exc, z0s), meas):
        r.append((s[:, 0] - m[:, 0]) / SIG_X)
        r.append(((s[:, 1:] - m[:, 1:]) / SIG_TH).ravel())
    return np.concatenate(r)


def draw_truth(rng):
    r = rng.normal(0.0, 0.25, 3)
    r -= r.mean()
    s = rng.uniform(np.log(0.6), np.log(1.6))
    th = np.zeros(P)
    th[0:3] = r + s
    th[3:6] = r + s + rng.normal(0.0, 0.10, 3)
    th[6] = rng.uniform(np.log(0.85), np.log(1.25))
    th[7] = rng.normal(0.0, 0.35)
    th[8] = rng.normal(0.0, 0.30)
    return th


def fisher(theta, exc, z0s, eps=1e-4):
    """J^T J of the noise-normalised residual at theta, central differences."""
    clean = simulate(theta, exc, z0s)
    cols = []
    for j in range(P):
        d = np.zeros(P)
        d[j] = eps
        cols.append((residual(theta + d, exc, z0s, clean)
                     - residual(theta - d, exc, z0s, clean)) / (2 * eps))
    J = np.column_stack(cols)
    return J.T @ J


def quotient(theta):
    """Gauge-invariant coordinates: every log-parameter minus log m_cart."""
    return np.delete(theta - theta[GAUGE], GAUGE)


def one_truth(t):
    rng = np.random.default_rng(1000 + t)
    truth = draw_truth(rng)
    nmax = max(NS)
    exc = [excitation(rng) for _ in range(nmax)]
    z0s = [z0_hanging(rng) for _ in range(nmax)]
    meas = [c + np.column_stack([rng.normal(0, SIG_X, STEPS),
                                 rng.normal(0, SIG_TH, (STEPS, 3))])
            for c in simulate(truth, exc, z0s)]
    hexc = [excitation(rng) for _ in range(4)]
    hz0 = [z0_hanging(rng) for _ in range(4)]
    hclean = simulate(truth, hexc, hz0)

    def heldout(theta):
        s = simulate(theta, hexc, hz0)
        return float(np.sqrt(np.mean([np.mean((a[:, 1:] - b[:, 1:]) ** 2)
                                      for a, b in zip(s, hclean)])))

    start = rng.normal(0.0, PRIOR, P)            # prior draw around nominal
    F = fisher(truth, exc[:4], z0s[:4])
    w, V = np.linalg.eigh(F)
    row = {"truth": truth.tolist(), "start": start.tolist(),
           "peak_force_N": peak_force(truth, exc, z0s),
           "fisher_eigs": w.tolist(),
           "fisher_min_eigvec_cos_to_generator": float(abs(V[:, 0] @ GEN)),
           "fisher_second_eig_over_min": float(w[1] / max(abs(w[0]), 1e-300)),
           "fits": []}
    for n in NS:
        args = (exc[:n], z0s[:n], meas[:n])
        raw = least_squares(residual, start, args=args, method="trf",
                            x_scale=0.1, diff_step=1e-4, max_nfev=60)

        def res_sim(x):
            th = np.zeros(P)
            th[FREE] = x
            return residual(th, *args)

        sim = least_squares(res_sim, quotient(start), method="trf",
                            x_scale=0.1, diff_step=1e-4, max_nfev=60)
        th_sim = np.zeros(P)
        th_sim[FREE] = sim.x
        f = {"N": n,
             "raw_theta": raw.x.tolist(), "sim_theta": th_sim.tolist(),
             "raw_abs_err": (raw.x - truth).tolist(),
             "raw_err_along_generator": float((raw.x - truth) @ GEN),
             "raw_quotient_err": (quotient(raw.x) - quotient(truth)).tolist(),
             "sim_quotient_err": (quotient(th_sim) - quotient(truth)).tolist(),
             "raw_heldout_rmse_rad": heldout(raw.x),
             "sim_heldout_rmse_rad": heldout(th_sim),
             "raw_nfev": int(raw.nfev), "sim_nfev": int(sim.nfev)}
        row["fits"].append(f)
        print("truth %d N=%2d  heldout raw %.2e sim %.2e | raw err along gen %+.3f"
              % (t, n, f["raw_heldout_rmse_rad"], f["sim_heldout_rmse_rad"],
                 f["raw_err_along_generator"]), flush=True)
    return row


def main() -> int:
    out = os.path.join(ROOT, "results", "PI_sysid")
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=TRUTHS) as ex:
        rows = list(ex.map(one_truth, range(TRUTHS)))
    with open(os.path.join(out, "sysid.json"), "w", encoding="utf-8") as fh:
        json.dump({"names": NAMES, "generator": GEN.tolist(), "Ns": list(NS),
                   "sigma_theta": SIG_TH, "sigma_x": SIG_X, "steps": STEPS,
                   "prior_sigma": PRIOR, "real_clamp_N": 349.5, "truths": rows},
                  fh, indent=1)
    print("wrote", os.path.join(out, "sysid.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
