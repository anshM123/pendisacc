"""Which fidelity measure predicts transfer?

Thirteen deployment simulators, each with an Isaac-measured success rate for the
same frozen policy, spanning two actuator MODEL FORMS:

    first-order lag   tau = 20, 50, 80, 100 ms
    second-order      omega_n = 13.4 ... 90 rad/s, zeta = 0.9

For each, compute what a practitioner could compute WITHOUT deploying, and ask
which correlates with the measured outcome:

  conventional, open-loop descriptors of the actuator alone
    d_tau_equiv    difference in equivalent time constant
    d_bandwidth    difference in -3 dB bandwidth
    d_rise         difference in 10-90% rise time
    d_phase        difference in phase lag at the plant's unstable frequency
    step_rmse      RMS difference of the actuator step responses

  policy-conditioned, from the closed loop
    D_SW           sum_k ||Phi(N,k+1) d_k||, model discrepancy weighted by how
                   much the trained closed loop amplifies it
    e_pred         the same sum with signs, i.e. predicted end-state error
    raw_gap        sum_k ||d_k||, the same discrepancy with NO weighting --
                   the control that isolates what the Phi weighting adds

Phi and the nominal trajectory depend only on the TRAINING simulator, so they
are computed once and reused for all thirteen deployments. That is the whole
practical argument for the measure: it costs one Jacobian sweep, after which
each candidate is nearly free.

Ground truth is Isaac. The predictors come from the standalone model, which is
rank-faithful but not calibrated (RESULTS.md section 5), so only RANK statistics
are reported.

  run.cmd experiments/predict_transfer.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import (  # noqa: E402
    IQ, NZ, ClosedLoop, DriveCfg, SimCfg, stability_weighted_gap, transition_matrices,
)
from dynamics.policy import Actor  # noqa: E402

CKPT = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                    "2026-09-05_22-45-54_rel3", "model_700.pt")
LAMBDA = 15.5378          # plant's dominant unstable eigenvalue [rad/s]
TAU_TRAIN = 0.100
ZETA = 0.9
HORIZON_S = 2.5           # covers swing-up and capture, where G peaks
N_IC = 8

# Isaac ground truth, 256 episodes each. Measured, not modelled.
CONDITIONS = [
    ("1st tau=20ms",  dict(order=1, tau=0.020), 0.016),
    ("1st tau=50ms",  dict(order=1, tau=0.050), 0.570),
    ("1st tau=80ms",  dict(order=1, tau=0.080), 0.938),
    ("1st tau=100ms", dict(order=1, tau=0.100), 1.000),
    ("2nd wn=13.4",   dict(order=2, wn=13.40), 0.000),
    ("2nd wn=18",     dict(order=2, wn=18.0), 0.000),
    ("2nd wn=22",     dict(order=2, wn=22.0), 0.066),
    ("2nd wn=25",     dict(order=2, wn=25.0), 0.090),
    ("2nd wn=28",     dict(order=2, wn=28.0), 0.457),
    ("2nd wn=32",     dict(order=2, wn=32.0), 0.344),
    ("2nd wn=40",     dict(order=2, wn=40.0), 0.383),
    ("2nd wn=60",     dict(order=2, wn=60.0), 0.180),
    ("2nd wn=90",     dict(order=2, wn=90.0), 0.016),
]


def cfg_of(spec) -> SimCfg:
    if spec["order"] == 1:
        return SimCfg(drive=DriveCfg(order=1, tau=spec["tau"]))
    return SimCfg(drive=DriveCfg(order=2, tau=1.0 / spec["wn"], zeta=ZETA))


# ------------------------------------------------- open-loop descriptors
def bw(spec) -> float:
    if spec["order"] == 1:
        return 1.0 / spec["tau"]
    wn, z = spec["wn"], ZETA
    return wn * np.sqrt(1 - 2 * z ** 2 + np.sqrt(4 * z ** 4 - 4 * z ** 2 + 2))


def rise(spec) -> float:
    if spec["order"] == 1:
        return spec["tau"] * np.log(9)
    return (2.16 * ZETA + 0.60) / spec["wn"]


def phase(spec) -> float:
    if spec["order"] == 1:
        return np.degrees(np.arctan(LAMBDA * spec["tau"]))
    wn, z = spec["wn"], ZETA
    return np.degrees(np.arctan2(2 * z * wn * LAMBDA, wn ** 2 - LAMBDA ** 2))


def step_response(spec, t) -> np.ndarray:
    if spec["order"] == 1:
        return 1.0 - np.exp(-t / spec["tau"])
    wn, z = spec["wn"], ZETA
    wd = wn * np.sqrt(max(1 - z ** 2, 1e-12))
    return 1.0 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z * wn / wd) * np.sin(wd * t))


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> int:
    actor = Actor(CKPT)
    d = np.load(os.path.join(ROOT, "results", "rollout_success.npz"), allow_pickle=True)
    dt = float(d["dt"])
    n = int(HORIZON_S / dt)

    rng = np.random.default_rng(3)
    Z0 = np.zeros((N_IC, NZ))
    for i in range(N_IC):
        Z0[i, IQ] = d["q"][0] + np.concatenate([[rng.uniform(-0.04, 0.04)],
                                                rng.uniform(-0.06, 0.06, 3)])
        Z0[i, 4:8] = np.concatenate([[rng.uniform(-0.04, 0.04)], rng.uniform(-0.12, 0.12, 3)])

    train_spec = dict(order=1, tau=TAU_TRAIN)
    loop_train = ClosedLoop(actor, cfg_of(train_spec))

    # Phi depends only on the TRAINING simulator: compute once, reuse for all.
    print("computing Phi along the nominal trajectory (%d initial conditions, %.1f s)..." % (N_IC, HORIZON_S))
    Zs, A = [], []
    for i in range(N_IC):
        z = loop_train.rollout(Z0[i], n)
        Zs.append(z)
        A.append(transition_matrices(loop_train, z))

    ts = np.arange(0.0, 0.6, dt)
    ref_step = step_response(train_spec, ts)

    rows = []
    for name, spec, success in CONDITIONS:
        loop_r = ClosedLoop(actor, cfg_of(spec))
        dsw = eprd = raw = 0.0
        for i in range(N_IC):
            g = stability_weighted_gap(loop_train, loop_r, Z0[i], n, A=A[i], Zs=Zs[i])
            dsw += g["D_SW"]; eprd += g["e_pred"]; raw += g["raw_gap"]
        dsw, eprd, raw = dsw / N_IC, eprd / N_IC, raw / N_IC

        tau_eq = spec.get("tau", 1.0 / spec.get("wn", 1.0))
        rows.append({
            "name": name, "success": success,
            "d_tau_equiv": abs(tau_eq - TAU_TRAIN),
            "d_bandwidth": abs(bw(spec) - bw(train_spec)),
            "d_rise": abs(rise(spec) - rise(train_spec)),
            "d_phase": abs(phase(spec) - phase(train_spec)),
            "step_rmse": float(np.sqrt(np.mean((step_response(spec, ts) - ref_step) ** 2))),
            "raw_gap": raw, "D_SW": dsw, "e_pred": eprd,
        })
        print("  %-14s success %5.1f%%   d_phase %5.1f   step_rmse %.4f   raw %8.3f   D_SW %10.3f"
              % (name, 100 * success, rows[-1]["d_phase"], rows[-1]["step_rmse"], raw, dsw))

    succ = np.array([r["success"] for r in rows])
    print("\nSpearman rank correlation with MEASURED success (n=%d conditions):" % len(rows))
    print("  (negative is what a good failure-predictor should show)")
    stats = {}
    for key, label in (("d_tau_equiv", "|tau - tau_train|"),
                       ("d_bandwidth", "|bandwidth error|"),
                       ("d_rise", "|rise-time error|"),
                       ("d_phase", "|phase error at lambda_max|"),
                       ("step_rmse", "actuator step-response RMSE"),
                       ("raw_gap", "raw model gap (unweighted)"),
                       ("e_pred", "predicted end-state error"),
                       ("D_SW", "D_SW (stability-weighted)")):
        v = np.array([r[key] for r in rows])
        rho = spearman(v, succ)
        stats[key] = rho
        mark = "  <-- policy-conditioned" if key in ("D_SW", "e_pred", "raw_gap") else ""
        print("  %-30s rho = %+.3f%s" % (label, rho, mark))

    out = os.path.join(ROOT, "results", "predict_transfer.json")
    json.dump({"conditions": rows, "spearman_vs_success": stats,
               "horizon_s": HORIZON_S, "n_initial_conditions": N_IC,
               "ground_truth": "Isaac, 256 episodes per condition"},
              open(out, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
