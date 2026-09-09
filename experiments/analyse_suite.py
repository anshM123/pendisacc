"""Score every predictor against the Isaac ground truth. The H2 test.

Kill criteria are fixed in PREREGISTRATION.md (commit d477e2a, before any label
in results/suite/ existed) and are applied here verbatim:

  1. R_TC must beat the best conventional baseline by >= 0.10 in |rho|
  2. that advantage must survive a condition-level bootstrap at P >= 0.80
  3. it must survive with each model family held out in turn
  4. R_TC must beat D_SW

Failing any one rejects H2, and the reported result becomes whichever predictor
wins.

The negative controls are the discriminating cases: four LARGE but measurably
harmless model errors (2.5x link mass, heavy joint damping, a force clamp cut
to a quarter, kv 150), all at 100% success. A predictor that tracks the size of
the model error must fail on them. They are reported separately for that reason.

  run.cmd experiments/analyse_suite.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import (  # noqa: E402
    IQ, NZ, ClosedLoop, DriveCfg, FrictionCfg, SimCfg,
    stability_weighted_gap, transfer_critical_risk, transition_matrices,
)
from dynamics.policy import Actor  # noqa: E402

CKPT = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                    "2026-09-05_22-45-54_rel3", "model_700.pt")
LAMBDA = 15.5378
TAU_TRAIN = 0.100
HORIZON_S = 2.5
N_IC = 8
DT = 1.0 / 250.0


def cfg_of(xi: dict) -> SimCfg:
    """Map a suite xi onto the standalone model. Same knobs, same meanings."""
    d = DriveCfg(
        order=int(xi.get("order", 1)),
        tau=float(xi.get("tau", TAU_TRAIN)),
        zeta=float(xi.get("zeta", 0.9)),
        omega_n=float(xi.get("omega_n", 0.0)),
        deadband=float(xi.get("deadband", 0.0)),
        delay_steps=int(round(float(xi.get("delay_s", 0.0)) / DT)),
        kv=float(xi.get("kv", 400.0)),
        f_clamp=float(xi.get("f_clamp", 349.5)),
    )
    jf, jd = float(xi.get("joint_friction", 0.0)), float(xi.get("joint_damping", 0.0))
    fr = FrictionCfg(model=("coulomb" if jf > 0 else ("viscous" if jd > 0 else "none")),
                     b_joint=jd, fc_joint=jf)
    ms = xi.get("mass_scale", [1.0, 1.0, 1.0])
    return SimCfg(drive=d, friction=fr,
                  mass_scale=tuple(ms), inertia_scale=tuple(ms),
                  cart_mass_scale=float(xi.get("cart_mass_scale", 1.0)),
                  gravity=float(xi.get("gravity", 9.80665)))


# ------------------------------------------------ open-loop actuator descriptors
def _bw(xi):
    if int(xi.get("order", 1)) == 1:
        return 1.0 / float(xi.get("tau", TAU_TRAIN))
    wn = float(xi.get("omega_n") or 1.0 / float(xi.get("tau", TAU_TRAIN)))
    z = float(xi.get("zeta", 0.9))
    return wn * np.sqrt(1 - 2 * z ** 2 + np.sqrt(4 * z ** 4 - 4 * z ** 2 + 2))


def _rise(xi):
    if int(xi.get("order", 1)) == 1:
        return float(xi.get("tau", TAU_TRAIN)) * np.log(9)
    wn = float(xi.get("omega_n") or 1.0 / float(xi.get("tau", TAU_TRAIN)))
    return (2.16 * float(xi.get("zeta", 0.9)) + 0.60) / wn


def _phase(xi):
    """Phase lag at the plant's unstable frequency, plus any dead time."""
    if int(xi.get("order", 1)) == 1:
        ph = np.degrees(np.arctan(LAMBDA * float(xi.get("tau", TAU_TRAIN))))
    else:
        wn = float(xi.get("omega_n") or 1.0 / float(xi.get("tau", TAU_TRAIN)))
        z = float(xi.get("zeta", 0.9))
        ph = np.degrees(np.arctan2(2 * z * wn * LAMBDA, wn ** 2 - LAMBDA ** 2))
    return ph + np.degrees(LAMBDA * float(xi.get("delay_s", 0.0)))


def _step(xi, t):
    if int(xi.get("order", 1)) == 1:
        y = 1.0 - np.exp(-t / float(xi.get("tau", TAU_TRAIN)))
    else:
        wn = float(xi.get("omega_n") or 1.0 / float(xi.get("tau", TAU_TRAIN)))
        z = float(xi.get("zeta", 0.9))
        wd = wn * np.sqrt(max(1 - z ** 2, 1e-12))
        y = 1.0 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z * wn / wd) * np.sin(wd * t))
    d = float(xi.get("delay_s", 0.0))
    return np.where(t < d, 0.0, y)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float((ra @ rb) / den) if den > 0 else 0.0


PREDICTORS = [
    ("d_tau", "|tau_eq - tau_train|", False),
    ("d_bw", "|bandwidth error|", False),
    ("d_rise", "|rise-time error|", False),
    ("d_phase", "|phase error at lambda_max|", False),
    ("step_rmse", "actuator step-response RMSE", False),
    ("traj_rmse", "short-window trajectory RMSE", False),
    ("raw_gap", "one-step model error (unweighted)", True),
    ("G_T", "G_T (amplification alone)", True),
    ("D_SW", "D_SW (stability-weighted)", True),
    ("R_TC", "R_TC (task-margin projected)", True),
]


def main() -> int:
    man = json.load(open(os.path.join(ROOT, "xi", "manifest.json"), encoding="utf-8"))
    actor = Actor(CKPT)
    ref = np.load(os.path.join(ROOT, "results", "rollout_success.npz"), allow_pickle=True)
    n = int(HORIZON_S / DT)

    rng = np.random.default_rng(3)
    Z0 = np.zeros((N_IC, NZ))
    for i in range(N_IC):
        Z0[i, IQ] = ref["q"][0] + np.concatenate([[rng.uniform(-0.04, 0.04)],
                                                  rng.uniform(-0.06, 0.06, 3)])
        Z0[i, 4:8] = np.concatenate([[rng.uniform(-0.04, 0.04)], rng.uniform(-0.12, 0.12, 3)])

    train_xi = {}
    loop_s = ClosedLoop(actor, cfg_of(train_xi))
    print("Phi along the TRAINING simulator's trajectory (computed once)...")
    Zs = [loop_s.rollout(Z0[i], n) for i in range(N_IC)]
    A = [transition_matrices(loop_s, Zs[i]) for i in range(N_IC)]

    ts = np.arange(0.0, 0.6, DT)
    ref_step = _step(train_xi, ts)
    nwin = int(0.4 / DT)

    rows = []
    for m in man:
        f = os.path.join(ROOT, "results", "suite", m["name"] + ".json")
        if not os.path.exists(f):
            continue
        success = json.load(open(f, encoding="utf-8"))["results"][0]["success_rate"]
        xi = m["xi"]
        loop_r = ClosedLoop(actor, cfg_of(xi))

        acc = {k: 0.0 for k in ("raw_gap", "G_T", "D_SW", "R_TC", "traj_rmse")}
        for i in range(N_IC):
            g = stability_weighted_gap(loop_s, loop_r, Z0[i], n, A=A[i], Zs=Zs[i])
            t = transfer_critical_risk(loop_s, loop_r, Z0[i], n, A=A[i], Zs=Zs[i])
            Zr = loop_r.rollout(Z0[i], nwin)
            acc["raw_gap"] += g["raw_gap"]; acc["G_T"] += g["G_T"]
            acc["D_SW"] += g["D_SW"]; acc["R_TC"] += t["R_TC"]
            acc["traj_rmse"] += float(np.sqrt(np.mean(
                np.sum((Zr[:, :8] - Zs[i][:nwin + 1, :8]) ** 2, axis=1))))
        for k in acc:
            acc[k] /= N_IC

        tau_eq = float(xi.get("tau", TAU_TRAIN)) if int(xi.get("order", 1)) == 1 \
            else 1.0 / float(xi.get("omega_n") or 1.0 / float(xi.get("tau", TAU_TRAIN)))
        rows.append({
            "name": m["name"], "family": m["family"], "plausible": m["plausible"],
            "success": success,
            "d_tau": abs(tau_eq - TAU_TRAIN),
            "d_bw": abs(_bw(xi) - _bw(train_xi)),
            "d_rise": abs(_rise(xi) - _rise(train_xi)),
            "d_phase": abs(_phase(xi) - _phase(train_xi)),
            "step_rmse": float(np.sqrt(np.mean((_step(xi, ts) - ref_step) ** 2))),
            **acc,
        })
        print("  %-20s success %5.1f%%  R_TC %8.3f  D_SW %9.1f  d_phase %5.1f"
              % (m["name"], 100 * success, acc["R_TC"], acc["D_SW"], rows[-1]["d_phase"]))

    succ = np.array([r["success"] for r in rows])
    print("\n%d conditions\n" % len(rows))
    print("Spearman rho vs measured success (negative = predicts failure):")
    rho = {}
    for key, label, cond in PREDICTORS:
        v = np.array([r[key] for r in rows])
        rho[key] = spearman(v, succ)
        print("  %-36s %+.3f%s" % (label, rho[key], "   [policy-conditioned]" if cond else ""))

    base_keys = [k for k, _, c in PREDICTORS if not c]
    best_base = max(base_keys, key=lambda k: abs(rho[k]))
    print("\nbest conventional baseline: %s (|rho| = %.3f)" % (best_base, abs(rho[best_base])))
    print("R_TC |rho| = %.3f, D_SW |rho| = %.3f" % (abs(rho["R_TC"]), abs(rho["D_SW"])))

    # ---- kill criteria, verbatim from the pre-registration ----
    B = 20000
    rr = np.random.default_rng(0)
    win_base, win_dsw = [], []
    for _ in range(B):
        i = rr.integers(0, len(rows), len(rows))
        if len(set(succ[i])) < 3:
            continue
        a = abs(spearman(np.array([rows[j]["R_TC"] for j in i]), succ[i]))
        b = abs(spearman(np.array([rows[j][best_base] for j in i]), succ[i]))
        c = abs(spearman(np.array([rows[j]["D_SW"] for j in i]), succ[i]))
        win_base.append(a - b); win_dsw.append(a - c)
    win_base, win_dsw = np.array(win_base), np.array(win_dsw)

    c1 = abs(rho["R_TC"]) - abs(rho[best_base]) >= 0.10
    c2 = (win_base > 0).mean() >= 0.80
    c4 = abs(rho["R_TC"]) > abs(rho["D_SW"])

    fam_ok, fam_detail = True, []
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] != fam]
        if len(sub) < 8:
            continue
        s2 = np.array([r["success"] for r in sub])
        a = abs(spearman(np.array([r["R_TC"] for r in sub]), s2))
        b = abs(spearman(np.array([r[best_base] for r in sub]), s2))
        fam_detail.append((fam, a, b))
        if a - b < 0.10:
            fam_ok = False
    c3 = fam_ok

    print("\nKILL CRITERIA (PREREGISTRATION.md, frozen at commit d477e2a):")
    print("  1. beats best baseline by >= 0.10 in |rho|      : %s  (%+.3f)"
          % ("PASS" if c1 else "FAIL", abs(rho["R_TC"]) - abs(rho[best_base])))
    print("  2. bootstrap P(R_TC better) >= 0.80             : %s  (P = %.2f)"
          % ("PASS" if c2 else "FAIL", (win_base > 0).mean()))
    print("  3. survives every family held out               : %s" % ("PASS" if c3 else "FAIL"))
    for fam, a, b in fam_detail:
        print("       without %-12s R_TC %.3f vs %-10s %.3f  (%+.3f)" % (fam, a, best_base, b, a - b))
    print("  4. beats D_SW                                   : %s  (P = %.2f)"
          % ("PASS" if c4 else "FAIL", (win_dsw > 0).mean()))
    verdict = c1 and c2 and c3 and c4
    print("\n  H2 %s" % ("SUPPORTED" if verdict else "REJECTED -- report the winner instead"))
    if not verdict:
        win = max(rho, key=lambda k: abs(rho[k]))
        print("  best predictor overall: %s (rho = %+.3f)" % (win, rho[win]))

    print("\nNEGATIVE CONTROLS (large model error, measured harmless):")
    print("  %-20s %-9s %-10s %-10s %-10s" % ("condition", "success", "raw_gap", "D_SW", "R_TC"))
    ctrl = [r for r in rows if r["family"] == "control"]
    other = [r for r in rows if r["family"] != "control" and r["success"] < 0.5]
    for r in ctrl:
        print("  %-20s %6.1f%%   %9.1f %10.1f %10.3f"
              % (r["name"], 100 * r["success"], r["raw_gap"], r["D_SW"], r["R_TC"]))
    if ctrl and other:
        for key in ("raw_gap", "D_SW", "R_TC"):
            cm = np.median([r[key] for r in ctrl]); om = np.median([r[key] for r in other])
            print("  median %-8s controls %10.3f vs failing conditions %10.3f  -> %s"
                  % (key, cm, om, "separates" if cm < om else "*** DOES NOT SEPARATE ***"))

    json.dump({"rows": rows, "spearman": rho, "best_baseline": best_base,
               "kill_criteria": {"beats_baseline_by_0.10": bool(c1),
                                 "bootstrap_P": float((win_base > 0).mean()),
                                 "family_holdout": bool(c3), "beats_D_SW": bool(c4)},
               "H2_supported": bool(verdict)},
              open(os.path.join(ROOT, "results", "h2_test.json"), "w"), indent=1)
    print("\n[out] results/h2_test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
