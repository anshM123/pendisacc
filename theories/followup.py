"""Round-2 theories. PRE-REGISTRATION: committed before any of these is computed.

Registered after reading the first screen (results/theories/screen.json). What
that screen showed, and why each follow-up exists:

  * F01 passed with a 42:1 gap between the first two singular values of the
    margin gradient: fragility appears to live in ONE direction of parameter
    space. That was seen for one policy only, and the direction itself was not
    inspected. F04-F06 ask whether it is a property of the robot (shared across
    policies), whether it predicts Isaac, and whether a free-swing test on the
    real arms can measure it (E11).
  * A14 passed (min closed-loop damping ratio ranks R*, Spearman 0.69, n=24)
    among 13 transfer theories that mostly failed. A20 replicates it on the
    fresh Isaac confirmation suite. A21 replicates the failed A01 there too,
    so a lucky pass and an unlucky fail are treated the same way.
  * B03 passed (AUC 0.95) but may be trivial: early checkpoints that cannot
    swing up are both unsuccessful and unstable. B06 removes them.
  * C01 passed with a threshold too loose to mean anything (the certified
    robust gains also satisfy it). C11 is the version that would mean
    something.
  * E01-E03 failed because hanging frequencies are invariant to a common scale
    of the arm masses (pinned cart). That is similarity again, not a bug. E11
    asks whether what they CAN see is what matters.

Isaac-dependent theories (A20, A21, F06) read results/confirm_theory/*.json,
produced by tools/confirm_theory_queue.sh from the suite frozen in
xi/confirm_theory before the first screen's results were examined.

  run.cmd theories/followup.py --stage cpu|isaac
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import screen as S  # noqa: E402

CONF = os.path.join(S.ROOT, "xi", "confirm_theory")
CRES = os.path.join(S.ROOT, "results", "confirm_theory")
COORD8 = ["m1", "m2", "m3", "I1", "I2", "I3", "mc", "kv"]


def spec8(x):
    e = np.exp(np.asarray(x, float))
    return {"mass_scale": [float(e[0]), float(e[1]), float(e[2])],
            "inertia_scale": [float(e[3]), float(e[4]), float(e[5])],
            "cart_mass_scale": float(e[6]), "kv": 400.0 * float(e[7])}


@lru_cache(maxsize=None)
def u_star(ck, asset="corrected", n=80, seed=3):
    """Dominant direction of the margin gradient over a +-0.3 log box in 8 coordinates."""
    K = S.K_of(ck)
    rng = np.random.default_rng(seed)
    G = []
    for _ in range(n):
        x = rng.uniform(-0.3, 0.3, 8)
        G.append([(S.rho_of(K, spec8(x + 1e-4 * np.eye(8)[j]), asset)
                   - S.rho_of(K, spec8(x - 1e-4 * np.eye(8)[j]), asset)) / 2e-4 for j in range(8)])
    _, sv, Vt = np.linalg.svd(np.array(G), full_matrices=False)
    u = Vt[0] * np.sign(Vt[0][np.argmax(np.abs(Vt[0]))])
    return tuple(u), tuple(sv)


def confirm_runs():
    return [l.strip() for l in open(os.path.join(CONF, "run_list.txt"), encoding="utf-8") if l.strip()]


def final_ck(run):
    return sorted(glob.glob(os.path.join(run, "model_*.pt")),
                  key=lambda f: int("".join(c for c in os.path.basename(f) if c.isdigit())))[-1]


def cpu_stable(run):
    return S.rho_of(S.K_of(final_ck(run)), {}, "corrected") < 1


def isaac_table():
    """{condition: {run_basename: success}} from the confirmation suite; None if incomplete."""
    man = json.load(open(os.path.join(CONF, "manifest.json"), encoding="utf-8"))
    names = ["nominal"] + [c["name"] for c in man["conditions"]]
    tab = {}
    for n in names:
        p = os.path.join(CRES, n + ".json")
        if not os.path.exists(p):
            return None, man
        tab[n] = {r["run"]: r["success_rate"] for r in json.load(open(p, encoding="utf-8"))["results"]}
    return tab, man


def xi_spec(xi):
    return {k: v for k, v in xi.items() if k in ("mass_scale", "cart_mass_scale", "tau", "delay_s", "kv")}


REG = []


def theory(tid, stage, claim, stat, op, thr):
    def deco(fn):
        REG.append({"id": tid, "stage": stage, "claim": claim, "stat": stat, "op": op, "thr": thr, "fn": fn.__name__})
        return fn
    return deco


# ---------------------------------------------------------------- CPU stage
@theory("F04", "cpu", "The fragile direction is a property of the robot: mean |cos| between u* of CPU-stable confirmation policies >= 0.9", "mean |cos|", ">=", 0.90)
def F04():
    us = [np.array(u_star(final_ck(r))[0]) for r in confirm_runs() if cpu_stable(r)]
    cs = [abs(float(a @ b)) for a, b in itertools.combinations(us, 2)]
    return (float(np.mean(cs)) if cs else float("nan")), len(us), {"min_cos": float(min(cs)) if cs else None,
                                                                  "u_first": [round(x, 3) for x in us[0]] if us else None}


@theory("F05", "cpu", "HOLDOUT (existing Isaac T1): -|x . u*| classifies success over 26 random 8-D directions, AUC >= 0.8", "AUC", ">=", 0.80)
def F05():
    t = S.J("results/T1/nullspace.json")
    ck = os.path.join(S.LOGS, "2026-09-11_19-37-34_CORR_s2", "model_999.pt")
    u = np.array(u_star(ck)[0])
    x = [np.log(np.array(r["factors"], float)) for r in t["rows"]]
    lb = [r["success"] >= 0.5 for r in t["rows"]]
    a = S.auc([-abs(float(xi @ u)) for xi in x], lb)
    raw = S.auc([-float(np.linalg.norm(xi)) for xi in x], lb)
    return a, len(lb), {"raw_norm_auc": raw, "u": [round(v, 3) for v in u]}


def _hang_logf8(x):
    e = np.exp(np.asarray(x, float))
    return np.log(S.hang((e[0], e[1], e[2], e[3], e[4], e[5], e[6]))[0])


@theory("E11", "cpu", "A free-swing test sees the fragile direction: |P_row(J_hang) u*| >= 0.95 (u* of CORR_s1)", "projection", ">=", 0.95)
def E11():
    u = np.array(u_star(E.checkpoints("CORR_s1")[-1])[0])
    Jh = np.column_stack([(_hang_logf8(1e-4 * np.eye(8)[j]) - _hang_logf8(-1e-4 * np.eye(8)[j])) / 2e-4 for j in range(8)])
    P = np.linalg.pinv(Jh) @ Jh
    return float(np.linalg.norm(P @ u)), 8, {"rank_Jhang": int(np.linalg.matrix_rank(Jh, tol=1e-6)),
                                             "u": [round(v, 3) for v in u]}


@theory("B06", "cpu", "Non-trivial B03: among checkpoints that DO reach upright, own-plant margin classifies own-sim success, AUC >= 0.8", "AUC", ">=", 0.80)
def B06():
    sc, lb = [], []
    for p in S.D1():
        K_plant = p["own"]
        for r in p["select"]:
            if r.get("ever_reached_upright", 0.0) < 0.9:
                continue
            ck = os.path.join(p["run"], r["checkpoint"])
            if os.path.exists(ck):
                sc.append(S.margin(S.K_of(ck), K_plant, "original"))
                lb.append(r["success_rate"] >= 0.5)
    return S.auc(sc, lb), len(sc), {"n_pos": int(sum(lb))}


@theory("C11", "cpu", "RL balances with less damping than a certified robust gain: >=80% of CPU-stable corrected policies below it", "fraction", ">=", 0.80)
def C11():
    v2 = json.load(open(os.path.join(S.ROOT, "results", "dr_certificate", "cert_v2.json"), encoding="utf-8"))
    zc = S.min_damping(np.array(v2["boxes"][1]["K"]), {}, "corrected")
    zs = [S.min_damping(S.K_of(final_ck(r)), {}, "corrected") for r in confirm_runs() if cpu_stable(r)]
    return float(np.mean([z < zc for z in zs])), len(zs), {"certified_min_damping": zc, "policies": zs}


# ---------------------------------------------------------------- Isaac stage
@theory("A20", "isaac", "REPLICATION of A14 on fresh Isaac data: min damping ratio ranks policies' mean confirmation-suite success", "spearman", ">=", 0.50)
def A20():
    tab, man = isaac_table()
    if tab is None:
        return float("nan"), 0, {"pending": True}
    runs = confirm_runs()
    conds = [c["name"] for c in man["conditions"]]
    z = [S.min_damping(S.K_of(final_ck(r)), {}, "corrected") for r in runs]
    y = [np.mean([tab[c].get(os.path.basename(r), np.nan) for c in conds]) for r in runs]
    return S.spear(z, y), len(runs), {"damping": z, "mean_success": y}


@theory("A21", "isaac", "REPLICATION of A01 on fresh Isaac data: exact-condition CPU margin classifies success>=50%", "AUC", ">=", 0.80)
def A21():
    tab, man = isaac_table()
    if tab is None:
        return float("nan"), 0, {"pending": True}
    sc, lb = [], []
    for c in man["conditions"]:
        for r in confirm_runs():
            s = tab[c["name"]].get(os.path.basename(r))
            if s is None:
                continue
            sc.append(S.margin(S.K_of(final_ck(r)), xi_spec(c["xi"]), "corrected"))
            lb.append(s >= 0.5)
    return S.auc(sc, lb), len(sc), {"n_pos": int(sum(lb))}


@theory("F06", "isaac", "Fresh Isaac: -|d . u*_p| classifies success on mass conditions (AUC >= 0.8) and beats raw distance by >= 0.10", "AUC", ">=", 0.80)
def F06():
    tab, man = isaac_table()
    if tab is None:
        return float("nan"), 0, {"pending": True}
    sc, raw, lb = [], [], []
    for c in man["conditions"]:
        if "mass_scale" not in c["xi"]:
            continue
        lm = np.log(np.array(c["xi"]["mass_scale"], float))
        d = np.concatenate([lm, lm, [0.0, 0.0]])
        for r in confirm_runs():
            s = tab[c["name"]].get(os.path.basename(r))
            if s is None:
                continue
            u = np.array(u_star(final_ck(r))[0])
            sc.append(-abs(float(d @ u)))
            raw.append(-float(np.linalg.norm(lm)))
            lb.append(s >= 0.5)
    a, ar = S.auc(sc, lb), S.auc(raw, lb)
    ok = a if (a == a and ar == ar and a - ar >= 0.10) else min(a, 0.0) if a == a else a
    return ok, len(sc), {"auc_u": a, "auc_raw": ar, "n_pos": int(sum(lb))}


def run_one(tid):
    rec = dict(next(r for r in REG if r["id"] == tid))
    t0 = time.time()
    try:
        v, n, det = globals()[rec["fn"]]()
        rec.update(value=float(v), n=int(n), detail=det, passed=S.passed(float(v), rec["op"], rec["thr"]))
    except Exception:
        rec.update(value=None, n=0, detail=traceback.format_exc()[-1500:], passed=False, error=True)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cpu", "isaac"), required=True)
    args = ap.parse_args()
    ids = [r["id"] for r in REG if r["stage"] == args.stage]
    E.basis("original")
    E.basis("corrected")
    out = []
    with ProcessPoolExecutor(max_workers=min(8, len(ids))) as ex:
        for fut in as_completed([ex.submit(run_one, i) for i in ids]):
            r = fut.result()
            out.append(r)
            print("%-4s %-5s %-4s %-12s %s %-6s value %-10s n=%-4s %5.0fs  %s" % (
                r["id"], r["stage"], "PASS" if r["passed"] else ("ERR" if r.get("error") else "fail"), r["stat"],
                r["op"], r["thr"], "None" if r["value"] is None else "%.4g" % r["value"], r["n"], r["seconds"],
                r["claim"][:95]), flush=True)
    os.makedirs(S.OUT, exist_ok=True)
    p = os.path.join(S.OUT, "followup_%s.json" % args.stage)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
