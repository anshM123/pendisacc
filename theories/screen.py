"""Bulk theory screen. THIS FILE IS THE PRE-REGISTRATION.

Every theory below states its claim, its statistic, and its pass threshold in
code, and this file is committed before it is run. Thresholds are not edited
after results are seen; a changed threshold is a new theory with a new id.

Screening ~50 theories at once guarantees some pass by chance. So nothing that
passes here is a result. It is a CANDIDATE, and becomes a result only if it
also passes on held-out data (the corrected-asset Isaac sets, group B) or in a
fresh, pre-registered Isaac run on the GPU.

Groups
  A  transfer    does a CPU linear quantity of a policy predict its Isaac transfer?
                 (24 policies x 5 Isaac conditions, original asset: h7_score.json)
  B  holdout     the same predictors on corrected-asset Isaac data never used above
  C  RL          what RL training does to closed-loop stability
  D  certificate what robust balance is physically possible (actuator/rail/delay)
  E  hardware    identification and hardware tolerances
  F  similarity  what the dimensionless structure buys
  calibration / control: sanity checks, not counted as theories

  run.cmd theories/screen.py [--only A01,B02]
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
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import engine as E  # noqa: E402
from dynamics.closed_loop import NZ  # noqa: E402

LOGS = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup")
OUT = os.path.join(ROOT, "results", "theories")
YI = E.YIDX
CMAT = np.zeros((9, NZ))
CMAT[np.arange(9), YI] = 1.0


def J(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


# ----------------------------------------------------------------- plants
def spec_to_AB(spec, asset):
    ms = spec.get("mass_scale", [1.0, 1.0, 1.0])
    iscale = spec.get("inertia_scale", ms)          # evaluate.py scales inertia with mass
    s7 = (*ms, *iscale, spec.get("cart_mass_scale", 1.0))
    dt = spec.get("dt", 1 / 250)
    delay = spec.get("delay_steps", int(round(spec.get("delay_s", 0.0) * 250)))
    return E.open_loop(s7, asset=asset, kv=spec.get("kv", 400.0), tau=spec.get("tau", 0.1),
                       delay=delay, dt=dt, b_joint=spec.get("joint_damping", 0.0),
                       order=spec.get("order", 1), omega_n=spec.get("omega_n", 0.0),
                       zeta=spec.get("zeta", 0.9))


@lru_cache(maxsize=None)
def _AB(key, asset):
    return spec_to_AB(json.loads(key), asset)


def ab(spec, asset):
    return _AB(json.dumps(spec, sort_keys=True), asset)


def rho_of(K, spec, asset):
    A, B = ab(spec, asset)
    return E.rho(A, B, np.asarray(K))


def margin(K, spec, asset):
    return 1.0 - rho_of(K, spec, asset)


def merge(a, b):
    d = dict(a)
    d.update(b)
    return d


def K_of(ck):
    return np.array(E.policy_gain(ck)[0])


def tip_c(asset):
    L = E.basis(asset)["L"]
    c = np.zeros(NZ)
    c[1:4] = L / L.sum()
    return c


# ----------------------------------------------------------------- data
ARM_XI = {"S_nominal": {}, "S_equiv_c8": {"mass_scale": [8.0, 8.0, 8.0]},
          "S_transverse": {"mass_scale": [1.5, 1.0, 1.0]},
          "S_twinA": {"order": 2, "omega_n": 28.0, "zeta": 0.5},
          "S_twinB": {"joint_damping": 0.004}, "S_delay": {"delay_s": 0.012},
          "none": {}, "box": {}, "geom": {}}
CONDS = {"S1": {"mass_scale": [1.25, 1.0, 1.0]}, "S2": {"tau": 0.13}, "S3": {"delay_s": 0.016},
         "S5": {"mass_scale": [1.0, 1.0, 1.2], "tau": 0.12},
         "R_star": {"tau": 0.075, "delay_s": 0.008, "mass_scale": [1.12, 1.09, 1.14], "cart_mass_scale": 1.07}}
STRESS = ("S1", "S2", "S3", "S5")


def _iters(name):
    return int("".join(c for c in name if c.isdigit()) or 0)


@lru_cache(maxsize=None)
def D1():
    """24 policies with Isaac outcomes, their H7 checkpoint, training plant, per-checkpoint own-sim."""
    h7 = J("results/h7_score.json")
    rows = []
    for sel, pref in (("claim5_select", "C5_"), ("h5_select", "")):
        for f in sorted(glob.glob(os.path.join(ROOT, "results", sel, "*.json"))):
            tag = os.path.basename(f)[:-5]
            if tag not in h7["per_condition"]:
                continue
            res = json.load(open(f, encoding="utf-8"))["results"]
            best = max(res, key=lambda r: (r["success_rate"], -r["early_termination_rate"]))
            core = tag.replace(pref, "", 1) if pref else tag
            run = sorted(glob.glob(os.path.join(LOGS, "*_" + core)))[-1]
            arm, seed = core.rsplit("_s", 1)
            arm = arm.replace("H5_", "")
            rows.append({"tag": tag, "arm": arm, "seed": seed, "run": run,
                         "ckpt": os.path.join(run, best["checkpoint"]),
                         "final": os.path.join(run, sorted(
                             [os.path.basename(x) for x in glob.glob(os.path.join(run, "model_*.pt"))],
                             key=_iters)[-1]),
                         "own": ARM_XI[arm], "isaac": h7["per_condition"][tag], "select": res})
    return tuple(rows)


def competent():
    return [p for p in D1() if p["isaac"]["own_sim"] >= 95]


def auc(score, label):
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    tot = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return float(tot / (len(pos) * len(neg)))


def spear(a, b):
    a = np.asarray(a, float)
    a = np.where(np.isneginf(a), -1e9, a)
    r = spearmanr(a, np.asarray(b, float)).correlation
    return float(r) if r == r else float("nan")


# ----------------------------------------------------------------- predictors
def delay_margin(K, base, asset):
    best = None
    for d in range(0, 5):
        r = rho_of(K, merge(base, {"delay_steps": d}), asset)
        if r < 1:
            best = (d, r)
        else:
            break
    if best is None:
        return -(rho_of(K, merge(base, {"delay_steps": 0}), asset) - 1)
    return best[0] + min(0.99, (1 - best[1]) * 50)


def tau_margin(K, base, asset):
    r0 = rho_of(K, merge(base, {"tau": 0.1}), asset)
    if r0 >= 1:
        return 0.1 - (r0 - 1)
    t = 0.1
    while t < 0.4 and rho_of(K, merge(base, {"tau": round(t + 0.01, 3)}), asset) < 1:
        t = round(t + 0.01, 3)
    return t


def mass_radius(K, base, asset, link, sign):
    def r(t):
        ms = list(base.get("mass_scale", [1.0, 1.0, 1.0]))
        ms[link] = float(ms[link] * np.exp(sign * t))
        return rho_of(K, merge(base, {"mass_scale": ms}), asset)
    r0 = r(0.0)
    if r0 >= 1:
        return -(r0 - 1)
    lo, hi = 0.0, None
    for t in np.arange(0.05, 1.0001, 0.05):
        if r(float(t)) >= 1:
            hi = float(t)
            break
        lo = float(t)
    if hi is None:
        return 1.0
    for _ in range(12):
        m = 0.5 * (lo + hi)
        if r(m) >= 1:
            hi = m
        else:
            lo = m
    return lo


def min_radius(K, base, asset):
    return min(mass_radius(K, base, asset, i, s) for i in range(3) for s in (1, -1))


def worst_box(K, asset, w=0.2, base=None):
    base = base or {}
    return 1 - max(rho_of(K, merge(base, {"mass_scale": [float(np.exp(x)) for x in g]}), asset)
                   for g in itertools.product((-w, 0.0, w), repeat=3))


def closed_eigs(K, spec, asset):
    A, B = ab(spec, asset)
    return np.linalg.eig(E.closed(A, B, np.asarray(K)))


def noise_var(K, spec, asset, sigma=3.8e-4):
    A, B = ab(spec, asset)
    Acl = E.closed(A, B, np.asarray(K))
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1:
        return np.inf
    G = np.outer(B, K)
    P = solve_discrete_lyapunov(Acl, sigma ** 2 * G @ G.T)
    c = tip_c(asset)
    return float(c @ P @ c)


def hinf_peak(K, spec, asset):
    A, B = ab(spec, asset)
    Acl = E.closed(A, B, np.asarray(K))
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1:
        return np.inf
    c, I = tip_c(asset), np.eye(NZ)
    return max(abs(c @ np.linalg.solve(np.exp(1j * x) * I - Acl, B)) for x in np.linspace(1e-3, np.pi, 300))


def min_damping(K, spec, asset, dt=1 / 250):
    lam, _ = closed_eigs(K, spec, asset)
    lam = lam[np.abs(lam) > 1e-6]
    s = np.log(lam.astype(complex)) / dt
    osc = s[np.abs(s.imag) > 0.5]
    if len(osc) == 0:
        return 1.0
    return float(np.min(-osc.real / np.abs(osc)))


def p_fast(s7, asset, kv=400.0):
    M, K = E.plant_mats(s7, asset)
    Bf = E.basis(asset)["B"]
    Mi = np.linalg.inv(M)
    Ac = np.zeros((8, 8))
    Ac[:4, 4:] = np.eye(4)
    Ac[4:, :4] = -Mi @ K
    Ac[4:, 4:] = -Mi @ (np.outer(Bf, np.eye(4)[0]) * kv)
    return float(np.max(np.linalg.eigvals(Ac).real))


def s7_of(spec):
    ms = spec.get("mass_scale", [1.0, 1.0, 1.0])
    return (*ms, *spec.get("inertia_scale", ms), spec.get("cart_mass_scale", 1.0))


# ----------------------------------------------------------------- certificate machinery
VARS = ((0, 1.0), (2, 1.3), (0, 0.7))       # (delay steps, tau factor)
ICS = np.zeros((2, NZ))
ICS[0, 1:4], ICS[0, 5:8] = [0.03, -0.02, 0.01], [0.05, -0.03, 0.02]
ICS[1, 1:4], ICS[1, 5:8] = [-0.02, 0.03, -0.03], [-0.04, 0.05, -0.01]


def family(w, axes=(0, 1, 2), variants=VARS, tau0=0.1, dt=1 / 250, full=False):
    grid = (-w, 0.0, w) if w > 0 else (0.0,)
    pts = []
    for g in itertools.product(*[(grid if i in axes else (0.0,)) for i in range(3)]):
        on_axes = [abs(g[i]) for i in axes]
        if full or all(v == 0 for v in g) or all(abs(v - w) < 1e-12 for v in on_axes):
            pts.append(g)
    out = []
    for g in pts:
        for d, tf in variants:
            out.append(ab({"mass_scale": [float(np.exp(v)) for v in g], "delay_steps": d,
                           "tau": tau0 * tf, "dt": dt}, "corrected"))
    return out


def lin_terms(K, As, Bs, Cy, ics, horizon):
    KC = K @ Cy
    Acl = As + Bs[:, :, None] * KC[None, None, :]
    r = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    X = np.broadcast_to(ics, (len(As),) + ics.shape).copy()
    AT = np.transpose(Acl, (0, 2, 1))
    pa = px = 0.0
    for _ in range(horizon):
        X = X @ AT
        pa = max(pa, float(np.abs(X @ KC).max()))
        px = max(px, float(np.abs(X[:, :, 0]).max()))
        if pa > 1e6 or not np.isfinite(pa):
            return r, 1e6, 1e6
    return r, pa, px


def certify(plants, verify, K0, a_max=1.0, x_max=0.4, rho_max=0.998, memory=False,
            starts=3, fev=1500, horizon=300):
    def pack(pl):
        As = np.array([p[0] for p in pl])
        Bs = np.array([p[1] for p in pl])
        n = As.shape[1]
        Cy, ics = CMAT.copy(), ICS.copy()
        if memory:
            Aa = np.zeros((len(As), n + 9, n + 9))
            Aa[:, :n, :n], Aa[:, n:, :n] = As, CMAT[None]
            Ba = np.zeros((len(As), n + 9))
            Ba[:, :n] = Bs
            Cy = np.zeros((18, n + 9))
            Cy[:9, :n], Cy[9:, n:] = CMAT, np.eye(9)
            As, Bs, ics = Aa, Ba, np.hstack([ics, np.zeros((len(ics), 9))])
        return As, Bs, Cy, ics

    As, Bs, Cy, ics = pack(plants)
    K0 = np.concatenate([K0, np.zeros(9)]) if memory else np.asarray(K0, float)
    scale = np.maximum(np.abs(K0), 0.05)

    def obj(x, As=As, Bs=Bs):
        r, pa, px = lin_terms(x * scale, As, Bs, Cy, ics, horizon)
        return max(r - rho_max, 0) * 100 + max(pa - a_max, 0) + max(px - x_max, 0) * 2.5 + 1e-3 * r

    rng = np.random.default_rng(0)
    best = None
    for s in range(starts):
        x0 = K0 / scale
        if s:
            x0 = x0 * rng.uniform(0.3, 1.2) * (1 + 0.3 * rng.standard_normal(len(x0)))
        r1 = minimize(obj, x0, method="Nelder-Mead", options={"maxfev": fev, "adaptive": True})
        r2 = minimize(obj, r1.x, method="Powell", options={"maxfev": fev})
        if best is None or r2.fun < best[0]:
            best = (r2.fun, r2.x)
    Av, Bv, _, _ = pack(verify)
    r, pa, px = lin_terms(best[1] * scale, Av, Bv, Cy, ics, horizon)
    ok = r <= rho_max and pa <= a_max and px <= x_max
    return bool(ok), {"rho": r, "peak_a": pa, "peak_x": px}


def cert_at(w, **kw):
    fam = dict((k, kw.pop(k)) for k in ("axes", "variants", "tau0", "dt") if k in kw)
    K0 = K_of(E.checkpoints("CORR_s1")[-1])
    return certify(family(w, **fam), family(w, full=True, **fam), K0, **kw)


def hang(s7, asset="corrected"):
    M, K = E.plant_mats(s7, asset, at="hang")
    w2, V = np.linalg.eig(np.linalg.solve(M[1:, 1:], K[1:, 1:]))
    o = np.argsort(w2.real)
    return np.sqrt(np.abs(w2.real[o])) / (2 * np.pi), V[:, o].real


# ----------------------------------------------------------------- registry
REG = []


def theory(tid, group, claim, stat, op, thr):
    def deco(fn):
        REG.append({"id": tid, "group": group, "claim": claim, "stat": stat, "op": op, "thr": thr, "fn": fn.__name__})
        return fn
    return deco


def passed(value, op, thr):
    if value is None or (isinstance(value, float) and value != value):
        return False
    return value >= thr if op == ">=" else value <= thr


# ---- A: transfer prediction on 24 policies x Isaac conditions (original asset)
@theory("A01", "A", "CPU linear margin at the exact Isaac condition classifies Isaac success>=50% (S1,S2,S3,S5,R*)", "AUC", ">=", 0.80)
def A01():
    sc, lb = [], []
    for p in D1():
        K = K_of(p["ckpt"])
        for c, spec in CONDS.items():
            sc.append(margin(K, spec, "original"))
            lb.append(p["isaac"][c] >= 50)
    return auc(sc, lb), len(sc), {"n_pos": int(sum(lb))}


@theory("A02", "A", "Exact-condition CPU margin rank-correlates with Isaac success, pooled", "spearman", ">=", 0.50)
def A02():
    sc, y = [], []
    for p in D1():
        K = K_of(p["ckpt"])
        for c, spec in CONDS.items():
            sc.append(margin(K, spec, "original"))
            y.append(p["isaac"][c])
    return spear(sc, y), len(sc), {}


@theory("A03", "A", "Margin on the policy's own training plant ranks Isaac R* transfer", "spearman", ">=", 0.50)
def A03():
    ps = D1()
    return spear([margin(K_of(p["ckpt"]), p["own"], "original") for p in ps],
                 [p["isaac"]["R_star"] for p in ps]), len(ps), {}


@theory("A04", "A", "Delay margin (steps) classifies Isaac S3 (16 ms delay) success", "AUC", ">=", 0.80)
def A04():
    ps = D1()
    return auc([delay_margin(K_of(p["ckpt"]), {}, "original") for p in ps],
               [p["isaac"]["S3"] >= 50 for p in ps]), len(ps), {}


@theory("A05", "A", "Servo-lag margin classifies Isaac S2 (tau 0.13) success", "AUC", ">=", 0.80)
def A05():
    ps = D1()
    return auc([tau_margin(K_of(p["ckpt"]), {}, "original") for p in ps],
               [p["isaac"]["S2"] >= 50 for p in ps]), len(ps), {}


@theory("A06", "A", "Link-1 upward mass radius classifies Isaac S1 (m1 x1.25) success", "AUC", ">=", 0.80)
def A06():
    ps = D1()
    return auc([mass_radius(K_of(p["ckpt"]), {}, "original", 0, 1) for p in ps],
               [p["isaac"]["S1"] >= 50 for p in ps]), len(ps), {}


@theory("A07", "A", "Exact-plant margin classifies Isaac S5 (m3 x1.2 + tau 0.12) success", "AUC", ">=", 0.80)
def A07():
    ps = D1()
    return auc([margin(K_of(p["ckpt"]), CONDS["S5"], "original") for p in ps],
               [p["isaac"]["S5"] >= 50 for p in ps]), len(ps), {}


@theory("A08", "A", "Exact-plant margin classifies Isaac R* success", "AUC", ">=", 0.80)
def A08():
    ps = D1()
    return auc([margin(K_of(p["ckpt"]), CONDS["R_star"], "original") for p in ps],
               [p["isaac"]["R_star"] >= 50 for p in ps]), len(ps), {}


@theory("A09", "A", "Worst margin over a +-20% arm-mass box ranks mean Isaac stress success", "spearman", ">=", 0.50)
def A09():
    ps = D1()
    return spear([worst_box(K_of(p["ckpt"]), "original") for p in ps],
                 [np.mean([p["isaac"][c] for c in STRESS]) for p in ps]), len(ps), {}


@theory("A10", "A", "Min margin over the five Isaac conditions ranks mean Isaac stress success", "spearman", ">=", 0.60)
def A10():
    ps = D1()
    return spear([min(margin(K_of(p["ckpt"]), s, "original") for s in CONDS.values()) for p in ps],
                 [np.mean([p["isaac"][c] for c in STRESS]) for p in ps]), len(ps), {}


@theory("A11", "A", "Smaller local gain norm ranks higher Isaac R* success", "spearman", ">=", 0.50)
def A11():
    ps = D1()
    return spear([-np.linalg.norm(K_of(p["ckpt"])) for p in ps], [p["isaac"]["R_star"] for p in ps]), len(ps), {}


@theory("A12", "A", "Lower sensor-noise tip variance at R* plant ranks higher R* success", "spearman", ">=", 0.50)
def A12():
    ps = D1()
    return spear([-np.log(noise_var(K_of(p["ckpt"]), CONDS["R_star"], "original") + 1e-30) for p in ps],
                 [p["isaac"]["R_star"] for p in ps]), len(ps), {}


@theory("A13", "A", "Lower H-inf peak (action disturbance -> tip) at R* ranks mean stress success", "spearman", ">=", 0.50)
def A13():
    ps = D1()
    return spear([-np.log(hinf_peak(K_of(p["ckpt"]), CONDS["R_star"], "original")) for p in ps],
                 [np.mean([p["isaac"][c] for c in STRESS]) for p in ps]), len(ps), {}


@theory("A14", "A", "Minimum closed-loop damping ratio on nominal ranks R* success", "spearman", ">=", 0.50)
def A14():
    ps = D1()
    return spear([min_damping(K_of(p["ckpt"]), {}, "original") for p in ps],
                 [p["isaac"]["R_star"] for p in ps]), len(ps), {}


@theory("A15", "A", "Picking the competent policy with best R*-plant CPU margin yields Isaac R* >= 70", "R* of pick", ">=", 70.0)
def A15():
    ps = competent()
    pick = max(ps, key=lambda p: margin(K_of(p["ckpt"]), CONDS["R_star"], "original"))
    return float(pick["isaac"]["R_star"]), len(ps), {"pick": pick["tag"]}


@theory("A16", "A", "Picking by target-agnostic robustness (arm box + delay + lag margins) yields R* >= 70", "R* of pick", ">=", 70.0)
def A16():
    ps = competent()
    cols = []
    for f in (lambda K: worst_box(K, "original"), lambda K: delay_margin(K, {}, "original"),
              lambda K: tau_margin(K, {}, "original")):
        v = np.array([f(K_of(p["ckpt"])) for p in ps])
        cols.append(np.argsort(np.argsort(v)))
    score = np.sum(cols, axis=0)
    pick = ps[int(np.argmax(score))]
    return float(pick["isaac"]["R_star"]), len(ps), {"pick": pick["tag"]}


@theory("A17", "A", "R*-plant margin orders every twin pair (same reward, different transfer) correctly", "pairs correct", ">=", 3)
def A17():
    by = {p["tag"]: p for p in D1()}
    ok = n = 0
    for k in ("1", "2", "3"):
        a, b = by.get("C5_S_twinA_s" + k), by.get("C5_S_twinB_s" + k)
        if not a or not b:
            continue
        n += 1
        ma, mb = (margin(K_of(x["ckpt"]), CONDS["R_star"], "original") for x in (a, b))
        ok += (mb > ma) == (b["isaac"]["R_star"] > a["isaac"]["R_star"])
    return ok, n, {}


@theory("A18", "A", "Within a training arm, R*-plant margin orders seeds whose R* differs by >=20 points", "fraction", ">=", 0.70)
def A18():
    ps = D1()
    ok = n = 0
    for a, b in itertools.combinations(ps, 2):
        if a["arm"] != b["arm"] or abs(a["isaac"]["R_star"] - b["isaac"]["R_star"]) < 20:
            continue
        n += 1
        ma, mb = (margin(K_of(x["ckpt"]), CONDS["R_star"], "original") for x in (a, b))
        ok += (ma > mb) == (a["isaac"]["R_star"] > b["isaac"]["R_star"])
    return (ok / n if n else float("nan")), n, {}


@theory("A19", "control", "CONTROL: own-simulator success classifies R*>=50 (expected to fail; the known blind spot)", "AUC", ">=", 0.80)
def A19():
    ps = D1()
    return auc([p["isaac"]["own_sim"] for p in ps], [p["isaac"]["R_star"] >= 50 for p in ps]), len(ps), {}


# ---- B: held-out corrected-asset Isaac data
@theory("B01", "B", "HOLDOUT: CPU margin classifies Isaac T4 landscape success (2 policies x 30 mass conditions)", "AUC", ">=", 0.80)
def B01():
    t = J("results/T4/landscape_corrected.json")
    sel = {s["policy"]: os.path.join(ROOT, s["run"], s["checkpoint"]) for s in t["selection"]}
    sc, lb = [], []
    for r in t["rows"]:
        c, d = r["c"], r["delta"]
        sc.append(margin(K_of(sel[r["policy"]]), {"mass_scale": [c * (1 + d), c, c]}, "corrected"))
        lb.append(r["success"] >= 0.5)
    return auc(sc, lb), len(sc), {"n_pos": int(sum(lb))}


@theory("B02", "B", "HOLDOUT: CPU margin classifies Isaac T1 success over 26 random 8-D parameter directions", "AUC", ">=", 0.80)
def B02():
    t = J("results/T1/nullspace.json")
    K = K_of(os.path.join(LOGS, "2026-09-11_19-37-34_CORR_s2", "model_999.pt"))
    sc, lb = [], []
    for r in t["rows"]:
        f = r["factors"]
        spec = {"mass_scale": f[0:3], "inertia_scale": f[3:6], "cart_mass_scale": f[6], "kv": 400.0 * f[7]}
        sc.append(margin(K, spec, "corrected"))
        lb.append(r["success"] >= 0.5)
    return auc(sc, lb), len(sc), {"n_pos": int(sum(lb))}


def _ckpt_rows():
    """(checkpoint path, own-plant spec, asset, own-sim success) for every per-checkpoint Isaac eval."""
    rows = []
    for p in D1():
        for r in p["select"]:
            ck = os.path.join(p["run"], r["checkpoint"])
            if os.path.exists(ck):
                rows.append((ck, p["own"], "original", r["success_rate"], p["run"]))
    for f in glob.glob(os.path.join(ROOT, "results", "corrected", "CORR_s*.json")):
        tag = os.path.basename(f)[:-5]
        for r in json.load(open(f, encoding="utf-8"))["results"]:
            for run in sorted(glob.glob(os.path.join(LOGS, "*_" + tag))):
                ck = os.path.join(run, r["checkpoint"])
                if os.path.exists(ck):
                    rows.append((ck, {}, "corrected", r["success_rate"], run))
                    break
    for f in glob.glob(os.path.join(ROOT, "results", "T5_VOID_fallthrough", "*.json")):
        tag = "VOID" + os.path.basename(f)[:-5]
        runs = sorted(glob.glob(os.path.join(LOGS, "*_" + tag)))
        if not runs:
            continue
        for r in json.load(open(f, encoding="utf-8"))["results"]:
            ck = os.path.join(runs[-1], r["checkpoint"])
            if os.path.exists(ck):
                rows.append((ck, {}, "corrected", r["success_rate"], runs[-1]))
    return rows


@theory("B03", "B", "HOLDOUT: own-plant CPU margin classifies per-checkpoint Isaac own-sim success (all runs)", "AUC", ">=", 0.80)
def B03():
    rows = _ckpt_rows()
    return auc([margin(K_of(ck), sp, a) for ck, sp, a, _, _ in rows], [s >= 0.5 for *_, s, _ in rows]), len(rows), {}


@theory("B04", "B", "FIDELITY: Isaac-competent final policies are locally stable on the CPU model", "fraction", ">=", 0.80)
def B04():
    rows = [(ck, sp, a) for ck, sp, a, s, _ in _ckpt_rows() if s >= 0.95]
    return float(np.mean([rho_of(K_of(ck), sp, a) < 1 for ck, sp, a in rows])), len(rows), {}


@theory("B05", "B", "HOLDOUT: CPU margin sign matches Isaac success>=50 on the H8 CAD-defect conditions", "fraction", ">=", 0.83)
def B05():
    K = K_of(os.path.join(LOGS, "2026-09-05_21-02-09_rel1", "model_800.pt"))
    ok = n = 0
    det = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "xi", "h8", "*.json"))):
        rp = os.path.join(ROOT, "results", "h8", os.path.basename(f))
        if not os.path.exists(rp):
            continue
        xi = json.load(open(f, encoding="utf-8"))
        s = json.load(open(rp, encoding="utf-8"))["results"][0]["success_rate"]
        m = margin(K, {k: v for k, v in xi.items() if k in ("mass_scale", "cart_mass_scale", "tau", "delay_s", "kv")},
                   "original")
        n += 1
        ok += (m > 0) == (s >= 0.5)
        det[os.path.basename(f)] = [round(m, 5), s]
    return (ok / n if n else float("nan")), n, det


# ---- C: what RL training does
def _rho_own(p, ck=None):
    return rho_of(K_of(ck or p["ckpt"]), p["own"], "original")


@theory("C01", "C", "RL converges to the edge of stability: |1-rho| < 0.01 for >=80% of competent policies", "fraction", ">=", 0.80)
def C01():
    ps = competent()
    v = [abs(1 - _rho_own(p)) < 0.01 for p in ps]
    return float(np.mean(v)), len(ps), {}


@theory("C02", "C", "Spectral radius rises toward 1 over training (Spearman(iter, rho) > 0.3) in >=60% of runs", "fraction", ">=", 0.60)
def C02():
    ok = n = 0
    for p in D1():
        cks = [os.path.join(p["run"], r["checkpoint"]) for r in p["select"]]
        cks = [c for c in cks if os.path.exists(c)]
        if len(cks) < 4:
            continue
        n += 1
        ok += spear([_iters(os.path.basename(c)) for c in cks], [_rho_own(p, c) for c in cks]) > 0.3
    return (ok / n if n else float("nan")), n, {}


def _mode_energy(K, spec, asset, idx, pick="max"):
    lam, V = closed_eigs(K, spec, asset)
    k = int(np.argmax(np.abs(lam)))
    v = np.abs(V[:8, k]) ** 2
    return float(v[idx].sum() / v.sum())


@theory("C03", "C", "The slowest closed-loop mode is cart drift: >=60% of its energy in cart states, >=80% of policies", "fraction", ">=", 0.80)
def C03():
    ps = competent()
    return float(np.mean([_mode_energy(K_of(p["ckpt"]), p["own"], "original", [0, 4]) >= 0.6 for p in ps])), len(ps), {}


@theory("C04", "C", "Where R* destabilises a policy, the unstable mode is a pendulum mode (>=60% angle energy), >=80%", "fraction", ">=", 0.80)
def C04():
    v = []
    for p in D1():
        K = K_of(p["ckpt"])
        if rho_of(K, CONDS["R_star"], "original") > 1:
            v.append(_mode_energy(K, CONDS["R_star"], "original", [1, 2, 3, 5, 6, 7]) >= 0.6)
    return (float(np.mean(v)) if v else float("nan")), len(v), {}


@theory("C05", "C", "Competent policies have a closed-loop time constant > 2 s (weak centring), >=80%", "fraction", ">=", 0.80)
def C05():
    ps = competent()
    t = []
    for p in ps:
        r = _rho_own(p)
        t.append(r < 1 and -(1 / 250) / np.log(r) > 2.0)
    return float(np.mean(t)), len(ps), {}


def _arm_mean(arm, f):
    v = [f(p) for p in D1() if p["arm"] == arm]
    return float(np.mean(v)) if v else float("nan")


@theory("C06", "C", "Training with 12 ms delay raises the CPU delay margin by >=1 step over nominal training", "steps", ">=", 1.0)
def C06():
    f = lambda p: delay_margin(K_of(p["ckpt"]), {}, "original")  # noqa: E731
    return _arm_mean("S_delay", f) - _arm_mean("S_nominal", f), 5, {}


@theory("C07", "C", "Mass DR (H5 box/geom) raises the minimum arm-mass radius by >=0.1 log over H5_none", "log radius", ">=", 0.10)
def C07():
    f = lambda p: min_radius(K_of(p["ckpt"]), {}, "original")  # noqa: E731
    return float(np.mean([_arm_mean("box", f), _arm_mean("geom", f)]) - _arm_mean("none", f)), 9, {}


@theory("C08", "C", "Training at m1 x1.5 raises the link-1 upward radius by >=0.1 log over nominal training", "log radius", ">=", 0.10)
def C08():
    f = lambda p: mass_radius(K_of(p["ckpt"]), {}, "original", 0, 1)  # noqa: E731
    return _arm_mean("S_transverse", f) - _arm_mean("S_nominal", f), 5, {}


@theory("C09", "C", "Common inertial scale x8 with drive fixed leaves the linear margin unchanged (|drho| < 1e-3)", "max |drho|", "<=", 1e-3)
def C09():
    v = [abs(rho_of(K_of(p["ckpt"]), {"mass_scale": [8.0, 8.0, 8.0]}, "original") - rho_of(K_of(p["ckpt"]), {}, "original"))
         for p in D1() if p["arm"] == "S_equiv_c8"]
    return float(max(v)), len(v), {}


@theory("C10", "C", "The best own-sim checkpoint has a larger CPU margin than the final one in >=70% of runs", "fraction", ">=", 0.70)
def C10():
    ok = n = 0
    for p in D1():
        if os.path.basename(p["ckpt"]) == os.path.basename(p["final"]) or not os.path.exists(p["final"]):
            continue
        n += 1
        ok += margin(K_of(p["ckpt"]), p["own"], "original") > margin(K_of(p["final"]), p["own"], "original")
    return (ok / n if n else float("nan")), n, {}


# ---- D: certificates (corrected asset; action, rail, lag/delay variants)
@theory("D00", "calibration", "CALIBRATION: light-budget certificate reproduces v2 (+-25% certified)", "certified", ">=", 1)
def D00():
    ok, t = cert_at(0.25)
    return int(ok), 1, t


@theory("D01", "D", "Rail is binding: allowing 0.55 m cart travel certifies +-37% arm-mass error", "certified", ">=", 1)
def D01():
    ok, t = cert_at(0.375, x_max=0.55)
    return int(ok), 1, t


@theory("D02", "D", "Motor is binding: a 1.5x stronger drive certifies +-37% arm-mass error", "certified", ">=", 1)
def D02():
    ok, t = cert_at(0.375, a_max=1.5)
    return int(ok), 1, t


@theory("D03", "D", "Actuator uncertainty is binding: with the nominal drive only, +-50% is certified", "certified", ">=", 1)
def D03():
    ok, t = cert_at(0.5, variants=((0, 1.0),))
    return int(ok), 1, t


@theory("D04", "D", "Anisotropy: link 2 has the smallest certifiable one-axis mass range", "link2 smallest", ">=", 1)
def D04():
    widths = {}
    for ax in range(3):
        w_ok = 0.0
        for w in (0.25, 0.5, 0.75, 1.0):
            ok, _ = cert_at(w, axes=(ax,), starts=2)
            if not ok:
                break
            w_ok = w
        widths[ax] = w_ok
    return int(widths[1] < widths[0] and widths[1] < widths[2]), 3, widths


@theory("D05", "D", "Measurement priority: with m2 known exactly, +-50% on m1 and m3 is certified", "certified", ">=", 1)
def D05():
    ok, t = cert_at(0.5, axes=(0, 2))
    return int(ok), 1, t


@theory("D06", "D", "Hardware lever: a 50 ms servo (half the lag) certifies +-50%", "certified", ">=", 1)
def D06():
    ok, t = cert_at(0.5, tau0=0.05)
    return int(ok), 1, t


@theory("D07", "D", "Memory helps: one step of observation history certifies +-37%", "certified", ">=", 1)
def D07():
    ok, t = cert_at(0.375, memory=True, starts=2)
    return int(ok), 1, t


@theory("D08", "D", "A 100 Hz controller still certifies +-12% arm-mass error", "certified", ">=", 1)
def D08():
    ok, t = cert_at(0.125, dt=1 / 100)
    return int(ok), 1, t


@theory("D09", "D", "At nominal masses one gain tolerates 0-16 ms delay", "certified", ">=", 1)
def D09():
    ok, t = cert_at(0.0, variants=((0, 1.0), (2, 1.0), (4, 1.0)))
    return int(ok), 1, t


# ---- E: identification and hardware
def _hang_logf(logrho):
    e = np.exp(logrho)
    return np.log(hang((e[0], e[1], e[2], e[0], e[1], e[2], 1.0))[0])


@theory("E01", "E", "The three hanging mode frequencies identify the three arm-mass ratios (Jacobian cond < 30)", "cond", "<=", 30.0)
def E01():
    Jm = np.column_stack([(_hang_logf(np.eye(3)[j] * 1e-4) - _hang_logf(-np.eye(3)[j] * 1e-4)) / 2e-4 for j in range(3)])
    return float(np.linalg.cond(Jm)), 3, {"J": Jm.tolist()}


@theory("E02", "E", "10 s of 0.05 rad free swinging at 14-bit resolution gives each arm-mass ratio to <2% (CRLB)", "max rel std", "<=", 0.02)
def E02():
    Jm = np.column_stack([(_hang_logf(np.eye(3)[j] * 1e-4) - _hang_logf(-np.eye(3)[j] * 1e-4)) / 2e-4 for j in range(3)])
    f = np.exp(_hang_logf(np.zeros(3)))
    N, fs, A, sig = 2500, 250.0, 0.05, 3.8e-4
    eta = A ** 2 / (2 * sig ** 2)
    var_f = 12.0 / ((2 * np.pi) ** 2 * eta * N * (N ** 2 - 1)) * fs ** 2
    S = np.diag(var_f / f ** 2)
    Ji = np.linalg.inv(Jm)
    std = np.sqrt(np.diag(Ji @ S @ Ji.T))
    return float(std.max()), 3, {"freqs_hz": f.tolist(), "rel_std": std.tolist()}


@theory("E03", "E", "Frequencies + mode shapes separate mass from inertia for all 3 arms (6 params, cond < 1e3)", "cond", "<=", 1e3)
def E03():
    def feat(x):
        e = np.exp(x)
        f, V = hang((e[0], e[1], e[2], e[3], e[4], e[5], 1.0))
        V = V / V[0:1, :]
        return np.concatenate([np.log(f), V[1:, :].ravel()])
    Jm = np.column_stack([(feat(np.eye(6)[j] * 1e-4) - feat(-np.eye(6)[j] * 1e-4)) / 2e-4 for j in range(6)])
    sv = np.linalg.svd(Jm, compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-300)), 6, {"sv": sv.tolist()}


def _frac_small(f, thr, pol=None):
    ps = pol or competent()
    return float(np.mean([f(K_of(p["ckpt"])) < thr for p in ps])), len(ps)


@theory("E04", "E", "Drive gain kv x0.5..x2 changes the margin by < 1e-3 for >=90% of competent policies", "fraction", ">=", 0.90)
def E04():
    v, n = _frac_small(lambda K: max(abs(rho_of(K, {"kv": k}, "original") - rho_of(K, {}, "original")) for k in (200.0, 800.0)), 1e-3)
    return v, n, {}


@theory("E05", "E", "Cart mass x0.8..x1.25 changes the margin by < 1e-3 for >=90% of competent policies", "fraction", ">=", 0.90)
def E05():
    v, n = _frac_small(lambda K: max(abs(rho_of(K, {"cart_mass_scale": c}, "original") - rho_of(K, {}, "original")) for c in (0.8, 1.25)), 1e-3)
    return v, n, {}


@theory("E06", "E", "4 ms encoder latency (1 step) keeps >=80% of competent policies locally stable", "fraction", ">=", 0.80)
def E06():
    ps = competent()
    return float(np.mean([rho_of(K_of(p["ckpt"]), {"delay_steps": 1}, "original") < 1 for p in ps])), len(ps), {}


@theory("E07", "E", "14-bit encoder noise gives RMS tip angle < 0.01 rad for >=80% of competent policies", "fraction", ">=", 0.80)
def E07():
    ps = competent()
    return float(np.mean([np.sqrt(noise_var(K_of(p["ckpt"]), {}, "original")) < 0.01 for p in ps])), len(ps), {}


@theory("E08", "E", "Joint viscous damping 0.004 changes the margin by < 5e-4 for >=90% of competent policies", "fraction", ">=", 0.90)
def E08():
    v, n = _frac_small(lambda K: abs(rho_of(K, {"joint_damping": 0.004}, "original") - rho_of(K, {}, "original")), 5e-4)
    return v, n, {}


@theory("E09", "E", "Plant-only fastest unstable pole (policy-free) classifies Isaac T4 success", "AUC", ">=", 0.80)
def E09():
    t = J("results/T4/landscape_corrected.json")
    sc, lb = [], []
    for r in t["rows"]:
        c, d = r["c"], r["delta"]
        ms = [c * (1 + d), c, c]
        sc.append(-p_fast((*ms, *ms, 1.0), "corrected"))
        lb.append(r["success"] >= 0.5)
    return auc(sc, lb), len(sc), {}


@theory("E10", "E", "Arm 3's mass moves the fastest unstable pole most (largest mean |dp/dlog m_i| on +-50% grid)", "link3 largest", ">=", 1)
def E10():
    g = np.linspace(-0.5, 0.5, 5)
    sens = np.zeros(3)
    for x in itertools.product(g, g, g):
        for i in range(3):
            up, dn = np.array(x, float), np.array(x, float)
            up[i] += 1e-4
            dn[i] -= 1e-4
            eu, ed = np.exp(up), np.exp(dn)
            sens[i] += abs(p_fast((*eu, *eu, 1.0), "corrected") - p_fast((*ed, *ed, 1.0), "corrected")) / 2e-4
    return int(np.argmax(sens) == 2), 3, {"mean_abs_sens": (sens / len(g) ** 3).tolist()}


# ---- F: similarity
@theory("F01", "F", "Only 3 directions of (m1,m2,m3,m_cart,kv) matter: top-3 share of margin-gradient energy >= 0.99", "energy", ">=", 0.99)
def F01():
    K = K_of(E.checkpoints("CORR_s1")[-1])
    rng = np.random.default_rng(3)
    G = []
    for _ in range(120):
        x = rng.uniform(-0.3, 0.3, 5)

        def r(y):
            e = np.exp(y)
            return rho_of(K, {"mass_scale": [e[0], e[1], e[2]], "cart_mass_scale": e[3], "kv": 400 * e[4]}, "corrected")
        G.append([(r(x + np.eye(5)[j] * 1e-4) - r(x - np.eye(5)[j] * 1e-4)) / 2e-4 for j in range(5)])
    sv = np.linalg.svd(np.array(G), compute_uv=False)
    return float((sv[:3] ** 2).sum() / (sv ** 2).sum()), 120, {"sv": sv.tolist()}


@theory("F02", "F", "Policy-free dimensionless pole x lag (p_fast * (tau + delay)) classifies pooled Isaac success on S1,S2,S3,S5,R*", "AUC", ">=", 0.80)
def F02():
    sc, lb = [], []
    for p in D1():
        for c, spec in CONDS.items():
            lag = spec.get("tau", 0.1) + spec.get("delay_s", 0.0)
            sc.append(-p_fast(s7_of(spec), "original", spec.get("kv", 400.0)) * lag)
            lb.append(p["isaac"][c] >= 50)
    return auc(sc, lb), len(sc), {}


@theory("F03", "calibration", "CALIBRATION: exact similarity incl. kv leaves the linear margin invariant (|drho| < 1e-9)", "max |drho|", "<=", 1e-9)
def F03():
    v = []
    for p in D1()[:3]:
        K = K_of(p["ckpt"])
        for c in (0.5, 2.0, 8.0):
            v.append(abs(rho_of(K, {"mass_scale": [c] * 3, "cart_mass_scale": c, "kv": 400 * c}, "original")
                         - rho_of(K, {}, "original")))
    return float(max(v)), len(v), {}


# ----------------------------------------------------------------- runner
def run_one(tid):
    rec = dict(next(r for r in REG if r["id"] == tid))
    t0 = time.time()
    try:
        value, n, detail = globals()[rec["fn"]]()
        value = float(value)
        rec.update(value=value, n=int(n), detail=detail, passed=passed(value, rec["op"], rec["thr"]))
    except Exception:
        rec.update(value=None, n=0, detail=traceback.format_exc()[-1500:], passed=False, error=True)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    ids = [r["id"] for r in REG]
    if args.only:
        ids = [i for i in ids if i in args.only.split(",")]
    E.basis("original")
    E.basis("corrected")
    os.makedirs(OUT, exist_ok=True)
    done = []
    heavy = [i for i in ids if i.startswith("D")]
    light = [i for i in ids if i not in heavy]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(run_one, heavy + light):
            done.append(rec)
            print("%-4s %-11s %s  %-12s %s %-8s value %-12s n=%-4s %5.0fs  %s"
                  % (rec["id"], rec["group"], "PASS" if rec["passed"] else ("ERR " if rec.get("error") else "fail"),
                     rec["stat"], rec["op"], rec["thr"],
                     "None" if rec["value"] is None else "%.4g" % rec["value"], rec["n"], rec["seconds"],
                     rec["claim"][:90]), flush=True)
    path = os.path.join(OUT, "screen.json" if not args.only else "screen_partial.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(done, fh, indent=1, default=str)
    counted = [r for r in done if r["group"] not in ("calibration", "control")]
    print("\n%d theories screened, %d pass (candidates only; see file docstring). wrote %s"
          % (len(counted), sum(r["passed"] for r in counted), path))


if __name__ == "__main__":
    main()
