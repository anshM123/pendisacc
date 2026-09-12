"""T3: does the EXACT quotient distance predict transfer, where the fitted one failed?

Pre-registered in PREREGISTRATION_OVERNIGHT.md. Not run overnight -- the GPU
queue was written and the CPU counterpart was not, so this runs now.

H9 ranked the 38-condition suite by theta^T G theta with a FITTED metric and
lost to trajectory RMSE, -0.322 against -0.532. Its diagnosed cause was that G
was fitted from 2% perturbations while the suite reaches +150%, so the
quadratic form did not extrapolate.

The exact group has no such neighbourhood. In log-parameter coordinates the
derived generator is a fixed direction, and the quantity that should matter is
the component of the error the group CANNOT absorb: the distance to the
symmetry orbit.

    theta   = log(perturbed / nominal),  per parameter
    d_perp  = || theta - (theta . u) u ||,  u the unit generator

  run.cmd experiments/t3_quotient.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamics.geometry import NAMES  # noqa: E402

MARGIN = 0.10
BASE = {"tau": 0.100, "kv": 400.0, "gravity": 9.80665}
ABS_SCALE = {"b_joint": 0.01, "fc_joint": 0.01}
EXPRESSIBLE = ("mass_scale", "cart_mass_scale", "gravity", "tau", "kv",
               "joint_damping", "joint_friction")
# the derived generator: every inertial and dissipative parameter, plus kv
IN_GROUP = ("m1", "m2", "m3", "I1", "I2", "I3", "cart_mass", "kv",
            "b_joint", "fc_joint")


def theta_log(xi):
    """Error in LOG coordinates; None if the condition is not expressible."""
    if any(k not in EXPRESSIBLE for k in xi):
        return None
    t = np.zeros(len(NAMES))
    idx = {n: i for i, n in enumerate(NAMES)}
    if "mass_scale" in xi:
        for i, s in enumerate(xi["mass_scale"]):
            t[idx["m%d" % (i + 1)]] = np.log(s)
            t[idx["I%d" % (i + 1)]] = np.log(s)
    if "cart_mass_scale" in xi:
        t[idx["cart_mass"]] = np.log(xi["cart_mass_scale"])
    for k, p in (("gravity", "gravity"), ("tau", "tau"), ("kv", "kv")):
        if k in xi:
            t[idx[p]] = np.log(xi[k] / BASE[k])
    # b_joint and fc_joint are zero at nominal, so a log ratio is undefined.
    # They enter the group linearly, so the additive coordinate used elsewhere
    # in this project is kept and treated as already-logarithmic in effect.
    if "joint_damping" in xi:
        t[idx["b_joint"]] = xi["joint_damping"] / ABS_SCALE["b_joint"]
    if "joint_friction" in xi:
        t[idx["fc_joint"]] = xi["joint_friction"] / ABS_SCALE["fc_joint"]
    return t


def rank(v):
    v = np.asarray(v, dtype=float)
    o = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b, seed=0, K=20000):
    ra, rb = rank(a), rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(K)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def main() -> int:
    h2 = {r["name"]: r for r in json.load(open(
        os.path.join(ROOT, "results", "h2_test.json"), encoding="utf-8"))["rows"]}
    u = np.array([1.0 if n in IN_GROUP else 0.0 for n in NAMES])
    u /= np.linalg.norm(u)

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "xi", "suite", "*.json"))):
        name = os.path.basename(f)[:-5]
        if name not in h2:
            continue
        t = theta_log(json.load(open(f, encoding="utf-8")))
        if t is None:
            continue
        along = float(t @ u)
        perp = float(np.linalg.norm(t - along * u))
        rows.append({"name": name, "success": 100.0 * h2[name]["success"],
                     "traj_rmse": h2[name]["traj_rmse"],
                     "d_perp": perp, "d_along": abs(along),
                     "d_total": float(np.linalg.norm(t))})

    rows.sort(key=lambda r: r["d_perp"])
    print("T3 -- exact quotient distance vs trajectory fidelity\n")
    print("  %-20s %8s %10s %10s %10s" % ("condition", "success", "d_perp", "|d_along|", "traj_rmse"))
    print("  " + "-" * 64)
    for r in rows:
        print("  %-20s %7.1f%% %10.4f %10.4f %10.4f"
              % (r["name"], r["success"], r["d_perp"], r["d_along"], r["traj_rmse"]))

    s = np.array([r["success"] for r in rows])
    dp = np.array([r["d_perp"] for r in rows])
    da = np.array([r["d_along"] for r in rows])
    dt = np.array([r["d_total"] for r in rows])
    tr = np.array([r["traj_rmse"] for r in rows])
    r_p, p_p = spearman(dp, s)
    r_a, p_a = spearman(da, s)
    r_t, p_t = spearman(dt, s)
    r_r, p_r = spearman(tr, s)
    print("\n  n = %d" % len(rows))
    print("    rho(d_perp,    success) = %+.3f  p = %.4f   <- quotient distance" % (r_p, p_p))
    print("    rho(d_along,   success) = %+.3f  p = %.4f   <- the FREE component" % (r_a, p_a))
    print("    rho(d_total,   success) = %+.3f  p = %.4f" % (r_t, p_t))
    print("    rho(traj_rmse, success) = %+.3f  p = %.4f   <- incumbent" % (r_r, p_r))

    v = {}
    v["P2"] = "SUPPORTED" if r_p < 0 else "FAIL"
    m = abs(r_p) - abs(r_r)
    v["P1"] = "SUPPORTED" if (m >= MARGIN and r_p < 0) else "FAIL"
    print("\n  P1  |rho(d_perp)| beats |rho(traj_rmse)| by >= %.2f" % MARGIN)
    print("      %.3f - %.3f = %+.3f   -> %s" % (abs(r_p), abs(r_r), m, v["P1"]))
    print("  P2  rho(d_perp) < 0   -> %s" % v["P2"])
    if abs(r_a) < 0.2:
        print("\n  Note: the component ALONG the group is uncorrelated with success")
        print("  (rho = %+.3f), which is the theorem showing up in the suite data." % r_a)

    out = os.path.join(ROOT, "results", "T3", "quotient.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows, "n": len(rows), "generator": u.tolist(),
               "rho_perp": r_p, "p_perp": p_p, "rho_along": r_a,
               "rho_total": r_t, "rho_traj": r_r, "margin": m, "verdicts": v},
              open(out, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
