"""H9: does the geometry-weighted displacement beat trajectory fidelity?

Pre-registered in PREREGISTRATION_H9.md. Uses only data already on disk: the
38-condition suite (results/h2_test.json), the condition definitions
(xi/suite/*.json) and the metric tensor (results/geometry_discover.json).

  run.cmd experiments/h9_score.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamics.geometry import NAMES, PARAMS  # noqa: E402

MARGIN = 0.10
BASE = {"tau": 0.100, "kv": 400.0, "gravity": 9.80665}
ABS_SCALE = {"b_joint": 0.01, "fc_joint": 0.01}
EXPRESSIBLE = ("mass_scale", "cart_mass_scale", "gravity", "tau", "kv",
               "joint_damping", "joint_friction")


def theta(xi):
    """Displacement in the geometry's dimensionless coordinates, or None."""
    if any(k not in EXPRESSIBLE for k in xi):
        return None
    t = np.zeros(len(NAMES))
    idx = {n: i for i, n in enumerate(NAMES)}
    if "mass_scale" in xi:
        for i, s in enumerate(xi["mass_scale"]):
            # the suite scales inertia with mass (a density error), so a mass
            # condition moves BOTH coordinates
            t[idx["m%d" % (i + 1)]] = s - 1.0
            t[idx["I%d" % (i + 1)]] = s - 1.0
    if "cart_mass_scale" in xi:
        t[idx["cart_mass"]] = xi["cart_mass_scale"] - 1.0
    for k, p in (("gravity", "gravity"), ("tau", "tau"), ("kv", "kv")):
        if k in xi:
            t[idx[p]] = xi[k] / BASE[k] - 1.0
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
    g = json.load(open(os.path.join(ROOT, "results", "geometry_discover.json"),
                       encoding="utf-8"))
    V = np.array(g["eigenvectors"])
    w = np.array(g["eigenvalues"])
    G = V @ np.diag(w) @ V.T

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "xi", "suite", "*.json"))):
        name = os.path.basename(f)[:-5]
        if name not in h2:
            continue
        xi = json.load(open(f, encoding="utf-8"))
        t = theta(xi)
        if t is None:
            continue
        rows.append({"name": name, "success": 100.0 * h2[name]["success"],
                     "traj_rmse": h2[name]["traj_rmse"],
                     "raw_gap": h2[name]["raw_gap"],
                     "dG": float(t @ G @ t), "norm": float(np.linalg.norm(t))})

    rows.sort(key=lambda r: r["dG"])
    print("%d expressible conditions\n" % len(rows))
    print("  %-20s %8s %12s %11s %10s" % ("condition", "success", "dG", "||theta||", "traj_rmse"))
    print("  " + "-" * 68)
    for r in rows:
        print("  %-20s %7.1f%% %12.3e %11.4f %10.4f"
              % (r["name"], r["success"], r["dG"], r["norm"], r["traj_rmse"]))

    s = np.array([r["success"] for r in rows])
    dG = np.array([r["dG"] for r in rows])
    tr = np.array([r["traj_rmse"] for r in rows])
    nm = np.array([r["norm"] for r in rows])

    r_dG, p_dG = spearman(dG, s)
    r_tr, p_tr = spearman(tr, s)
    r_nm, p_nm = spearman(nm, s)
    print("\n  on these %d conditions:" % len(rows))
    print("    rho(dG,        success) = %+.3f   p = %.4f" % (r_dG, p_dG))
    print("    rho(traj_rmse, success) = %+.3f   p = %.4f   <- incumbent" % (r_tr, p_tr))
    print("    rho(||theta||, success) = %+.3f   p = %.4f" % (r_nm, p_nm))

    print("\nPRE-REGISTERED VERDICTS (PREREGISTRATION_H9.md)\n")
    v = {}
    v["P2"] = "SUPPORTED" if r_dG < 0 else "FAIL -- metric is inverted"
    m_tr = abs(r_dG) - abs(r_tr)
    m_nm = abs(r_dG) - abs(r_nm)
    v["P1"] = "SUPPORTED" if (m_tr >= MARGIN and v["P2"] == "SUPPORTED") else "FAIL"
    v["P3"] = "SUPPORTED" if (m_nm >= MARGIN and v["P2"] == "SUPPORTED") else "FAIL"
    print("  P1  |rho(dG)| beats |rho(traj_rmse)| by >= %.2f" % MARGIN)
    print("      %.3f - %.3f = %+.3f\n      -> %s\n" % (abs(r_dG), abs(r_tr), m_tr, v["P1"]))
    print("  P2  rho(dG) < 0")
    print("      %+.3f\n      -> %s\n" % (r_dG, v["P2"]))
    print("  P3  |rho(dG)| beats |rho(||theta||)| by >= %.2f" % MARGIN)
    print("      %.3f - %.3f = %+.3f\n      -> %s\n" % (abs(r_dG), abs(r_nm), m_nm, v["P3"]))
    if p_dG >= 0.05:
        print("  NOTE: p = %.3f >= 0.05, so no claim is made on this correlation"
              % p_dG)
    ok = sum(1 for x in v.values() if x == "SUPPORTED")
    print("  %d of 3 supported: %s" % (ok, v))

    json.dump({"rows": rows, "n": len(rows), "margin_required": MARGIN,
               "rho_dG": r_dG, "p_dG": p_dG, "rho_traj_rmse": r_tr,
               "p_traj_rmse": p_tr, "rho_norm": r_nm, "p_norm": p_nm,
               "margin_over_traj": m_tr, "margin_over_norm": m_nm,
               "verdicts": v},
              open(os.path.join(ROOT, "results", "h9_score.json"), "w"), indent=1)
    print("\n[out] results/h9_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
