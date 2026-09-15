"""Apply PREREGISTRATION_FRACTAL.md's decision rule to the Isaac runs. CPU only.

  run.cmd tools/fractal_analyze.py [--grid results/fractal/grid_A.json] [--pairs ...] [--repeat ...]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def grid_stats(g, rep=None):
    n = g["meta"]["n"]
    S = np.array(g["success"], int)
    H = np.array(g["hold"], float)
    main, dups = S[: n * n], S[n * n:]
    d0 = float(np.mean(main[np.array(g["meta"]["dup_src"])] != dups))
    M = main.reshape(n, n)
    # boundary cells: any 4-neighbour with a different outcome
    diff = np.zeros_like(M)
    diff[1:, :] += M[1:, :] != M[:-1, :]
    diff[:-1, :] += M[:-1, :] != M[1:, :]
    diff[:, 1:] += M[:, 1:] != M[:, :-1]
    diff[:, :-1] += M[:, :-1] != M[:, 1:]
    k = int(np.argmax(diff.ravel()))
    xs, ys = g["meta"]["xs"], g["meta"]["ys"]
    out = {"success_frac": float(main.mean()), "d0_within_run": d0,
           "boundary_cell_frac": float(np.mean(diff > 0)),
           "zoom_centre": [xs[k // n], ys[k % n]], "zoom_centre_disagreeing_neighbours": int(diff.ravel()[k]),
           "hold_mean": float(H[: n * n].mean())}
    if rep is not None:
        out["r0_run_to_run"] = float(np.mean(np.array(rep["success"][: n * n]) != main))
    return out, M


def pair_stats(pz):
    eps = pz["meta"]["eps"]
    P = pz["meta"]["pairs"]
    S = np.array(pz["success"], int)
    H = np.array(pz["hold"], float)
    rows = []
    for i, e in enumerate(eps):
        a, b = S[2 * i * P: 2 * i * P + P], S[2 * i * P + P: 2 * (i + 1) * P]
        ha, hb = H[2 * i * P: 2 * i * P + P], H[2 * i * P + P: 2 * (i + 1) * P]
        flip = a != b
        straddle = flip & (np.abs(ha - 0.95) < 0.05) & (np.abs(hb - 0.95) < 0.05)
        f = float(flip.mean())
        rows.append({"eps": e, "f": f, "se": float(np.sqrt(max(f * (1 - f), 1e-12) / P)),
                     "flips": int(flip.sum()), "straddle_frac": float(straddle.sum() / max(flip.sum(), 1)),
                     "base_success": float(a.mean())})
    p0 = rows[0]
    usable = [r for r in rows[1:] if r["f"] - p0["f"] >= 3 * np.hypot(r["se"], p0["se"])]
    alpha = None
    if len(usable) >= 2:
        x = np.log([r["eps"] for r in usable])
        y = np.log([r["f"] - p0["f"] for r in usable])
        alpha = float(np.polyfit(x, y, 1)[0])
    ci = None
    if alpha is not None:
        rng = np.random.default_rng(0)
        flips = [(S[2 * i * P: 2 * i * P + P] != S[2 * i * P + P: 2 * (i + 1) * P]) for i in range(len(eps))]
        ue = [eps.index(e) for e in [r["eps"] for r in usable]]
        boots = []
        for _ in range(1000):
            f0 = flips[0][rng.integers(0, P, P)].mean()
            ys, xs = [], []
            for i in ue:
                fi = flips[i][rng.integers(0, P, P)].mean() - f0
                if fi > 0:
                    ys.append(np.log(fi)); xs.append(np.log(eps[i]))
            if len(xs) >= 2:
                boots.append(np.polyfit(xs, ys, 1)[0])
        if boots:
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"rows": rows, "p0": p0["f"], "usable_eps": [r["eps"] for r in usable], "alpha": alpha,
            "alpha_ci95": ci, "C_half": (2 ** (1 / alpha)) if alpha else None,
            "D": (2 - alpha) if alpha is not None else None,
            "straddle_at_smallest_usable": (min(usable, key=lambda r: r["eps"])["straddle_frac"] if usable else None)}


def decide(gs, ps):
    reasons = []
    if gs.get("r0_run_to_run") is not None and gs["r0_run_to_run"] >= 0.05:
        reasons.append("r0 >= 5%: simulator too nondeterministic")
    if ps["alpha"] is None or len(ps["usable_eps"]) < 3:
        reasons.append("fewer than 3 usable eps above the noise floor")
    elif ps["alpha"] >= 0.8:
        reasons.append("alpha >= 0.8: boundary behaves as smooth")
    if ps["straddle_at_smallest_usable"] is not None and ps["straddle_at_smallest_usable"] > 0.5:
        reasons.append("threshold-jitter control failed")
    if reasons:
        return "KILL", reasons
    if ps["alpha"] <= 0.5 and len(ps["usable_eps"]) >= 4:
        return "CANDIDATE (needs zoom + second policy for LOCK IN)", []
    return "CONTINUE (rough boundary; zoom test only)", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/fractal/grid_A.json")
    ap.add_argument("--repeat", default="results/fractal/grid_B.json")
    ap.add_argument("--pairs", default="results/fractal/pairs.json")
    ap.add_argument("--tag", default="gate")
    a = ap.parse_args()
    g, rep, pz = load(a.grid), load(a.repeat), load(a.pairs)
    res = {}
    if g:
        gs, M = grid_stats(g, rep)
        res["grid"] = gs
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs, ys = g["meta"]["xs"], g["meta"]["ys"]
            fig, ax = plt.subplots(figsize=(5, 4.6))
            ax.imshow(M.T, origin="lower", cmap="gray", extent=[xs[0], xs[-1], ys[0], ys[-1]], interpolation="nearest")
            ax.set_xlabel("log m1 scale"); ax.set_ylabel("log m3 scale")
            ax.set_title("frozen policy transfer map (white = success)")
            os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
            fig.savefig(os.path.join(ROOT, "figures", "fractal_%s_map.png" % a.tag), dpi=160, bbox_inches="tight")
        except Exception as e:  # figure is optional
            res["figure_error"] = str(e)
    if pz:
        res["pairs"] = pair_stats(pz)
    if g and pz:
        res["decision"], res["kill_reasons"] = decide(res["grid"], res["pairs"])
    os.makedirs(os.path.join(ROOT, "results", "fractal"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "fractal", "analysis_%s.json" % a.tag), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
