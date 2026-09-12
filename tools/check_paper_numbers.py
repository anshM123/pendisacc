"""Verify the numbers asserted in paper/main.tex against results/*.json.

A paper is the one artefact where a transcription slip is invisible and
expensive, so every figure quoted in the text is re-read from its source here
and asserted. Rewritten for the symmetry-group paper.

  run.cmd tools/check_paper_numbers.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
FAILS, N = [], 0


def load(rel):
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def claim(desc, paper, data, tol=0.05):
    global N
    N += 1
    ok = abs(paper - data) <= tol
    if not ok:
        FAILS.append("%s: paper %s, data %.4f" % (desc, paper, data))
    print("  %-46s paper %-10s data %-10.4f %s"
          % (desc, paper, data, "ok" if ok else "MISMATCH"))


def main() -> int:
    print("anisotropy, original asset")
    sv = load("results/h4_sweep_velocity.json")["rows"]

    def cell(rows, c, pre):
        return float(np.mean([100 * r["success"] for r in rows
                              if abs(r["c"] - c) < 1e-9 and r["direction"].startswith(pre)]))

    for c, w in ((1.5, 100.0), (2.5, 100.0), (4.0, 98.8), (8.0, 100.0)):
        claim("uniform c=%.1f" % c, w, cell(sv, c, "uniform"), 0.4)
    claim("dominated link1 c=8", 0.0, cell(sv, 8.0, "dominated_link1"), 0.4)
    h4 = load("results/h4_score.json")
    claim("A_velocity", 100.0, h4["A_velocity_mean"])
    claim("A_force", 95.7, h4["A_force_mean"], 0.06)
    claim("A gap", 4.3, h4["gap"], 0.06)

    print("\nT4 landscape, corrected asset")
    g = load("results/T4/landscape_corrected.json")
    cs, ds = g["c_values"], g["deltas"]
    P = np.array([[np.mean([100 * r["success"] for r in g["rows"]
                            if r["c"] == c and r["delta"] == d]) for c in cs] for d in ds])
    gm = P.mean()
    ss = ((P - gm) ** 2).sum()
    ssd = len(cs) * ((P.mean(axis=1) - gm) ** 2).sum()
    ssc = len(ds) * ((P.mean(axis=0) - gm) ** 2).sum()
    claim("variance by delta", 97.6, 100 * ssd / ss, 0.1)
    claim("variance by c", 0.7, 100 * ssc / ss, 0.1)
    sat = max(max(r["force_sat_frac"], r["speed_sat_frac"]) for r in g["rows"])
    claim("worst saturation pct", 0.007, 100 * sat, 0.001)

    print("\nsymmetry group")
    sg = load("results/symmetry_group.json")
    claim("group dim, velocity+dissipation", 3,
          sg["velocity_loop_dissipative_fixed_rate"]["dim"], 0)

    print("\nT1 null space")
    t1 = load("results/T1/nullspace.json")["rows"]
    claim("control c=1", 99.6, 100 * [r for r in t1 if r["name"] == "control_c1"][0]["success"], 0.4)
    claim("group direction", 100.0, 100 * [r for r in t1 if r["name"] == "GROUP"][0]["success"], 0.4)
    rnd = [r for r in t1 if r["name"].startswith("rand")]
    claim("n random >= 90pct", 0, sum(1 for r in rnd if r["success"] >= 0.90), 0)
    claim("worst random", 0.8, 100 * max(r["success"] for r in rnd), 0.4)
    claim("best aligned random, cos", 0.790,
          max(r["cos_to_group"] for r in rnd), 0.02)

    print("\nT2 policy independence")
    t2 = load("results/T2/policy_independence.json")["rows"]
    for r in t2:
        claim("%s at c=64" % r["policy"], 100.0 if r["policy"] == "CORR_s1" else 99.6,
              100 * r["success"][-1], 0.4)
    drops = [100 * (r["success"][0] - r["success"][-1]) for r in t2]
    claim("drop spread", 0.4, max(drops) - min(drops), 0.1)

    # T5 is WITHHELD from the paper pending re-measurement (the orbit and
    # transverse arms fell through to a second, isotropic perturbation, so the
    # transverse arm carried a strictly larger displacement). Nothing to check
    # while the section is out. This block stays as a marker: when the section
    # returns, its numbers must be asserted here before the paper ships.
    print("")
    print("T5 orbit vs transverse: WITHHELD pending re-measurement")

    print("\nH10 realisability")
    h10 = load("results/h10_score.json")["results"]
    claim("full c=16 + clamp", 100.0, h10["full_c16_clamp"], 0.4)
    claim("full c=32 + clamp", 99.6, h10["full_c32_clamp"], 0.4)
    claim("full c=64 + clamp", 100.0, h10["full_c64_clamp"], 0.4)
    claim("full c=32, NO clamp", 0.0, h10["full_c32"], 0.4)

    print("\nCAD defect")
    a = load("results/cad_density_audit.json")
    claim("scale c", 0.744, a["overall_scale_c"], 0.002)
    claim("ratio distortion", 2.21, a["ratio_distortion"], 0.01)
    for i, w in enumerate((0.81, 1.06, 0.48)):
        claim("link%d factor" % (i + 1), w, a["scale_factors_mid"][i], 0.006)
    h8 = load("results/h8_score.json")["results"]
    claim("scale only", 100.0, h8["scale_only"], 0.4)
    claim("ratio only", 1.2, h8["ratio_only"], 0.4)
    claim("real defect", 0.8, h8["corrected_mid"], 0.4)

    print("\nfailures")
    t3 = load("results/T3/quotient.json")
    claim("rho d_perp", -0.243, t3["rho_perp"], 0.002)
    claim("rho traj_rmse", -0.532, t3["rho_traj"], 0.002)
    claim("rho along group", 0.065, t3["rho_along"], 0.002)
    claim("rho fitted metric (H9)", -0.322, load("results/h9_score.json")["rho_dG"], 0.002)
    sb = load("results/selection_blindness.json")
    claim("rho own-sim vs transfer", -0.090, sb["spearman_rho"], 0.002)
    claim("own-sim spread", 2.0, sb["own_sim_spread"], 0.06)
    claim("R* spread", 100.0, sb["R_star_spread"], 0.06)
    claim("nominal-arm R* spread", 77.0, sb["nominal_arm"]["R_star_spread"], 0.1)
    h7 = load("results/h7_score.json")
    claim("rho own-sim, non-degenerate", -0.570, h7["rho_own_nondegenerate"], 0.002)
    tw = load("results/twins.json")["pairs"][0]
    claim("twin rel diff pct", 1.51, 100 * tw["rel_diff"], 0.02)
    c5 = load("results/claim5_score.json")["scores"]
    claim("H3 equiv_c8 mean", 1.0, float(np.mean(c5["S_equiv_c8"])), 0.06)
    claim("H3 transverse mean", 4.9, float(np.mean(c5["S_transverse"])), 0.06)

    print("")
    if FAILS:
        print("%d of %d checks FAILED:" % (len(FAILS), N))
        for f in FAILS:
            print("  -", f)
        return 1
    print("all %d checks passed" % N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
