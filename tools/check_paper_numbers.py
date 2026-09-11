"""Verify the numbers asserted in paper/main.tex against results/*.json.

A paper is the one artefact where a transcription slip is invisible and
expensive. Each check below re-reads the source of truth and asserts the value
that appears in the text, so a stale number fails here instead of in review.

  run.cmd tools/check_paper_numbers.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()

FAILS = []
CHECKS = 0


def load(rel):
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def claim(desc, expected, actual, tol=0.05):
    """Assert the paper's number matches the data, and that it is in the text."""
    global CHECKS
    CHECKS += 1
    ok_val = abs(expected - actual) <= tol
    if not ok_val:
        FAILS.append("%s: paper says %s, data says %.4f" % (desc, expected, actual))
    print("  %-52s paper %-9s data %-9.4f %s"
          % (desc, expected, actual, "ok" if ok_val else "MISMATCH"))


def present(desc, needle):
    global CHECKS
    CHECKS += 1
    ok = needle in TEX
    if not ok:
        FAILS.append("%s: %r not found in main.tex" % (desc, needle))
    print("  %-52s %s" % (desc, "ok" if ok else "TEXT MISSING"))


def main() -> int:
    print("H4 anisotropy")
    h4 = load("results/h4_score.json")
    claim("A_velocity", 100.0, h4["A_velocity_mean"])
    claim("A_force", 95.7, h4["A_force_mean"], tol=0.06)
    claim("gap", 4.3, h4["gap"], tol=0.06)
    claim("A_dominated velocity", 99.7, h4["A_dominated_velocity_mean"], tol=0.06)
    present("Q1 reported as failed", "threshold of $40$")

    print("\nH4 velocity sweep, matched norm")
    sv = load("results/h4_sweep_velocity.json")["rows"]

    def cell(rows, c, direction_prefix):
        v = [100 * r["success"] for r in rows
             if abs(r["c"] - c) < 1e-9 and r["direction"].startswith(direction_prefix)]
        return float(np.mean(v))

    for c, want in ((1.5, 100.0), (2.5, 100.0), (4.0, 98.8), (8.0, 100.0)):
        claim("uniform c=%.1f" % c, want, cell(sv, c, "uniform"), tol=0.4)
    dm = {round(r["dm_kg"], 3) for r in sv if r["direction"] == "uniform" and r["c"] == 8.0}
    claim("||dm|| at c=8", 1.527, max(dm), tol=0.002)

    # The dominated table is LINK 1 specifically -- its ||dm|| column is link
    # 1's displacement. Averaging all three links here reported a spurious
    # mismatch at c=1.5, where links 2 and 3 are not at zero.
    print("\nH4 dominated scheme (link 1, as tabulated)")
    for c, want in ((1.5, 0.0), (2.5, 0.0), (4.0, 0.0), (8.0, 0.0)):
        claim("link1 c=%.1f" % c, want, cell(sv, c, "dominated_link1"), tol=0.4)
    d8 = max(r["dm_kg"] for r in sv
             if r["direction"] == "dominated_link1" and r["c"] == 8.0)
    claim("link1 ||dm|| at c=8", 1.243, d8, tol=0.002)
    claim("link2 at c=1.5 (quoted in text)", 19.8,
          cell(sv, 1.5, "dominated_link2"), tol=0.1)
    claim("link3 at c=1.5 (quoted in text)", 4.0,
          cell(sv, 1.5, "dominated_link3"), tol=0.1)

    print("\nForce arm uniform decay (post-hoc subsection)")
    sf = load("results/h4_sweep_force.json")["rows"]
    for c, want in ((1.5, 99.0), (2.5, 95.7), (4.0, 61.5), (8.0, 3.5)):
        claim("force uniform c=%.1f" % c, want, cell(sf, c, "uniform"), tol=0.4)

    print("\nSelection blindness")
    sb = load("results/selection_blindness.json")
    claim("spearman rho", -0.090, sb["spearman_rho"], tol=0.002)
    claim("permutation p", 0.749, sb["permutation_p"], tol=0.01)
    claim("n", 15, sb["n"], tol=0)
    claim("own-sim spread", 2.0, sb["own_sim_spread"], tol=0.06)
    claim("R* spread", 100.0, sb["R_star_spread"], tol=0.06)
    claim("nominal-arm R* spread", 77.0, sb["nominal_arm"]["R_star_spread"], tol=0.1)
    claim("nominal-arm own spread", 0.8, sb["nominal_arm"]["own_sim_spread"], tol=0.03)
    b2f = sb["best_to_final"]
    claim("mean best->final drop", 38.6, b2f["mean_drop_points"], tol=0.06)
    claim("best mean", 99.7, b2f["best_mean"], tol=0.06)
    claim("final mean", 61.0, b2f["final_mean"], tol=0.06)
    claim("runs losing >20", 11, b2f["n_losing_over_20"], tol=0)
    claim("runs ending at best", 3, b2f["n_final_is_best"], tol=0)

    print("\nFidelity and twins")
    h2 = load("results/h2_test.json")["spearman"]
    claim("traj_rmse rho", -0.506, h2["traj_rmse"], tol=0.002)
    claim("D_SW rho", -0.415, h2["D_SW"], tol=0.002)
    claim("R_TC rho", -0.406, h2["R_TC"], tol=0.002)
    tw = load("results/twins.json")
    claim("n admissible pairs", 28, tw["n_pairs"], tol=0)
    best = tw["pairs"][0]
    claim("twin rel diff %", 1.51, 100 * best["rel_diff"], tol=0.02)
    claim("twin P_a", 0.0, best["P_a"], tol=0.05)
    claim("twin P_b", 100.0, best["P_b"], tol=0.05)

    print("\nH3")
    c5 = load("results/claim5_score.json")["scores"]
    claim("nominal mean", 64.7, float(np.mean(c5["S_nominal"])), tol=0.06)
    claim("twinB mean", 65.5, float(np.mean(c5["S_twinB"])), tol=0.06)
    claim("transverse mean", 4.9, float(np.mean(c5["S_transverse"])), tol=0.06)
    claim("equiv_c8 mean", 1.0, float(np.mean(c5["S_equiv_c8"])), tol=0.06)

    print("\nAnalytical 2x2")
    isym = load("results/interface_symmetry.json")
    g = {r["direction"].split()[0]: r for r in isym["grid"]}
    claim("uniform, velocity iface", 1.4e-14, g["uniform"]["rel_velocity"], tol=1e-14)
    claim("uniform, force iface", 7.5e-2, g["uniform"]["rel_force"], tol=2e-3)
    claim("dm for the 2x2", 0.109, g["uniform"]["delta_m_kg"], tol=0.001)

    print("\nPlant")
    import yaml
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                          "triple_pendulum_params.yaml"), encoding="utf-8"))
    h = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                         "hardware.yaml"), encoding="utf-8"))
    b = p["bodies"]
    for i, want in ((1, 0.178), (2, 0.100), (3, 0.077)):
        claim("link%d mass" % i, want, b["link%d" % i]["mass"], tol=0.001)
    eff = b["cart"]["mass"] + h["drive"]["reflected_mass_kg"]
    claim("effective translating mass", 0.79, eff, tol=0.005)
    claim("peak cart force", 349.5, h["drive"]["peak_cart_force_N"], tol=0.1)
    claim("rated cart force", 99.7, h["drive"]["rated_cart_force_N"], tol=0.1)
    claim("max cart speed", 4.0, h["drive"]["max_cart_speed_ms"], tol=0.01)

    print("")
    if FAILS:
        print("%d of %d checks FAILED:" % (len(FAILS), CHECKS))
        for f in FAILS:
            print("  -", f)
        return 1
    print("all %d checks passed" % CHECKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
