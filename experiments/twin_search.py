"""Do trajectory-fidelity twins exist? Pre-registered thresholds.

The claim under test is NOT that trajectory fidelity is useless -- our own data
contradicts that, since short-window trajectory RMSE was the best aggregate
predictor in the heterogeneous test (rho = -0.506, results/h2_test.json). The
claim is the sharper one:

    trajectory fidelity is informative statistically but cannot CERTIFY
    transfer: two simulators of indistinguishable fidelity can produce opposite
    control outcomes.

That requires finding pairs that are equivalent by the fidelity measure and
opposite by outcome. Which means the tolerance has to be fixed FIRST, because
picking the most flattering pair and then choosing a threshold around it would
manufacture the result.

FROZEN BEFORE RUNNING (this file is committed before it is executed):

  TOL_REL   = 0.10   two conditions count as fidelity-equivalent when their
                     short-window trajectory RMSE differs by <= 10% of the
                     larger. 10% is the order of the spread already seen ACROSS
                     initial conditions within one condition, so it is below
                     what the measure itself resolves.
  MIN_GAP   = 50     the claim needs a transfer gap of at least 50 percentage
                     points to count as "opposite outcomes"; anything less is a
                     quantitative difference, not a qualitative one.
  PLAUSIBLE = both members must be flagged plausible for the real build, so a
                     twin cannot rest on a deliberately absurd condition.

KILL CONDITION: if no admissible pair reaches MIN_GAP, the claim is dropped and
reported as unsupported rather than restated with a looser tolerance.

  run.cmd experiments/twin_search.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOL_REL = 0.10
MIN_GAP = 50.0
REQUIRE_PLAUSIBLE = True
FIDELITY = "traj_rmse"      # the measure that WON the heterogeneous test


def main() -> int:
    h2 = json.load(open(os.path.join(ROOT, "results", "h2_test.json"), encoding="utf-8"))
    rows = h2["rows"]
    print("pre-registered: tol %.0f%% relative, min gap %.0f points, plausible-only %s"
          % (100 * TOL_REL, MIN_GAP, REQUIRE_PLAUSIBLE))
    print("fidelity measure: %s (the strongest baseline, rho = %+.3f)\n"
          % (FIDELITY, h2["spearman"][FIDELITY]))

    cand = [r for r in rows if (r["plausible"] or not REQUIRE_PLAUSIBLE)]
    print("%d of %d conditions admissible\n" % (len(cand), len(rows)))

    pairs = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            ea, eb = a[FIDELITY], b[FIDELITY]
            rel = abs(ea - eb) / max(ea, eb, 1e-12)
            if rel > TOL_REL:
                continue
            gap = abs(a["success"] - b["success"]) * 100
            pairs.append({"a": a["name"], "b": b["name"],
                          "family_a": a["family"], "family_b": b["family"],
                          "E_a": ea, "E_b": eb, "rel_diff": rel,
                          "P_a": 100 * a["success"], "P_b": 100 * b["success"],
                          "gap": gap})
    pairs.sort(key=lambda p: -p["gap"])
    print("%d fidelity-equivalent pairs within tolerance\n" % len(pairs))

    if not pairs:
        print("NO admissible pairs -- claim dropped.")
        return 1

    print("  strongest pairs by transfer gap:")
    print("  %-19s %-19s   E_a      E_b     rel     P_a     P_b     gap")
    print("  " + "-" * 88)
    for p in pairs[:12]:
        print("  %-19s %-19s %7.4f  %7.4f  %4.1f%%  %5.1f%%  %5.1f%%  %5.1f"
              % (p["a"], p["b"], p["E_a"], p["E_b"], 100 * p["rel_diff"],
                 p["P_a"], p["P_b"], p["gap"]))

    best = pairs[0]
    ok = best["gap"] >= MIN_GAP
    print("\n  best gap: %.1f points (threshold %.0f) -> %s"
          % (best["gap"], MIN_GAP, "CLAIM SUPPORTED" if ok else "CLAIM DROPPED"))
    if ok:
        print("\n  %s and %s have trajectory RMSE within %.1f%% of each other" %
              (best["a"], best["b"], 100 * best["rel_diff"]))
        print("  yet transfer at %.1f%% and %.1f%%." % (best["P_a"], best["P_b"]))
        cross = sum(1 for p in pairs if p["gap"] >= MIN_GAP and p["family_a"] != p["family_b"])
        print("  %d of %d qualifying pairs span DIFFERENT model families."
              % (cross, sum(1 for p in pairs if p["gap"] >= MIN_GAP)))

    json.dump({"tolerance_rel": TOL_REL, "min_gap_points": MIN_GAP,
               "fidelity": FIDELITY, "plausible_only": REQUIRE_PLAUSIBLE,
               "n_pairs": len(pairs), "supported": bool(ok), "pairs": pairs[:40]},
              open(os.path.join(ROOT, "results", "twins.json"), "w"), indent=1)
    print("\n[out] results/twins.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
