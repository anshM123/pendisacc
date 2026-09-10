"""Score H5 against the predictions frozen in PREREGISTRATION_H5.md.

Written before any H5 policy is trained.

  P1  (primary)  geom - box >= 15 points, on seed means
  P2             both randomised arms beat none, so the budget buys something
                 at all -- otherwise the experiment says nothing about how to
                 SPEND a budget that does not work

  KILL           box >= geom. The geometry would not be actionable and the
                 practical recommendation is withdrawn; the diagnosis in H4
                 stands either way.
  INCONCLUSIVE   per-seed ranges overlap the claimed gap. Expected: the H3
                 nominal arm showed a 77-point within-arm spread at identical
                 settings, 5x the effect claimed here. Said in advance so an
                 underpowered result is reported as unresolved, not absent.

BUDGET MATCHING IS VERIFIED, NOT ASSUMED. Each run writes dr_applied.json with
the REALISED E||dm|| and the realised leakage E|dm.u|. Projection removes ~32%
of the budget, so geom's width is scaled by 1.478; if that correction were
wrong the arms would differ in how much they randomise and P1 would be
meaningless. This refuses to score unless the realised budgets agree.

  run.cmd experiments/h5_score.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1_MIN = 15.0
BUDGET_TOL = 0.02          # realised E||dm|| of box and geom must agree to 2%
ARMS = ("none", "box", "geom")


def deployments():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "h5", "H5_*.json"))):
        tag = os.path.basename(f)[:-5]              # H5_<arm>_s<N>
        arm = tag.split("_")[1]
        d = json.load(open(f, encoding="utf-8"))
        out.setdefault(arm, []).append(100.0 * d["results"][0]["success_rate"])
    return out


def budgets():
    """Realised randomisation budget per run, from dr_applied.json."""
    out = {}
    root = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup")
    for d in sorted(glob.glob(os.path.join(root, "*_H5_*"))):
        p = os.path.join(d, "dr_applied.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        out.setdefault(j["mode"], []).append(j)
    return out


def fmt(v):
    return ", ".join("%.1f" % x for x in sorted(v, reverse=True))


def main() -> int:
    scores = deployments()
    missing = [a for a in ARMS if a not in scores]
    if missing:
        print("no deployment results for: %s" % ", ".join(missing))
        return 1

    # ---- budget check first: it can invalidate the whole comparison --------
    bud = budgets()
    print("REALISED RANDOMISATION BUDGET (from dr_applied.json, per run)\n")
    print("  arm     runs   E||dm|| [kg]        E|dm.u| [kg]   leakage along u")
    print("  " + "-" * 68)
    means = {}
    for arm in ("box", "geom"):
        js = bud.get(arm, [])
        if not js:
            print("  %-7s  --    (no dr_applied.json found)" % arm)
            continue
        e = float(np.mean([j["E_norm_dm_kg"] for j in js]))
        a = float(np.mean([j["E_abs_dm_dot_u"] for j in js]))
        means[arm] = e
        print("  %-7s %4d   %.6f            %.3e     %s"
              % (arm, len(js), e, a, "projected out" if a < 1e-9 else "present"))
    ok_budget = None
    if "box" in means and "geom" in means:
        rel = abs(means["geom"] - means["box"]) / max(means["box"], 1e-12)
        ok_budget = rel <= BUDGET_TOL
        print("\n  box vs geom budget differ by %.2f%% (tolerance %.0f%%) -> %s"
              % (100 * rel, 100 * BUDGET_TOL, "OK" if ok_budget else "MISMATCH"))
        if not ok_budget:
            print("\n  REFUSING TO SCORE. The two arms did not spend the same budget,")
            print("  so any difference between them confounds geometry with amount")
            print("  of randomisation. Fix GEOM_WIDTH_FACTOR and retrain.")
            return 1

    print("\n\nDEPLOYMENT INTO R* (the target frozen for H3)\n")
    print("  arm      seeds   success in R* (per seed)     mean")
    print("  " + "-" * 60)
    for arm in ARMS:
        v = scores[arm]
        print("  %-8s %5d   %-26s %6.1f%%" % (arm, len(v), fmt(v), np.mean(v)))

    mb, mg, mn = (float(np.mean(scores[a])) for a in ("box", "geom", "none"))
    gap = mg - mb
    overlap = not (min(scores["geom"]) > max(scores["box"])
                   or min(scores["box"]) > max(scores["geom"]))

    print("\nPRE-REGISTERED VERDICTS (PREREGISTRATION_H5.md)\n")
    verdicts = {}
    if mb >= mg:
        verdicts["P1"] = "FAIL -- KILL CONDITION MET"
    elif gap < P1_MIN:
        verdicts["P1"] = "FAIL"
    elif overlap:
        verdicts["P1"] = "INCONCLUSIVE -- per-seed ranges overlap"
    else:
        verdicts["P1"] = "SUPPORTED"
    print("  P1  geom - box >= %.0f points" % P1_MIN)
    print("      geom %.1f%% (seeds %s)" % (mg, fmt(scores["geom"])))
    print("      box  %.1f%% (seeds %s)" % (mb, fmt(scores["box"])))
    print("      gap  %+.1f\n      -> %s\n" % (gap, verdicts["P1"]))

    verdicts["P2"] = "SUPPORTED" if (mb > mn and mg > mn) else "FAIL"
    print("  P2  both randomised arms beat none (%.1f%%)" % mn)
    print("      box %+.1f, geom %+.1f\n      -> %s\n"
          % (mb - mn, mg - mn, verdicts["P2"]))
    if verdicts["P2"] != "SUPPORTED":
        print("      A budget that buys nothing says nothing about how to spend it,")
        print("      so P1 should be read as uninterpretable rather than as a")
        print("      finding about geometry.\n")

    if verdicts["P1"].startswith("FAIL -- KILL"):
        print("  KILL CONDITION MET. Geometry-aware randomisation did not beat")
        print("  isotropic randomisation at equal budget. The practical")
        print("  recommendation is withdrawn; the H4 diagnosis is unaffected.\n")

    print("  %d of 2 supported: %s"
          % (sum(1 for v in verdicts.values() if v == "SUPPORTED"), verdicts))

    json.dump({"scores": scores, "verdicts": verdicts, "P1_min": P1_MIN,
               "means": {"none": mn, "box": mb, "geom": mg}, "gap": gap,
               "seed_ranges_overlap": bool(overlap),
               "realised_budgets": means, "budget_ok": ok_budget},
              open(os.path.join(ROOT, "results", "h5_score.json"), "w"), indent=1)
    print("\n[out] results/h5_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
