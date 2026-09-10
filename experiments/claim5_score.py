"""Score H3: does simulator-error geometry change what RL learns?

Reads the deployment evaluations produced by tools/claim5_deploy.ps1 and applies
the predictions frozen in PREREGISTRATION_H3.md (commit 3d453a0, before any
policy was trained). Written before any deployment number exists, so the
analysis cannot be shaped by the result.

  P1  directional. S_equiv_c8 lies far further from R* in parameter distance
      than S_transverse -- an 8x mass error against 1.5x -- but along a
      direction the passive dynamics provably cannot see. Predicted:
      pi_equiv_c8 beats pi_transverse by >= 20 points on seed means. If
      magnitude governed transfer this must come out the other way.

  P2  NON-directional, deliberately. The twins are 1.5% apart in trajectory
      RMSE yet gave 0.0% and 100.0% for the frozen policy. As training
      simulators they should differ by >= 30 points. No direction is predicted;
      the geometry gives no principled reason to say which twin trains the
      better policy.

  P3  control. Nominal should be competitive with transverse, else the protocol
      is at fault rather than the theory.

Seed is the experimental unit. With 2-3 seeds per arm the power is low, and any
comparison whose seed spread overlaps the claimed threshold is reported as
INCONCLUSIVE rather than as support -- that rule is fixed here, in advance.

  run.cmd experiments/claim5_score.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1_MIN = 20.0
P2_MIN = 30.0


def arm_scores() -> dict:
    """Deployment success in R*, per simulator, one entry per seed."""
    out: dict = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "claim5", "*.json"))):
        name = os.path.basename(f)[:-5]                # C5_S_xxx_sN
        sim = name.rsplit("_s", 1)[0].replace("C5_", "")
        d = json.load(open(f, encoding="utf-8"))
        out.setdefault(sim, []).append(100.0 * d["results"][0]["success_rate"])
    return out


def summary(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.min(), v.max())


def compare(a, b, thresh, name_a, name_b, directional):
    """Returns (verdict, detail). Overlapping seed ranges -> INCONCLUSIVE."""
    ma, ra = summary(a)
    mb, rb = summary(b)
    gap = ma - mb if directional else abs(ma - mb)
    # Two ranges are DISJOINT when one starts above where the other ends.
    # The earlier form of this test had an extra min() guard that made it
    # report overlap whenever arm A sat entirely ABOVE arm B. Neither P1 nor
    # P2 was in that configuration, so the reported verdicts are unchanged --
    # verified in the commit message -- but the bug's direction was to call a
    # clean separation "inconclusive", i.e. to under-claim.
    overlap = not (ra[0] > rb[1] or rb[0] > ra[1])
    detail = ("%s %.1f%% (seeds %.1f-%.1f)  vs  %s %.1f%% (seeds %.1f-%.1f)  gap %+.1f"
              % (name_a, ma, ra[0], ra[1], name_b, mb, rb[0], rb[1], gap))
    if gap < thresh:
        return ("FAIL" if not overlap else "FAIL/INCONCLUSIVE"), detail
    if overlap:
        return "INCONCLUSIVE", detail + "   [seed ranges overlap]"
    return "SUPPORTED", detail


def main() -> int:
    scores = arm_scores()
    if not scores:
        print("no deployment results in results/claim5/ yet")
        return 1

    rstar = json.load(open(os.path.join(ROOT, "xi", "R_star.json"), encoding="utf-8"))
    print("deployment target R* = %s\n" % json.dumps(rstar))
    print("  training simulator   seeds   success in R* (per seed)      mean")
    print("  " + "-" * 68)
    for sim in sorted(scores):
        v = scores[sim]
        print("  %-20s %5d   %-28s %6.1f%%"
              % (sim, len(v), ", ".join("%.1f" % x for x in sorted(v, reverse=True)), np.mean(v)))

    print("\nPRE-REGISTERED PREDICTIONS (PREREGISTRATION_H3.md, commit 3d453a0)\n")
    verdicts = {}
    need = ("S_equiv_c8", "S_transverse", "S_twinA", "S_twinB", "S_nominal")
    missing = [n for n in need if n not in scores]
    if missing:
        print("  missing arms: %s -- cannot score" % ", ".join(missing))
        return 1

    v, d = compare(scores["S_equiv_c8"], scores["S_transverse"], P1_MIN,
                   "equiv_c8", "transverse", directional=True)
    verdicts["P1"] = v
    print("  P1  equivalence beats transverse by >= %.0f pts" % P1_MIN)
    print("      %s\n      -> %s\n" % (d, v))

    v, d = compare(scores["S_twinA"], scores["S_twinB"], P2_MIN,
                   "twinA", "twinB", directional=False)
    verdicts["P2"] = v
    print("  P2  fidelity twins differ by >= %.0f pts (no direction predicted)" % P2_MIN)
    print("      %s\n      -> %s\n" % (d, v))

    mn, mt = np.mean(scores["S_nominal"]), np.mean(scores["S_transverse"])
    v3 = "SUPPORTED" if mn >= mt else "FAIL -- protocol suspect"
    verdicts["P3"] = v3
    print("  P3  control: nominal competitive with transverse")
    print("      nominal %.1f%% vs transverse %.1f%%\n      -> %s\n" % (mn, mt, v3))

    supported = sum(1 for x in verdicts.values() if x == "SUPPORTED")
    print("  %d of 3 supported: %s" % (supported, verdicts))
    if verdicts["P1"] != "SUPPORTED" or verdicts["P2"] != "SUPPORTED":
        print("\n  H3 is not fully supported. The honest reading is that error geometry")
        print("  governs the robustness of a FIXED policy but does not measurably change")
        print("  what RL learns -- narrower than the paper wants, and reported as such.")

    json.dump({"scores": scores, "verdicts": verdicts, "R_star": rstar,
               "P1_min": P1_MIN, "P2_min": P2_MIN},
              open(os.path.join(ROOT, "results", "claim5_score.json"), "w"), indent=1)
    print("\n[out] results/claim5_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
