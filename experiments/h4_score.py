"""Score H4 against the predictions frozen in PREREGISTRATION_H4.md.

Written before the force arm finished training, so the analysis cannot be
shaped by the result. The pre-registration fixes:

  A = success(uniform) - mean success(transverse) at MATCHED ||dm||, at c=2.5
      -- the anisotropy: how much the DIRECTION of a model error matters when
      its magnitude is held fixed.

  Q1  A_velocity - A_force >= 40 points          (primary, directional)
  Q2  A_velocity >= 40 points                    (the effect must exist at all)
  Q3  both arms >= 80% at c = 1.0                (competence gate)

  KILL      A_force >= A_velocity
  INCONCL.  a failed competence gate; or a force arm below 20% everywhere,
            making A_force a difference between two floors; or a within-arm
            seed spread of A that overlaps the claimed between-arm gap.

Seed is the experimental unit, not the episode.

  run.cmd experiments/h4_score.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_STAR = 2.5
Q1_MIN = 40.0
Q2_MIN = 40.0
GATE = 80.0        # pre-registered in PREREGISTRATION_H4.md
GATE_STRICT = 95.0 # stricter standard requested after H4 was frozen. Reported
                   # alongside, never in place of: moving a pre-registered
                   # threshold once data exists is how a kill condition gets
                   # quietly repealed.
FLOOR = 20.0


def load(interface):
    p = os.path.join(ROOT, "results", "h4_sweep_%s.json" % interface)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def anisotropy(rows, policy, c=C_STAR, scheme="matched_norm"):
    """A = success(uniform) - mean success(off-axis), in points.

    scheme "matched_norm" is the pre-registered primary: every condition sits
    at the same ||dm||, so only direction varies. scheme "dominated" is the
    robustness check: uniform x c against one link x c, where uniform is the
    strictly LARGER perturbation, so a positive A there cannot be an artefact
    of how the magnitudes were matched.
    """
    sel = [r for r in rows if r["policy"] == policy and abs(r["c"] - c) < 1e-9]
    uni = [r["success"] for r in sel if r["direction"] == "uniform"]
    prefix = "transverse" if scheme == "matched_norm" else "dominated"
    off = [r["success"] for r in sel if r["direction"].startswith(prefix)]
    if not uni or not off:
        return None
    if scheme == "matched_norm":
        dm = {round(r["dm_kg"], 9) for r in sel
              if r["direction"] == "uniform" or r["direction"].startswith(prefix)}
        assert len(dm) == 1, "c=%s not at matched ||dm||: %s" % (c, sorted(dm))
    else:
        # uniform must be the LARGER perturbation for this scheme to mean what
        # it claims; assert it rather than trusting the sweep's arithmetic
        dm_u = max(r["dm_kg"] for r in sel if r["direction"] == "uniform")
        dm_o = max(r["dm_kg"] for r in sel if r["direction"].startswith(prefix))
        assert dm_u > dm_o, ("dominated scheme is not dominated at c=%s: "
                             "uniform ||dm||=%.4f vs off-axis %.4f" % (c, dm_u, dm_o))
    return 100.0 * (float(np.mean(uni)) - float(np.mean(off)))


def nominal(rows, policy):
    v = [r["success"] for r in rows if r["policy"] == policy
         and abs(r["c"] - 1.0) < 1e-9 and r["direction"] == "uniform"]
    return 100.0 * float(np.mean(v)) if v else None


def arm(data):
    rows = data["rows"]
    pols = sorted({r["policy"] for r in rows})
    return [{"policy": p,
             "A": anisotropy(rows, p, scheme="matched_norm"),
             "A_dominated": anisotropy(rows, p, scheme="dominated"),
             "nominal": nominal(rows, p),
             "best_any": 100.0 * max(r["success"] for r in rows if r["policy"] == p)}
            for p in pols]


def fmt(vals):
    return ", ".join("%.1f" % v for v in vals)


def main() -> int:
    dv, df = load("velocity"), load("force")
    if dv is None or df is None:
        print("missing sweep results: need both results/h4_sweep_velocity.json and "
              "results/h4_sweep_force.json")
        return 1

    av, af = arm(dv), arm(df)
    print("H4: does the anisotropy of the reality gap belong to the interface?")
    print("Anisotropy A = success(uniform) - success(transverse) at matched")
    print("||dm||, c = %.1f. Higher A means direction matters more.\n" % C_STAR)
    print("  arm        policy        nominal(c=1)   best any     A(matched)  A(dominated)")
    print("  " + "-" * 78)
    for name, a in (("velocity", av), ("force", af)):
        for r in a:
            ad = "     n/a" if r["A_dominated"] is None else "%+8.1f" % r["A_dominated"]
            print("  %-10s %-13s %7.1f%%      %7.1f%%    %+8.1f  %s"
                  % (name, r["policy"], r["nominal"], r["best_any"], r["A"], ad))

    Av = [r["A"] for r in av]
    Af = [r["A"] for r in af]
    mAv, mAf = float(np.mean(Av)), float(np.mean(Af))
    gap = mAv - mAf
    print("\n  A_velocity  %+.1f  (seeds %s)" % (mAv, fmt(sorted(Av, reverse=True))))
    print("  A_force     %+.1f  (seeds %s)" % (mAf, fmt(sorted(Af, reverse=True))))
    print("  gap         %+.1f  (threshold %.0f)" % (gap, Q1_MIN))

    print("\nPRE-REGISTERED VERDICTS (PREREGISTRATION_H4.md, commit 9aa34a0)\n")
    verdicts = {}

    # ---- Q3 competence gate first: it can invalidate everything else -------
    gate_fail = [(n, r["policy"], r["nominal"])
                 for n, a in (("velocity", av), ("force", af))
                 for r in a if r["nominal"] < GATE]
    strict_fail = [(n, r["policy"], r["nominal"])
                   for n, a in (("velocity", av), ("force", af))
                   for r in a if r["nominal"] < GATE_STRICT]
    verdicts["Q3"] = "SUPPORTED" if not gate_fail else "FAIL"
    print("  Q3  competence: every policy >= %.0f%% at c = 1 in its own interface" % GATE)
    if gate_fail:
        for n, p, v in gate_fail:
            print("      %s/%s only %.1f%%" % (n, p, v))
    print("      -> %s" % verdicts["Q3"])
    print("      stricter %.0f%% standard (added after H4 was frozen; reported"
          % GATE_STRICT)
    print("      alongside the pre-registered gate, never in place of it): %s"
          % ("all pass" if not strict_fail
             else ", ".join("%s/%s %.1f%%" % t for t in strict_fail)))
    print("")

    floored = all(r["best_any"] < FLOOR for r in af)
    overlap = not (min(Av) > max(Af) or min(Af) > max(Av))

    # ---- Q1 primary -------------------------------------------------------
    if gate_fail:
        verdicts["Q1"] = "INCONCLUSIVE -- competence gate failed"
    elif floored:
        verdicts["Q1"] = "INCONCLUSIVE -- force arm below %.0f%% at every condition" % FLOOR
    elif mAf >= mAv:
        verdicts["Q1"] = "FAIL -- KILL CONDITION MET"
    elif gap < Q1_MIN:
        verdicts["Q1"] = "FAIL"
    elif overlap:
        verdicts["Q1"] = "INCONCLUSIVE -- per-seed A ranges overlap between arms"
    else:
        verdicts["Q1"] = "SUPPORTED"
    print("  Q1  A_velocity - A_force >= %.0f points" % Q1_MIN)
    print("      %+.1f - %+.1f = %+.1f" % (mAv, mAf, gap))
    print("      -> %s\n" % verdicts["Q1"])

    # ---- Q2 the effect must exist at all ----------------------------------
    verdicts["Q2"] = "SUPPORTED" if mAv >= Q2_MIN else "FAIL"
    print("  Q2  A_velocity >= %.0f points on its own" % Q2_MIN)
    print("      %+.1f\n      -> %s\n" % (mAv, verdicts["Q2"]))

    # ---- robustness: the scheme that cannot be argued with -----------------
    Dv = [r["A_dominated"] for r in av if r["A_dominated"] is not None]
    Df = [r["A_dominated"] for r in af if r["A_dominated"] is not None]
    if Dv and Df:
        mDv, mDf = float(np.mean(Dv)), float(np.mean(Df))
        print("  ROBUSTNESS (not a pre-registered verdict): the DOMINATED scheme,")
        print("  where uniform x c is a strictly LARGER error than one link x c.")
        print("      A_velocity %+.1f  (seeds %s)" % (mDv, fmt(sorted(Dv, reverse=True))))
        print("      A_force    %+.1f  (seeds %s)" % (mDf, fmt(sorted(Df, reverse=True))))
        print("      gap        %+.1f" % (mDv - mDf))
        if mDv > 0:
            print("      A_velocity > 0 here means the LARGER error transferred better,")
            print("      which no choice of magnitude matching can produce by itself.\n")
        else:
            print("")
    else:
        mDv = mDf = None

    ok = sum(1 for v in verdicts.values() if v == "SUPPORTED")
    print("  %d of 3 supported: %s" % (ok, verdicts))
    if verdicts["Q1"].startswith("FAIL -- KILL"):
        print("\n  KILL CONDITION MET. The anisotropy is not a property of the")
        print("  interface. The geometry account remains true of this plant and")
        print("  stops generalising; the paper must say so.")

    out = {"c_star": C_STAR, "Q1_min": Q1_MIN, "Q2_min": Q2_MIN, "gate": GATE,
           "velocity": av, "force": af, "A_velocity_mean": mAv,
           "A_force_mean": mAf, "gap": gap,
           "A_dominated_velocity_mean": mDv, "A_dominated_force_mean": mDf, "seed_ranges_overlap": bool(overlap),
           "force_arm_floored": bool(floored), "verdicts": verdicts}
    json.dump(out, open(os.path.join(ROOT, "results", "h4_score.json"), "w"), indent=1)
    print("\n[out] results/h4_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
