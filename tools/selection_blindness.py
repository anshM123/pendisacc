"""In-simulator performance carries no information about transfer.

This is not a new experiment. It is the H3 data (`results/claim5_select/`,
`results/claim5/`) read along an axis the pre-registration did not ask about,
and it is reported as exploratory for exactly that reason -- no threshold was
frozen for it in advance.

Every one of the 15 H3 policies was selected the way a practitioner without
access to reality must select: best measured dead-hang success in its OWN
training simulator. That selection criterion is near-saturated for all of them.
Their success in the common deployment target R* is not.

The comparison that matters is the nominal arm, where the simulator, the
protocol, the reward, the architecture and the iteration count are identical
and the random seed is the only thing that differs.

  run.cmd tools/selection_blindness.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "selection_blindness.json")


def rows():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "claim5_select", "*.json"))):
        tag = os.path.basename(f)[:-5]
        dep_path = os.path.join(ROOT, "results", "claim5", "%s.json" % tag)
        if not os.path.exists(dep_path):
            continue
        sel = json.load(open(f, encoding="utf-8"))
        best = max(sel["results"],
                   key=lambda r: (r["success_rate"], -r["early_termination_rate"]))
        dep = json.load(open(dep_path, encoding="utf-8"))
        out.append({"policy": tag.replace("C5_", ""),
                    "checkpoint": best["checkpoint"],
                    "own_sim": 100.0 * best["success_rate"],
                    "in_R_star": 100.0 * dep["results"][0]["success_rate"]})
    return out


def spearman(a, b):
    """Rank correlation without a scipy dependency, plus a permutation p."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        for x in np.unique(v):
            m = v == x
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(20000)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return rho, p


def main() -> int:
    rs = rows()
    if not rs:
        print("no H3 results found")
        return 1

    print("Every policy was selected on its OWN training simulator -- the only")
    print("criterion available without access to reality.\n")
    print("  policy                own-sim      in R*")
    print("  " + "-" * 44)
    for r in rs:
        print("  %-20s %6.1f%%    %6.1f%%" % (r["policy"], r["own_sim"], r["in_R_star"]))

    o = np.array([r["own_sim"] for r in rs])
    d = np.array([r["in_R_star"] for r in rs])
    rho, p = spearman(o, d)
    print("\n  own-sim   mean %5.1f%%   range %5.1f - %5.1f   spread %5.1f points"
          % (o.mean(), o.min(), o.max(), o.ptp()))
    print("  in R*     mean %5.1f%%   range %5.1f - %5.1f   spread %5.1f points"
          % (d.mean(), d.min(), d.max(), d.ptp()))
    print("\n  Spearman rho(own-sim, R*) = %+.3f   permutation p = %.3f   n = %d"
          % (rho, p, len(rs)))
    print("  The selection criterion every practitioner has available is")
    print("  uninformative about the quantity every practitioner wants.")

    nom = [r for r in rs if r["policy"].startswith("S_nominal")]
    no = np.array([r["own_sim"] for r in nom])
    nd = np.array([r["in_R_star"] for r in nom])
    print("\n  THE CONTROLLED VERSION. Same simulator, same protocol, same reward,")
    print("  same architecture, same iteration count. The seed is the only")
    print("  difference between these three policies:\n")
    print("    policy             own-sim      in R*")
    print("    " + "-" * 40)
    for r in nom:
        print("    %-18s %6.1f%%    %6.1f%%" % (r["policy"], r["own_sim"], r["in_R_star"]))
    print("\n    own-sim spread %4.1f points        in R* spread %5.1f points"
          % (no.ptp(), nd.ptp()))
    print("\n  Seed variance within one simulator moves transfer by %.0f points while"
          % nd.ptp())
    print("  moving the selection criterion by %.1f. Simulator choice is not the" % no.ptp())
    print("  only thing that is not being measured -- neither is the policy.")

    json.dump({"rows": rs, "spearman_rho": rho, "permutation_p": p, "n": len(rs),
               "own_sim_spread": float(o.ptp()), "R_star_spread": float(d.ptp()),
               "nominal_arm": {"own_sim_spread": float(no.ptp()),
                               "R_star_spread": float(nd.ptp()),
                               "rows": nom},
               "exploratory": True,
               "note": "Not pre-registered. Reported as exploratory: this axis was "
                       "not named in PREREGISTRATION_H3.md and no threshold was "
                       "frozen for it in advance."},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
