"""Score H7 against PREREGISTRATION_H7.md.

NOTE ON ORDER. Unlike the H4/H5/H6 scorers, this one was written after the
stress evaluations had run. The pre-registration fixes every quantity it uses
-- the stress set, the aggregation Q = mean over S1..S5, the thresholds
(rho >= 0.50, p < 0.01, margin >= 0.40 over in-sim success, rho <= 0.98 as a
circularity cap) and the secondary restriction to the 12 non-degenerate
policies -- so this is transcription rather than choice. It is recorded here
because the distinction matters and the reader cannot check it otherwise.

  run.cmd experiments/h7_score.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_MIN = 0.50
P_MAX = 0.01
MARGIN_MIN = 0.40
RHO_CIRCULAR = 0.98
LO, HI = 2.0, 98.0        # the non-degenerate band


def rank(v):
    v = np.asarray(v, float)
    order = v.argsort()
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b, n_perm=20000, seed=0):
    ra, rb = rank(a), rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def main() -> int:
    tags = ["S1", "S2", "S3", "S4", "S5"]
    per = {}
    for t in tags:
        p = os.path.join(ROOT, "results", "h7", "%s.json" % t)
        if not os.path.exists(p):
            print("missing", p)
            return 1
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            per.setdefault(r["policy"], {})[t] = r["stress"]
            per[r["policy"]]["own_sim"] = r["own_sim"]
            per[r["policy"]]["R_star"] = r["R_star"]

    names = sorted(per)
    Q = np.array([np.mean([per[n][t] for t in tags]) for n in names])
    own = np.array([per[n]["own_sim"] for n in names])
    R = np.array([per[n]["R_star"] for n in names])

    print("  policy                 %s    Q      own-sim   R*"
          % "".join("%6s" % t for t in tags))
    print("  " + "-" * 78)
    for i, n in enumerate(names):
        print("  %-22s%s %6.1f   %6.1f   %6.1f"
              % (n, "".join("%6.1f" % per[n][t] for t in tags), Q[i], own[i], R[i]))

    rq, pq = spearman(Q, R)
    ro, po = spearman(own, R)
    print("\n  n = %d policies" % len(names))
    print("  rho(Q,        R*) = %+.3f   permutation p = %.4f" % (rq, pq))
    print("  rho(own-sim,  R*) = %+.3f   permutation p = %.4f" % (ro, po))
    print("  margin over the incumbent selection rule: %+.3f" % (rq - ro))

    nd = (R > LO) & (R < HI)
    rq_nd = pq_nd = ro_nd = float("nan")
    if nd.sum() >= 5:
        rq_nd, pq_nd = spearman(Q[nd], R[nd])
        ro_nd, _ = spearman(own[nd], R[nd])
        print("\n  restricted to the %d non-degenerate policies (%.0f%% < R* < %.0f%%):"
              % (int(nd.sum()), LO, HI))
        print("    rho(Q, R*) = %+.3f  p = %.4f      rho(own-sim, R*) = %+.3f"
              % (rq_nd, pq_nd, ro_nd))

    print("\nPRE-REGISTERED VERDICTS (PREREGISTRATION_H7.md)\n")
    v = {}
    if rq > RHO_CIRCULAR:
        v["P3"] = "FAIL -- circular"
        v["P1"] = "VOID -- the stress set is effectively a copy of the target"
    else:
        v["P3"] = "SUPPORTED"
        v["P1"] = ("SUPPORTED" if (rq >= RHO_MIN and pq < P_MAX) else "FAIL")
    v["P2"] = "SUPPORTED" if (rq - ro) >= MARGIN_MIN else "FAIL"

    print("  P1  rho(Q, R*) >= %.2f with p < %.2f" % (RHO_MIN, P_MAX))
    print("      %+.3f, p = %.4f\n      -> %s\n" % (rq, pq, v["P1"]))
    print("  P2  Q beats in-simulator success by >= %.2f in rho" % MARGIN_MIN)
    print("      %+.3f - (%+.3f) = %+.3f\n      -> %s\n" % (rq, ro, rq - ro, v["P2"]))
    print("  P3  rho <= %.2f (not a disguised copy of the target)" % RHO_CIRCULAR)
    print("      %+.3f\n      -> %s\n" % (rq, v["P3"]))

    if not np.isnan(rq_nd) and (rq >= RHO_MIN) != (rq_nd >= RHO_MIN):
        print("  The full and restricted correlations DISAGREE about P1. The")
        print("  pre-registration says the restricted one is believed: %+.3f" % rq_nd)

    ok = sum(1 for x in v.values() if x == "SUPPORTED")
    print("  %d of 3 supported: %s" % (ok, v))
    if v["P1"] == "FAIL":
        print("\n  Nothing cheap measured in simulation predicts transfer on this")
        print("  system. After H3, H4, H5 and H6 that is the honest headline.")

    json.dump({"policies": names, "Q": Q.tolist(), "own_sim": own.tolist(),
               "R_star": R.tolist(), "per_condition": per,
               "rho_Q": rq, "p_Q": pq, "rho_own": ro, "p_own": po,
               "margin": rq - ro, "n": len(names),
               "n_nondegenerate": int(nd.sum()),
               "rho_Q_nondegenerate": rq_nd, "p_Q_nondegenerate": pq_nd,
               "rho_own_nondegenerate": ro_nd,
               "thresholds": {"rho_min": RHO_MIN, "p_max": P_MAX,
                              "margin_min": MARGIN_MIN, "rho_circular": RHO_CIRCULAR},
               "verdicts": v},
              open(os.path.join(ROOT, "results", "h7_score.json"), "w"), indent=1)
    print("\n[out] results/h7_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
