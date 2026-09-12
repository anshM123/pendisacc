"""Round 3: turn the damping candidate into a METHOD and test it in Isaac. PRE-REGISTRATION.

Registered before A20 (the fresh-Isaac replication of A14) has a result.

Method (zero Isaac cost): among a run's checkpoints that already pass the usual
own-simulator gate (Isaac nominal success >= 95%), deploy the one whose CPU
linearisation has the LARGEST minimum closed-loop damping ratio, instead of the
one with the best own-simulator success.

Test: both picks, for every corrected-asset run that has per-checkpoint Isaac
evaluations, scored on the frozen confirmation suite (xi/confirm_theory, 16
conditions). Seed/run is the unit.

  S01  damping pick beats own-sim pick on mean suite success by >= 10 points in >= 5 of 7 runs
  S02  mean paired improvement >= 10 points
  S03  (control) the two rules pick different checkpoints in >= 4 of 7 runs; otherwise
       the test has no power and S01/S02 are reported as uninformative, not failed

  run.cmd theories/round3_select.py --stage select   (CPU: write the two pick lists)
  run.cmd theories/round3_select.py --stage score    (after tools/confirm_select_queue.sh)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import screen as S  # noqa: E402

CONF = os.path.join(S.ROOT, "xi", "confirm_theory")
RES = os.path.join(S.ROOT, "results", "confirm_select")
SOURCES = [("results/corrected/CORR_s1.json", "*_CORR_s1"), ("results/corrected/CORR_s2.json", "*_CORR_s2"),
           ("results/corrected/CORR_s3.json", "*_CORR_s3"),
           ("results/T5_VOID_fallthrough/T5_orbit_s1.json", "*_VOIDT5_orbit_s1"),
           ("results/T5_VOID_fallthrough/T5_orbit_s2.json", "*_VOIDT5_orbit_s2"),
           ("results/T5_VOID_fallthrough/T5_transverse_s1.json", "*_VOIDT5_transverse_s1"),
           ("results/T5_VOID_fallthrough/T5_transverse_s2.json", "*_VOIDT5_transverse_s2")]


def picks():
    out = []
    for src, pat in SOURCES:
        res = json.load(open(os.path.join(S.ROOT, src), encoding="utf-8"))["results"]
        run = None
        for d in sorted(glob.glob(os.path.join(S.LOGS, pat))):
            if all(os.path.exists(os.path.join(d, r["checkpoint"])) for r in res):
                run = d
        if run is None:
            continue
        gate = [r for r in res if r["success_rate"] >= 0.95]
        if not gate:
            continue
        base = sorted(gate, key=lambda r: (-r["success_rate"], r["early_termination_rate"]))[0]
        damp = {r["checkpoint"]: S.min_damping(S.K_of(os.path.join(run, r["checkpoint"])), {}, "corrected")
                for r in gate}
        meth = max(gate, key=lambda r: damp[r["checkpoint"]])
        out.append({"run": run, "baseline": os.path.join(run, base["checkpoint"]),
                    "damping": os.path.join(run, meth["checkpoint"]), "damping_values": damp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("select", "score"), required=True)
    a = ap.parse_args()
    sel_path = os.path.join(CONF, "select_picks.json")
    if a.stage == "select":
        if os.path.exists(sel_path):
            print("REFUSING: picks already frozen")
            return 1
        P = picks()
        json.dump(P, open(sel_path, "w", encoding="utf-8"), indent=1)
        cks = sorted({p["baseline"] for p in P} | {p["damping"] for p in P})
        open(os.path.join(CONF, "select_list.txt"), "w", encoding="utf-8").write("\n".join(cks) + "\n")
        diff = sum(p["baseline"] != p["damping"] for p in P)
        print("%d runs; rules differ in %d; %d checkpoints to score" % (len(P), diff, len(cks)))
        for p in P:
            print("  %s  own-sim pick %s | damping pick %s" % (os.path.basename(p["run"]),
                  os.path.basename(p["baseline"]), os.path.basename(p["damping"])))
        return 0
    P = json.load(open(sel_path, encoding="utf-8"))
    man = json.load(open(os.path.join(CONF, "manifest.json"), encoding="utf-8"))
    conds = [c["name"] for c in man["conditions"]]
    tab = {}
    for c in conds:
        for r in json.load(open(os.path.join(RES, c + ".json"), encoding="utf-8"))["results"]:
            tab[(c, r["run"], r["checkpoint"])] = r["success_rate"]

    def mean_of(ck):
        return 100 * float(np.mean([tab[(c, os.path.basename(os.path.dirname(ck)), os.path.basename(ck))] for c in conds]))
    rows = [{"run": os.path.basename(p["run"]), "baseline": mean_of(p["baseline"]), "damping": mean_of(p["damping"]),
             "differ": p["baseline"] != p["damping"]} for p in P]
    imp = [r["damping"] - r["baseline"] for r in rows]
    out = {"rows": rows,
           "S01_runs_improved_ge10": int(sum(i >= 10 for i in imp)), "S01_pass": sum(i >= 10 for i in imp) >= 5,
           "S02_mean_improvement": float(np.mean(imp)), "S02_pass": float(np.mean(imp)) >= 10,
           "S03_differ": int(sum(r["differ"] for r in rows)), "S03_pass": sum(r["differ"] for r in rows) >= 4}
    os.makedirs(S.OUT, exist_ok=True)
    json.dump(out, open(os.path.join(S.OUT, "round3_select.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
