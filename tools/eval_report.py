"""Collate the per-seed evaluations into one table.

The number that matters is the success rate of the BEST checkpoint per seed --
"got up from dead hang and was still up at the end" -- and, across seeds,
whether that number is reproducible. A single good seed is not a result; that
was the whole failure mode being fixed.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows = []
for s in (1, 2, 3):
    path = os.path.join(ROOT, "results", "eval_rel%d.json" % s)
    if not os.path.exists(path):
        rows.append((s, None))
        continue
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    best = max(data["results"], key=lambda r: (r["success_rate"], -r["early_termination_rate"]))
    rows.append((s, (data, best)))

print("seed  checkpoint        success   early-term   ever-up   mean tip")
print("-" * 68)
rates = []
for s, item in rows:
    if item is None:
        print("%4d  %s" % (s, "(no evaluation)"))
        continue
    data, b = item
    rates.append(b["success_rate"])
    print("%4d  %-16s %6.1f%%      %5.1f%%    %5.1f%%    %+.3f"
          % (s, b["checkpoint"], 100 * b["success_rate"],
             100 * b["early_termination_rate"], 100 * b["ever_reached_upright"],
             b["mean_tip_height"]))

if rates:
    n = data["num_envs"]
    print("-" * 68)
    print("across %d seeds: min %.1f%%  mean %.1f%%  max %.1f%%   (%d episodes each)"
          % (len(rates), 100 * min(rates), 100 * sum(rates) / len(rates), 100 * max(rates), n))
    print("\nReliability claim rests on the MINIMUM, not the mean or the best seed.")
