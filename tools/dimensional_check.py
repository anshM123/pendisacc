"""Is the closed-loop symmetry group anything more than dimensional analysis?

results/symmetry_group.json holds the scaling group solved from the closed-loop
equations of motion. This checks, against a table of physical dimensions,
whether every generator of that group is accounted for by:

  * a change of the MASS unit   (every parameter scaled by its M exponent),
  * a change of the LENGTH unit (every parameter scaled by its L exponent),
  * a change of the TIME unit   (only admissible when time is not frozen),
  * a parameter that never enters the equations at all (a trivial generator).

If the group's span equals the span of those, the group contains nothing that
Buckingham-Pi does not already give, and the solver is a convenience rather
than a discovery. That is the claim the paper makes, so it is checked rather
than asserted.

  run.cmd tools/dimensional_check.py
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "symmetry_group.json")
OUT = os.path.join(ROOT, "results", "dimensional_check.json")

# (M, L, T) exponents
DIM = {
    "g": (0, 1, -2),
    "m_cart": (1, 0, 0), "kv": (1, 0, -1), "v_cmd": (0, 1, -1),
    "b_joint": (1, 2, -1),     # N m s / rad
    "fc_joint": (1, 2, -2),    # N m
    "b_cart": (1, 0, -1),      # N s / m
    "x": (0, 1, 0), "t": (0, 0, 1),
}
for i in range(1, 5):
    DIM["m%d" % i] = (1, 0, 0)
    DIM["I%d" % i] = (1, 2, 0)
    DIM["L%d" % i] = (0, 1, 0)
    DIM["lc%d" % i] = (0, 1, 0)


def rank(rows):
    return int(np.linalg.matrix_rank(np.array(rows, dtype=float))) if rows else 0


def in_span(basis, v):
    return rank(basis + [v]) == rank(basis)


def check(case_name, case):
    free = case["free_exponents"]
    basis = [[float(Fraction(b.get(n, "0"))) for n in free] for b in case["basis"]]
    unknown = [n for n in free if n not in DIM]
    if unknown:
        raise SystemExit("no dimensions recorded for: %s" % unknown)

    time_free = "t" in free
    unit = {"mass": [DIM[n][0] for n in free],
            "length": [DIM[n][1] for n in free]}
    if time_free:
        unit["time"] = [DIM[n][2] for n in free]

    covered = {k: in_span(basis, v) for k, v in unit.items()}
    trivial = [free[int(np.flatnonzero(np.array(b))[0])]
               for b in basis if np.count_nonzero(np.array(b)) == 1]
    explained = [list(v) for v in unit.values()]
    for name in trivial:
        explained.append([1.0 if n == name else 0.0 for n in free])
    nothing_else = (rank(basis) == rank(explained)
                    and rank(basis + explained) == rank(basis))

    print(case_name)
    print("  group dimension                         %d" % rank(basis))
    for k, ok in covered.items():
        print("  %-6s-unit scaling lies in the group   %s" % (k, ok))
    print("  parameters that never enter the EOM     %s" % (trivial or "none"))
    print("  group == unit scalings + unused params  %s" % nothing_else)
    print("")
    return {"group_dim": rank(basis), "time_free": time_free,
            "unit_scalings_in_group": covered, "unused_parameters": trivial,
            "nothing_beyond_dimensional_analysis": bool(nothing_else)}


def main() -> int:
    data = json.load(open(SRC, encoding="utf-8"))
    out = {}
    for name in ("velocity_loop_dissipative_fixed_rate",
                 "velocity_loop_dissipative_free_time",
                 "velocity_loop_fixed_rate",
                 "prescribed_base_fixed_rate"):
        if name in data:
            out[name] = check(name, data[name])
    verdict = all(v["nothing_beyond_dimensional_analysis"] for v in out.values())
    print("every solved group is dimensional analysis plus unused parameters: %s"
          % verdict)
    out["verdict_nothing_beyond_dimensional_analysis"] = bool(verdict)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
