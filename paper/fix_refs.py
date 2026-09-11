"""Reference every float from the body. IEEE expects it and the lint flags it."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "main.tex")

EDITS = [
    (r"""On the system studied here that ordering is wrong, and not marginally. Scaling""",
     r"""The plant (Fig.~\ref{fig:task}) is a triple inverted pendulum on a cart
performing swing-up from dead hang. On this system that ordering is wrong, and
not marginally. Scaling""",
     "reference the task figure"),

    (r"""aggregate predictor of transfer. Two pre-registered policy-conditioned
alternatives were \textbf{rejected}.""",
     r"""aggregate predictor of transfer (Table~\ref{tab:fidelity}). Two
pre-registered policy-conditioned alternatives were \textbf{rejected}.""",
     "reference the fidelity table"),

    (r"""tolerance and four clear the gap threshold; two of those span different model
families.""",
     r"""tolerance and four clear the gap threshold (Table~\ref{tab:twins}); two of
those span different model families.""",
     "reference the twins table"),

    (r"""Spearman $\rho = -0.090$ (permutation $p = 0.749$, $n = 15$). The selection
criterion every practitioner has available is uninformative about the quantity
every practitioner wants.""",
     r"""Spearman $\rho = -0.090$ (permutation $p = 0.749$, $n = 15$), shown in
Fig.~\ref{fig:blind}. The selection criterion every practitioner has available
is uninformative about the quantity every practitioner wants.""",
     "reference the selection figure"),

    (r"""training in an equivalence-direction simulator (uniform $\times 8$) would
transfer at least 20 points better than training in a transverse one
(mass $[1.5,1,1]$).""",
     r"""training in an equivalence-direction simulator (uniform $\times 8$) would
transfer at least 20 points better than training in a transverse one
(mass $[1.5,1,1]$). Table~\ref{tab:h3} gives the outcome.""",
     "reference the H3 table"),
]


def main() -> int:
    s = open(TEX, encoding="utf-8").read()
    missed = []
    for old, new, why in EDITS:
        if old not in s:
            missed.append(why)
            continue
        s = s.replace(old, new, 1)
    open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("applied %d of %d" % (len(EDITS) - len(missed), len(EDITS)))
    for m in missed:
        print("  MISSED:", m)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
