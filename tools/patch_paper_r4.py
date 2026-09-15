"""Fit the replication table to one column, and add the R4 outcome-change observation."""
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "main.tex")
s = open(P, encoding="utf-8").read()
R = [
 (r"""\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llccc@{}}""",
  r"""\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}llccc@{}}"""),
 (r"""\bottomrule
\end{tabular}
\end{table}""",
  r"""\bottomrule
\end{tabular}}
\end{table}"""),
 (r"""\emph{Resolution.} Halving the physics step at a fixed $250$\,Hz control rate gives $\alphaFine$
$[\alphaFinelo,\alphaFinehi]$, with floor $\alphaFinefloor$.""",
  r"""\emph{Resolution.} Halving the physics step at a fixed $250$\,Hz control rate gives $\alphaFine$
$[\alphaFinelo,\alphaFinehi]$, with floor $\alphaFinefloor$. The step change alters $\changeFine$ of
individual outcomes on the same parameter pairs. Which simulators succeed depends on integration detail,
but how predictability scales with $\epsilon$ does not change regime."""),
]
bad = []
for a, b in R:
    if s.count(a) != 1:
        bad.append(a[:50])
    else:
        s = s.replace(a, b)
open(P, "w", encoding="utf-8").write(s)
print("replaced %d/%d" % (len(R) - len(bad), len(R)), "not found:", bad)
