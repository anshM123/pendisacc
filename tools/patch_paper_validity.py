"""Add the measurement procedure as an algorithm box and a threats-to-validity section."""
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "main.tex")
s = open(P, encoding="utf-8").read()
R = [
 (r"""We do not call the boundary fractal. The resolved range is about $2.5$ decades, bounded below by
simulator precision (Sec.~\ref{sec:floor}), and one registered zoom test failed.""",
  r"""We do not call the boundary fractal. The resolved range is about $2.5$ decades, bounded below by
simulator precision (Sec.~\ref{sec:floor}), and one registered zoom test failed.

\smallskip\noindent\fbox{\parbox{0.96\columnwidth}{\small
\textbf{Procedure: outcome predictability of a frozen policy}\\[2pt]
1.~Fix the initial state, success criterion, window $W$ in log-parameter space, scales $\{\epsilon_k\}$, and $N$.\\
2.~Draw $N$ base points in $W$; pair each with $\theta+\epsilon_k u$ for each $k$, and with an identical copy ($\epsilon=0$).\\
3.~Run all pairs in one vectorised batch; also re-run once in a new process.\\
4.~$f_k$ = fraction of pairs with different outcomes; $f_0$ from identical copies.\\
5.~Keep scales with $f_k-f_0>3\,\mathrm{SE}$; fit $\alpha$ = slope of $\log(f_k-f_0)$ against $\log\epsilon_k$; bootstrap pairs.\\
6.~Report $\alpha$, CI, usable scales, $f_0$, and $C_{1/2}=2^{1/\alpha}$ for the resolved range only.}}
\smallskip"""),
 (r"""\section{Limitations and What Failed}""",
  r"""\section{Threats to Validity}
\emph{Success threshold.} Flips could come from episodes sitting on the hold-fraction cut. Fewer than $3\%$ of
flips at any usable scale had both episodes within $0.05$ of the cut.
\emph{Simulator artefact.} The same pairs in an independent explicit integrator give $\alphaCPU$, and halving
the PhysX step gives $\alphaFine$. Neither exponent returns to the smooth value $1$.
\emph{Batch nondeterminism.} Identical parameters in different environment slots disagree at a rate $f_0$,
measured and subtracted for every exponent. Full re-runs are identical.
\emph{Window choice.} Windows and centres were fixed before each run. The one zoom whose registered rule landed
on an isolated island failed and is reported.
\emph{Single initial state.} Five registered initial conditions give exponents within $\alphaICmin$--$\alphaICmax$.
\emph{Sampling.} Exponents use $600$ pairs per scale, with bootstrap intervals over pairs.

\section{Limitations and What Failed}"""),
]
bad = []
for a, b in R:
    if s.count(a) != 1:
        bad.append(a[:50])
    else:
        s = s.replace(a, b)
open(P, "w", encoding="utf-8").write(s)
print("replaced %d/%d" % (len(R) - len(bad), len(R)), "not found:", bad)
