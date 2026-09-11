"""Apply the expansion edits to paper/main.tex.

Kept as a script rather than done by hand so the edits are reviewable and
re-runnable, and so the inserted numbers can be checked against results/*.json
by tools/check_paper_numbers.py afterwards.

  run.cmd paper/sections_expand.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "main.tex")

EDITS = []


def edit(old, new, why):
    EDITS.append((old, new, why))


# ---------------------------------------------------------------- figure 1
edit(
r"""\section{Introduction}""",
r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/swingup_cad_frames.png}
\caption{The task. Eight frames of one PPO swing-up from dead hang, with the
tip trace. The plant is a triple inverted pendulum on a belt-driven cart;
every episode in this paper starts hanging and must reach upright \emph{and
stay there} for the final quarter of the episode. Link geometry is taken from
CAD of a physical build.}
\label{fig:task}
\end{figure*}

\section{Introduction}""",
"add the task figure")

# ------------------------------------------------------- related work depth
edit(
r"""\textbf{Behavioural equivalence} in RL, via bisimulation metrics
\cite{ferns2004,castro2020}, formalises when two states or MDPs are
interchangeable for control. That work quotients the \emph{state} space; the
object here is an equivalence on \emph{physical parameters} induced by a task
and an actuation interface.""",
r"""\textbf{Behavioural equivalence} in RL, via bisimulation metrics
\cite{ferns2004,castro2020}, formalises when two states or MDPs are
interchangeable for control. That work quotients the \emph{state} space; the
object here is an equivalence on \emph{physical parameters} induced by a task
and an actuation interface.

\textbf{Dimensional analysis and dynamic similarity} have a long history in
legged locomotion, where Froude-number scaling relates gaits across animals of
different size. The symmetry in \S\ref{sec:sym} is of that character---a
statement that a family of physically distinct systems produces identical
motion---but it is derived for a driven, underactuated chain and, crucially, is
contingent on how the base is actuated rather than on the mechanics alone.

\textbf{What is new here} is not that model errors differ in importance, which
is widely understood, but the strength and structure of the effect: at
\emph{identical} parameter distance, one direction is free and another is
fatal; the privileged direction has an exact closed-form characterisation; and
the natural mechanistic explanation for it---the actuation interface---turns
out not to survive a pre-registered test.""",
"deepen related work and state the delta")

# ----------------------------------------------------- protocol detail block
edit(
r"""\subsection{Learning and evaluation}
PPO (\texttt{rsl-rl}) in Isaac Lab, $4096$ environments, $1000$ iterations,
frozen hyperparameters. Checkpoints are selected by measured success in the
policy's \emph{own} training simulator---the only criterion available without
access to reality. Evaluation uses $256$ independent episodes.""",
r"""\subsection{Learning and evaluation}
PPO (\texttt{rsl-rl}) in Isaac Lab. Actor and critic are $[256,256,128]$ MLPs
with ELU activations; $\gamma = 0.999$ (a $\sim\!4\,$s horizon at $250\,$Hz,
long enough that the policy values an upright state it cannot reach for two
seconds), $\lambda = 0.95$, clip $0.2$, learning rate $3\!\times\!10^{-4}$,
entropy coefficient $0.006$, $4$ minibatches, $96$ steps per environment per
iteration, $4096$ environments, $1000$ iterations. Hyperparameters are frozen
across every experiment in this paper.

Observations are cart state, link angles as $(\sin,\cos)$ pairs, link rates,
and the previous action. Checkpoints are selected by measured success in the
policy's \emph{own} training simulator---the only criterion available without
access to reality---and evaluation uses $256$ independent episodes.

\textbf{The experimental unit is the policy seed, not the episode.} This is not
a formality: \S\ref{sec:blind} measures a 77-point transfer spread across seeds
at identical settings, which exceeds most effects one might wish to claim.

\subsection{Baseline competence}
The frozen baseline reaches $3072/3072$ nominal swing-ups from dead hang. Under
randomised initial conditions and sensor noise, three seeds pool to $99.51\%$
($3057/3072$); because per-seed success probabilities visibly differ
($99.80$, $99.71$, $99.02$), the pooled Wilson interval understates uncertainty
across seeds and we quote the \textbf{worst seed, $99.02\%$}, as the honest
figure. The results below are therefore about a controller that works, not one
that is marginal.""",
"expand protocol, add PPO detail and baseline competence")

# remove the now-duplicated experimental-unit paragraph
edit(
r"""\textbf{The experimental unit is the policy seed, not the episode.} This is not
a formality: \S\ref{sec:blind} measures a 77-point transfer spread across seeds
at identical settings, which exceeds most effects one might wish to claim.

\section{The Gap is Anisotropic}""",
r"""\section{The Gap is Anisotropic}""",
"drop the duplicated paragraph")

# ------------------------------------------------------- dominated table
edit(
r"""The anisotropy, defined as
$A = P(\text{uniform}) - \overline{P(\text{transverse})}$ at matched
$\|\Delta m\|$, is $+100.0$ points at $c = 2.5$ with a per-seed spread of
$2.2$ points. Under the dominated scheme it is $+99.7$: the \emph{larger} error
transfers better, which no choice of magnitude matching can produce on its own.""",
r"""The anisotropy, defined as
$A = P(\text{uniform}) - \overline{P(\text{transverse})}$ at matched
$\|\Delta m\|$, is $+100.0$ points at $c = 2.5$ with a per-seed spread of
$2.2$ points.

Table~\ref{tab:dominated} gives the dominated scheme, which removes the
matching question entirely.

\begin{table}[t]
\caption{Dominated scheme, cart-velocity interface. Uniform $\times c$ against
a \emph{single} link $\times c$: the uniform perturbation is strictly larger in
every norm, yet it is the harmless one.}
\label{tab:dominated}
\centering
\begin{tabular}{rrrrr}
\toprule
& \multicolumn{2}{c}{uniform $\times c$} & \multicolumn{2}{c}{one link $\times c$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$c$ & $\|\Delta m\|$ & success & $\|\Delta m\|$ & success\\
\midrule
1.5 & 0.109 & $100.0\%$ & 0.089 & $0.0\%$\\
2.5 & 0.327 & $100.0\%$ & 0.266 & $0.0\%$\\
4.0 & 0.654 & $98.8\%$  & 0.533 & $0.0\%$\\
8.0 & 1.527 & $100.0\%$ & 1.243 & $0.0\%$\\
\bottomrule
\end{tabular}
\end{table}

At every row the uniform condition displaces the model \emph{further} and
transfers perfectly while the single-link condition displaces it less and fails
completely. No choice of norm or matching convention produces that.""",
"add the dominated table")

# ------------------------------------------------- authority table in sec:sym
edit(
r"""We confirm (iii) causally elsewhere: with frozen
policy weights, uniform $\times 20$ gives $0.0\%$ at $k_v\!=\!400$/$349\,$N and
$100.0\%$ at $k_v\!=\!4000$/$7000\,$N, and neither raising the gain alone nor the
clamp alone suffices.""",
r"""Hypothesis (iii) we confirm causally. Table~\ref{tab:authority} changes only
the drive, with frozen policy weights.

\begin{table}[t]
\caption{Authority gates the symmetry. Frozen policy weights; only the drive
changes. Raising the gain alone, or the clamp alone, leaves $0.0\%$: the two
limits bound each other through
$F = \min(F_{\mathrm{clamp}}, k_v(v_{\max} - \dot x))$.}
\label{tab:authority}
\centering
\begin{tabular}{lrr}
\toprule
model error & $k_v\,400$, $349\,$N & $k_v\,4000$, $7000\,$N\\
\midrule
uniform $\times 20$        & $0.0\%$   & $\mathbf{100.0\%}$\\
uniform $\times 30$        & $0.0\%$   & $\mathbf{100.0\%}$\\
asymmetric $[5,1,1]$       & $0.0\%$   & $0.0\%$\\
uniform $\times 5$ (control) & $100.0\%$ & $100.0\%$\\
\bottomrule
\end{tabular}
\end{table}

Authority restores an \emph{intact} symmetry and cannot manufacture a broken
one: the asymmetric error stays at $0.0\%$ even under $20\times$ force and
$10\times$ gain. This is the sense in which the equivalence is realisable only
while the drive can still prescribe the base.""",
"promote authority result to a table")

# ------------------------------------------------------- fidelity tables
edit(
r"""Across 38 heterogeneous conditions spanning actuation, dissipation, rigid-body
and combined errors, conventional short-window trajectory RMSE is the best
aggregate predictor of transfer ($\rho = -0.506$). Two pre-registered
policy-conditioned alternatives were \textbf{rejected}: a stability-weighted gap
$D_{SW}$ ($\rho = -0.415$) and a task-margin projected risk $R_{TC}$
($\rho = -0.406$), the latter failing all four of its frozen kill criteria.""",
r"""Across 38 heterogeneous conditions spanning actuation, dissipation, rigid-body
and combined errors, conventional short-window trajectory RMSE is the best
aggregate predictor of transfer. Two pre-registered policy-conditioned
alternatives were \textbf{rejected}.

\begin{table}[t]
\caption{Rank correlation with measured transfer over 38 heterogeneous
conditions. The two policy-conditioned metrics we pre-registered
($D_{SW}$, $R_{TC}$) lose to conventional trajectory fidelity; $R_{TC}$ failed
all four of its frozen kill criteria.}
\label{tab:fidelity}
\centering
\begin{tabular}{lr}
\toprule
predictor & $\rho$ vs.\ measured success\\
\midrule
short-window trajectory RMSE & $\mathbf{-0.506}$\\
$|$phase error at $\lambda_{\max}|$ & $-0.505$\\
step-response RMSE & $-0.451$\\
$D_{SW}$ (stability-weighted gap) & $-0.415$\\
$R_{TC}$ (task-margin projected risk) & $-0.406$\\
raw model gap & $-0.390$\\
actuator rise-time error & $-0.350$\\
$G_T$ (trajectory amplification) & $-0.191$\\
\bottomrule
\end{tabular}
\end{table}""",
"fidelity correlation table")

edit(
r"""But the winner cannot \emph{certify}. With a tolerance frozen in advance
($10\%$ relative, $\geq 50$-point gap, plausible conditions only), a
second-order actuator condition and a joint-damping condition differ by $1.51\%$
in trajectory RMSE and transfer at $0.0\%$ and $100.0\%$. Twenty-eight
admissible pairs fall within tolerance. Trajectory fidelity is therefore
\emph{informative statistically and ambiguous locally}---a sharper claim than
``insufficient'', and one our own data supports rather than contradicts.""",
r"""But the winner cannot \emph{certify}. With a tolerance frozen in advance
($10\%$ relative difference, a $\geq 50$-point transfer gap, and both members
flagged plausible for the real build), 28 admissible pairs fall within
tolerance and four clear the gap threshold; two of those span different model
families.

\begin{table}[t]
\caption{Fidelity twins: simulator pairs that a trajectory-RMSE score cannot
separate but whose transfer is opposite. Tolerance and gap threshold were
frozen before the search ran.}
\label{tab:twins}
\centering
\begin{tabular}{llrrr}
\toprule
condition A & condition B & $\Delta E$ & $P_A$ & $P_B$\\
\midrule
2nd-order act., $\zeta{=}0.5$ & joint damping $0.004$ & $1.5\%$ & $0.0\%$ & $100.0\%$\\
joint friction $0.012$ & joint damping $0.004$ & $8.3\%$ & $5.9\%$ & $100.0\%$\\
2nd-order act., $\omega_n{=}28$ & joint friction $0.004$ & $3.0\%$ & $46.9\%$ & $100.0\%$\\
$\tau = 50\,$ms & 2nd-order, $\omega_n{=}20$ & $9.8\%$ & $57.4\%$ & $1.2\%$\\
\bottomrule
\end{tabular}
\end{table}

The leading pair differs by $1.51\%$ in trajectory RMSE---below what the
measure resolves across initial conditions within a single condition---and
transfers at $0.0\%$ and $100.0\%$. Trajectory fidelity is therefore
\emph{informative statistically and ambiguous locally}: a sharper claim than
``insufficient'', and one our own data supports rather than contradicts.""",
"twins table")

# ----------------------------------------------- selection figure + checkpoint
edit(
r"""Nor is the criterion stable within a run.""",
r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/selection_blindness.png}
\caption{Left: every policy, own-simulator success against success in
$R^\star$. The horizontal axis spans two points and the vertical axis spans a
hundred. Right: the controlled version---identical simulator, protocol, reward,
architecture and iteration count, differing only in random seed.}
\label{fig:blind}
\end{figure}

Nor is the criterion stable within a run.""",
"selection-blindness figure")

# ------------------------------------------------------------- H3 table
edit(
r"""\textbf{It failed, in the opposite direction}: $1.0\%$ versus $4.9\%$. A""",
r"""\begin{table}[t]
\caption{Training \emph{in} each simulator, then deploying into a common
$R^\star$ fixed in advance. Per-seed values show the spread that motivates
treating the seed as the experimental unit.}
\label{tab:h3}
\centering
\begin{tabular}{lrlr}
\toprule
training simulator & seeds & success in $R^\star$ & mean\\
\midrule
nominal              & 3 & $100.0,\,71.1,\,23.0$ & $64.7\%$\\
joint damping $0.004$& 3 & $100.0,\,73.8,\,22.7$ & $65.5\%$\\
transverse $[1.5,1,1]$& 2 & $7.4,\,2.3$          & $4.9\%$\\
uniform $\times 8$   & 2 & $2.0,\,0.0$           & $1.0\%$\\
2nd-order act.\ $\zeta{=}0.5$ & 3 & $2.0,\,0.8,\,0.0$ & $0.9\%$\\
$12\,$ms dead time   & 2 & $1.6,\,0.0$           & $0.8\%$\\
\bottomrule
\end{tabular}
\end{table}

\textbf{It failed, in the opposite direction}: $1.0\%$ versus $4.9\%$. A""",
"H3 table")

# -------------------------------------------------- threats to validity
edit(
r"""\section{Limitations}""",
r"""\section{Measurement Defects Found En Route}

Four defects were found by measurement rather than by inspection, and each
would have produced a plausible but wrong number. We record them because they
bear on how much weight any single simulation result should carry.

\textbf{A dynamically assigned event term silently did nothing.} Mass
perturbations assigned to the environment configuration after instantiation
never registered; the event manager reported only \texttt{[reset]} and the
perturbation was applied to nothing while the run reported a plausible success
rate. Every mass condition in this paper is now applied through the physics
view after construction and the realised masses are written into the output
file.

\textbf{Reward was a misleading success metric.} One run climbed to reward
$58$ while never lifting the tip above $-0.334$, farming near-upright starts.
Success is therefore defined physically and measured, never inferred from
return.

\textbf{A closed-loop quantity was estimated on an open-loop distribution.}
Cart force demand came out at $22.7\,$N when sampled from an open-loop state
distribution and $98.7\,$N along the trajectory the policy actually flies---a
factor of four, and in the direction that would have made a drive look
adequate when it was not.

\textbf{An action clipping wrapper hid the action from its own penalty.}
Clipping before the environment step meant the effort penalty read the clipped
value, so the policy's exploration noise grew without cost until the commanded
standard deviation reached $47.2$.

\section{Limitations}""",
"add the measurement-defects section")


def main() -> int:
    s = open(TEX, encoding="utf-8").read()
    applied, missed = 0, []
    for old, new, why in EDITS:
        if old not in s:
            missed.append(why)
            continue
        s = s.replace(old, new, 1)
        applied += 1
    open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("applied %d of %d edits" % (applied, len(EDITS)))
    for m in missed:
        print("  MISSED:", m)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
