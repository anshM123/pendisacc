"""Rewrite the state-space claims in paper/main.tex to match R3's registered outcome (uninformative:
the initial-state grid stayed > 98% success after both widenings)."""
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "main.tex")
s = open(P, encoding="utf-8").read()
R = [
 (r"""For a frozen reinforcement-learning swing-up controller on a triple inverted
pendulum we sample pairs of simulators, or of initial states, that differ by a
perturbation of size $\epsilon$.""",
  r"""For a frozen reinforcement-learning swing-up controller on a triple inverted
pendulum we sample pairs of simulators that differ by a perturbation of size
$\epsilon$."""),
 (r"""conditions (median $\alphaICmed$), an independent non-PhysX dynamics implementation
($\alphaCPU$), a halved physics step ($\alphaFine$), and perturbations of the initial
state rather than the model ($\alphaState$).""",
  r"""conditions (median $\alphaICmed$), an independent non-PhysX dynamics implementation
($\alphaCPU$), and a halved physics step ($\alphaFine$). By contrast, the same policy
is almost insensitive to its initial state: $\stateSucc$ success over $\pm0.8$\,rad of
initial link offsets. The unpredictability lives in the model, not the start."""),
 (r"""learned policy \emph{succeeds}, in the space robotics actually calibrates: simulator
parameters, as well as initial states.""",
  r"""learned policy \emph{succeeds}, in the space robotics actually calibrates: simulator
parameters."""),
 (r"""\item \textbf{Pre-registered replication} across policies, physical and actuator uncertainty,
initial conditions, an independent dynamics implementation, numerical resolution, and state space.
Registered failures are reported.""",
  r"""\item \textbf{Pre-registered replication} across policies, physical and actuator uncertainty,
initial conditions, an independent dynamics implementation and numerical resolution. A contrasting
null in initial-state space and the registered failures are reported."""),
 (r"""\emph{State space.} Holding the physics nominal and perturbing the initial angles of link~1 and joint~2
(window $\pm\stateHalf$\,rad) gives $\alpha=\alphaState$ $[\alphaStatelo,\alphaStatehi]$. The same slow
payoff therefore applies to state estimation, not only to model calibration.""",
  r"""\emph{Initial state (contrast).} Holding the physics nominal, we perturbed the initial angles of link~1
and joint~2. The registered rule widened the window from $\pm0.05$ to $\pm0.2$ and then $\pm0.8$\,rad while
success stayed above $98\%$. At $\pm0.8$\,rad it was still $\stateSucc$, so the test is uninformative on
$\alpha$. There is almost no boundary to measure. The swing-up funnels a wide range of starting states
into success, while tiny changes to the plant decide it."""),
 (r"""Initial state         & $\delta\theta_1\times\delta\theta_2$ & $\alphaState$ [$\alphaStatelo$, $\alphaStatehi$] & $\alphaStaten$ & $\alphaStateC$\\""",
  r"""Initial state         & $\pm0.8$\,rad, nominal plant & \multicolumn{3}{c}{uninformative: $\stateSucc$ success}\\"""),
 (r"""(c)~Initial-state plane with nominal physics; zoom on the main boundary; the failed zoom on an
isolated island, which is flat.}""",
  r"""(c)~Zoom on the main boundary, and the failed zoom on an isolated island, which is flat.}"""),
 (r"""\emph{For state estimation.} The initial-state plane shows the same payoff. Better estimates of the
starting configuration buy little certainty about success for states near the boundary.""",
  r"""\emph{Model versus state.} For this task the ambiguity is concentrated in the plant, not the start. The
policy tolerates large initial-state errors but not $0.1\%$ model errors near the boundary. Effort spent on
state estimation at the start would be wasted, and effort on model accuracy pays off slowly."""),
 (r"""the outcome-predictability exponent of a \emph{fixed learned controller} over \emph{simulator parameters}
and initial states, show that it replicates, and translate it into what calibration can and cannot
buy.""",
  r"""the outcome-predictability exponent of a \emph{fixed learned controller} over \emph{simulator parameters},
show that it replicates, and translate it into what calibration can and cannot buy."""),
]
bad = []
for a, b in R:
    if s.count(a) != 1:
        bad.append(a[:60])
    else:
        s = s.replace(a, b)
open(P, "w", encoding="utf-8").write(s)
print("replaced %d/%d" % (len(R) - len(bad), len(R)), "not found:", bad)
