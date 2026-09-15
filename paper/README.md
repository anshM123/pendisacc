# Paper build

`main.tex`: *When Better Models Barely Help: Predictability Limits in Learned
Nonlinear Robot Control* (IEEE conference format, ICRA).

The previous draft, on similarity coordinates, is preserved as
`main_similarity_draft.tex`.

## Regenerate numbers and figures, then build

Every number in the paper is a macro in `numbers.tex`. Figures live in `../figures/`.
Neither is edited by hand.

```
run.cmd tools/fractal_numbers.py    # results/fractal/*.json -> paper/numbers.tex
run.cmd tools/fractal_figures.py    # -> figures/fig_pred_{main,repl,mech}.png
run.cmd tools/paper_check.py        # undefined or missing macros, citations, figure files
cd paper && pdflatex main && pdflatex main
```

**The page count has not been checked.** There is no LaTeX toolchain on the machine
this was written on. ICRA allows 8 pages including references; build once
(for example on Overleaf) before submitting.

## What the paper claims

- **Outcome-flip scaling.** For a frozen PPO swing-up policy, the probability that two
  simulators ε apart disagree on success falls as ε^α with α ≈ 0.28 over about
  2.5 decades. That gives C_½ = 2^(1/α) ≈ 12× precision to halve outcome ambiguity.
- **Replication**, all pre-registered in `../PREREGISTRATION_FRACTAL.md`: a second
  policy, a mass × servo-lag plane, a zoom on the main boundary, five initial
  conditions, independent CPU dynamics, and a halved physics step.
- **Mechanism.** Finite-time amplification of outcome-divergent twins, which sets
  a numerical floor.

## What it does not claim

- **"Fractal".** Only about 2.5 decades are resolved, and the first registered zoom
  failed.
- **Chaos.**
- **Hardware results.**
- **A state-space effect.** The initial-state test was uninformative, with success
  near 100% over ±0.8 rad.
- **A quantitative cartpole exponent.**
- **Novelty beyond** the fixed-policy, simulator-parameter-space measurement and
  its calibration consequence.
