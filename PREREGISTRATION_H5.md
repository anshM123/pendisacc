# Pre-registration — H5: is the geometry *useful*?

**Committed before any H5 policy is trained.** No `logs/rsl_rl/tip_swingup/*_H5_*`
run exists at this commit.

Everything so far is diagnosis. H5 asks whether the diagnosis changes what you
should *do*, which is the only reason a practitioner would care.

## The practical claim

Domain randomisation spends a budget. The budget is real: every unit of
randomisation width makes the training distribution harder and costs final
performance somewhere. The standard practice is to randomise physical
parameters within a box, treating every direction in parameter space alike.

If model space has the geometry the earlier sections describe, that is
wasteful. Randomising along an **equivalence direction** perturbs nothing the
closed loop can see, so it buys no robustness while still costing distribution
width. The same budget spent **transverse** to the equivalence should buy
strictly more.

> **Do not randomise what the interface makes invisible.**

## Design

Three arms, **equal randomisation budget** measured as the expected Euclidean
displacement in link-mass space, `E‖Δm‖`, which is what makes the comparison
fair. Budget is matched by construction and asserted numerically in the runner,
not assumed.

| arm | randomisation | seeds |
|---|---|---:|
| `H5_none` | none — nominal training | 3 |
| `H5_box` | isotropic box on all three link masses (standard DR) | 3 |
| `H5_geom` | same `E‖Δm‖`, confined to the subspace transverse to the uniform direction | 3 |

The uniform direction is `m0/‖m0‖`. `H5_geom` samples in its orthogonal
complement; `H5_box` samples isotropically, so a fraction of its budget lands
along the uniform direction and — if the theory holds — is wasted.

All three deploy into the **same `R*`** already frozen in
`PREREGISTRATION_H3.md`. `R*` contains a mass error of `[1.12, 1.09, 1.14]`,
which is neither purely uniform nor purely transverse, so neither arm is
targeted at it.

## Predictions

**P1 (primary).** `H5_geom` transfers better than `H5_box` at equal budget:

    P_R*(H5_geom) - P_R*(H5_box)  >=  15 points, on seed means

**P2.** Both randomised arms beat `H5_none`, confirming the budget buys
something at all. If not, the experiment says nothing about *how* to spend a
budget that does not work.

## Kill conditions

* **P1 fails** if `H5_box >= H5_geom`. The geometry would then not be
  actionable, and the paper's practical claim is withdrawn — the diagnosis can
  stand without it, but the recommendation cannot.
* **INCONCLUSIVE** if the per-seed spread within an arm overlaps the
  between-arm gap. This is not a formality: the H3 nominal arm showed a
  **77-point** within-arm seed spread at identical settings
  (`results/selection_blindness.json`), which is 5x the effect size claimed
  here. **H5 is very likely to come back underpowered at 3 seeds**, and that is
  stated now rather than discovered later. If it does, the honest report is
  that the effect could not be resolved at this seed count, not that it is
  absent.

## What this cannot show

Three seeds against a 77-point nuisance spread is weak, and one plant with one
task is not evidence that geometry-aware randomisation generalises. H5 at best
establishes that the recommendation is *not contradicted* on the system where
the geometry was characterised. A real version needs more seeds, more plants,
and a real robot.
