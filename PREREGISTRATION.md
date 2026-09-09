# Pre-registration — H2

**Written 2026-09-09, before the heterogeneous evaluation suite was generated.**
No outcome label for any condition in that suite exists at the time of this
commit. Check `git log` for the ordering: this file is committed before
`results/heterogeneous_*.json`.

The point of writing this down first is that the previous hypothesis was tested
honestly and **lost** (`results/predict_transfer.json`, commit `7771c6e`):
`D_SW` scored rho = -0.698 against a phase-error baseline's -0.747. The metric
below was inspired by *why* it lost, using thirteen actuator conditions whose
labels are already known. Those thirteen therefore cannot be evidence for it.
It is frozen here so that the heterogeneous suite is a genuine test.

---

## H2

> Across heterogeneous parameter **and model-form** mismatch, transfer
> performance is better predicted by model discrepancy propagated through the
> learned closed loop **and projected onto task-failure margins** than by raw
> trajectory fidelity, parameter distance, actuator descriptors, or
> plant-only frequency-domain metrics.

The motivating idea, stated so it can be wrong:

> A simulator is adequate not when it is globally accurate, but when its errors
> are not amplified by the learned closed loop toward task-critical failure
> boundaries.

## The metric, frozen

Implemented in `dynamics/closed_loop.py::transfer_critical_risk`, exactly as
committed here. Deviation obeys the first-order recursion

    e_{t+1} = A_t e_t + d_t,    e_0 = 0,
    A_t = dF_S/dz|_{z_t},       d_t = F_R(z_t) - F_S(z_t)

margins `g_j(x) > 0` with gradients from
`dynamics/closed_loop.py::task_margins`, currently the single rail constraint

    g_rail(x) = 0.60 - |x_cart|

and the statistic is

    R_TC = max_{j,t}  [ -grad g_j(x_t)^T e_t ]_+ / ( g_j(x_t) + eps ),   eps = 1e-3.

Fixed choices, not to be adjusted after labels are seen: `eps = 1e-3`; horizon
2.5 s; 8 initial conditions from seed 3; margins as above; the max (not a mean
or a quantile) over both `j` and `t`.

If the rail margin turns out not to be the operative constraint in some family,
adding a margin is a **change to the metric** and must be reported as such,
with the original number kept.

## Baselines it must beat

All computed on the same conditions, from the same standalone model where they
require one.

| baseline | family |
|---|---|
| `\|tau_equiv - tau_train\|` | parameter distance |
| `\|bandwidth error\|` | actuator spec |
| `\|rise-time error\|` | actuator spec |
| actuator step-response RMSE | actuator engineering |
| `\|phase error at lambda_max\|` | plant-conditioned — **the incumbent, rho = -0.747** |
| short-window trajectory RMSE (0.4 s) | conventional system ID |
| one-step model error, `raw_gap` | direct dynamics fidelity |
| `D_SW` | previous policy-conditioned metric |
| `G_T` | amplification alone, no discrepancy |

The honest contest is against **phase error at lambda_max** and **short-window
trajectory RMSE**. The rest are included so the comparison cannot be accused of
a strawman.

## Primary endpoint

Spearman rank correlation between each predictor and the Isaac-measured success
rate, over the heterogeneous suite, with **held-out model families**: correlation
is also reported on families whose model form appears in no other condition.

Uncertainty by bootstrap that resamples **conditions**, and where multiple
policy seeds are used, resamples seeds as the outer level. Not by treating
individual episodes as independent units — see the note in RESULTS.md section 4
about pseudo-replication.

## Kill criteria, fixed in advance

H2 is **rejected** if any of the following holds on the heterogeneous suite:

1. `R_TC` does not beat the best conventional/plant-conditioned baseline by at
   least **0.10** in |rho|, i.e. `|rho_RTC| - |rho_best_baseline| < 0.10`.
2. That advantage does not survive bootstrap: `P(R_TC better) < 0.80`.
3. The advantage vanishes on held-out model families (present in the pooled
   number but absent when a family is excluded from the fit).
4. `R_TC` is not better than `D_SW`, i.e. the margin projection adds nothing
   over the plain stability weighting.

If H2 is rejected, the finding to report is whichever predictor wins, including
if that is a classical control quantity. A result of the form "feedback-relevant
classical plant distance predicts learned-policy transfer better than simulator
fidelity metrics" would be published as the outcome.

## What the thirteen actuator conditions may be used for

Sanity-checking that the implementation runs, and as a **reported ablation**.
They may not be cited as confirmatory evidence for H2, because `R_TC` was
designed after seeing them.

## The heterogeneous suite

Five families, with combinations, ranges justified by the build rather than
chosen to make anything win:

| family | varied |
|---|---|
| actuation | order (1/2), tau, transport delay, deadband, force clamp |
| rigid body | link masses, inertias, cart + reflected inertia |
| dissipation | none / viscous / Coulomb / Stribeck, cart and joints |
| transmission | belt compliance, transmission damping |
| sensing & control | observation delay, quantisation |

**Hidden pseudo-realities** contain model forms absent from the candidate
family, so that a domain-randomisation objection does not apply.

## Status of the standalone model

Predictors are computed in `dynamics/closed_loop.py`, which is rank-faithful on
a first-order `tau` sweep but pessimistic by 20-27 points in absolute success,
and **not yet validated across model-form changes** (RESULTS.md section 5).
Ground truth is Isaac throughout. If the standalone model's ranking proves
unfaithful on a family, results for that family are reported as unusable rather
than quietly kept.
