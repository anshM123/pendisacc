# Pre-registration — H7: can a stress score predict transfer when in-simulator success cannot?

**Committed before `results/h7_stress.json` exists.**

## Why this, after five failures

Five pre-registered hypotheses have now failed: the action interface (H4), the
training-time geometry (H3 P1), geometry-aware randomisation (H5, both
predictions), and policy-dependence of the estimated geometry (H6, invalidated
by its own control). They share a structure. **Every one attempted to predict
or improve transfer from a property of the *simulator*.**

Meanwhile the quantity that demonstrably moves transfer is the *policy*. At
identical simulator, reward, architecture and iteration count, random seed
alone moves transfer into `R*` by **77 points** while moving in-simulator
success by **0.8** (`results/selection_blindness.json`). Across all 15 H3
policies, Spearman $\rho$(own-simulator success, transfer) $= -0.090$
($p = 0.749$).

So the signal exists and is large, and the standard measurement misses it
entirely. H7 asks whether a different, equally cheap measurement finds it.

## The claim

> A policy's success under a small set of *perturbed* simulators predicts its
> transfer to an unseen deployment target, even though its success in the
> nominal simulator does not.

If true, the practical consequence is immediate and requires none of the
failed machinery: **do not select the nominally best seed; stress-test
candidates and select on that.**

## Material

The **24 policies** that have already been deployed into the frozen `R*`
(`results/claim5/*.json`, `results/h5/*.json`). Transfer scores span
$0.0\%$–$100.0\%$, mean $30.4$, sd $38.2$, with 12 strictly between $2\%$ and
$98\%$. These were all trained and deployed before H7 existed.

## The stress set, fixed here

Five conditions, chosen on principle rather than by looking at correlations:
one per model family, all plausible for the real build, **all disjoint from
`R*`**, and all in directions the anisotropy result marks as sensitive.

| id | condition | family |
|---|---|---|
| `S1` | `mass_scale [1.25, 1, 1]` | rigid body, transverse |
| `S2` | `tau 0.130` | actuation |
| `S3` | `delay_s 0.016` | actuation, dead time |
| `S4` | `joint_friction 0.020` | dissipation |
| `S5` | `mass_scale [1,1,1.20]`, `tau 0.120` | combined |

`R*` is `{tau 0.075, joint_friction 0.005, delay_s 0.008, mass_scale
[1.12,1.09,1.14], cart_mass_scale 1.07}`. No stress condition equals it, and
no stress condition uses `cart_mass_scale` at all, so the score cannot be a
disguised copy of the target.

$$Q(\pi) = \tfrac{1}{5}\sum_{i=1}^{5} P_{S_i}(\pi)$$

## Predictions

**P1 (primary).** $\rho\big(Q(\pi),\, P_{R^\star}(\pi)\big) \geq 0.50$ over the
24 policies, with permutation $p < 0.01$.

**P2 (the comparison that matters).** $Q$ beats in-simulator success by at
least $0.40$ in $\rho$. Own-simulator success is the incumbent selection rule
and scored $-0.090$; a stress score that merely ties it is useless.

**P3 (honesty check).** $Q$ must not be trivially equal to $P_{R^\star}$: if
$\rho > 0.98$ the stress set is effectively a copy of the target and the result
is reported as circular rather than predictive.

## Kill conditions

* **P1 fails** if $\rho < 0.50$ or $p \geq 0.01$. Then nothing cheap measured
  in simulation predicts transfer on this system — which, after five prior
  failures, is itself the honest headline and would be reported as such.
* **P2 fails** if the margin over in-simulator success is below $0.40$.
* The 9 policies at the $0\%$ floor and 3 at the $100\%$ ceiling make $\rho$
  partly a test of separating dead policies from live ones. **A secondary
  $\rho$ restricted to the 12 non-degenerate policies is reported alongside**,
  and if the two disagree the restricted one is believed.

## What this cannot show

One plant, one task, one deployment target. The stress set was chosen by hand
using knowledge of which families are sensitive on *this* system, so H7 tests
whether stress-testing works, not whether the stress set can be chosen
automatically. Choosing it automatically is what H5 attempted, and H5 failed.
