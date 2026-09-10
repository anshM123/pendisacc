# Pre-registration — H3: does simulator-error geometry change what RL learns?

**Committed before any policy in this experiment is trained.** No entry in
`logs/rsl_rl/tip_swingup/*_C5_*` exists at this commit; check `git log`.

Everything so far is `pi_0 -> many xi`: one frozen policy evaluated across
simulators. That measures the robustness of one network. The sim-to-real
question is `S_i -> pi_i -> R*`, and it is the one claim the project has no
evidence for.

## H3

> If simulator error has the geometry the earlier results indicate, then error
> along an equivalence direction should damage the LEARNED policy much less
> than a smaller error transverse to it; and two simulators indistinguishable
> by trajectory fidelity should be able to train policies with materially
> different transfer.

## The deployment target, fixed in advance

    R* = { tau 0.075, joint_friction 0.005, delay_s 0.008,
           mass_scale [1.12, 1.09, 1.14], cart_mass_scale 1.07 }

Mixed across three families (actuation, dissipation, rigid body), all values
plausible for the real build, equal to none of the six training simulators, and
absent from the 38-condition discovery suite.

**Disclosed calibration.** Three candidate targets were screened with the
FROZEN policy to find one that discriminates; `R*_a` and `R*_b` floored at 0.0%
and were discarded, `R*_c` scored 40.2% and was chosen. That is calibration of
the measuring instrument, not a result, and it used only the frozen policy --
no policy in this experiment existed yet. A target at 0% or 100% cannot
separate anything.

## Training simulators

| simulator | xi | seeds | role |
|---|---|---:|---|
| `S_nominal` | none | 3 | reference |
| `S_equiv_c8` | uniform mass x8 | 2 | far along the EXACT equivalence direction |
| `S_transverse` | mass [1.5, 1, 1] | 2 | transverse: breaks the mass-ratio symmetry |
| `S_twinA` | 2nd-order actuator, wn 28, zeta 0.5 | 3 | fidelity twin A |
| `S_twinB` | joint viscous damping 0.004 | 3 | fidelity twin B |
| `S_delay` | 12 ms dead time | 2 | near-boundary, different mechanism |

15 policies. Protocol frozen from `configs/FROZEN_BASELINE.md`: 1000
iterations, 4096 envs, unchanged reward, observation and PPO hyperparameters.
Checkpoint selected by measured dead-hang success **in its own training
simulator**, which is what a practitioner with no access to reality would do.

Then every policy is evaluated in `R*`, 256 episodes, `TIP-SwingUp-Play-v0`.

## Predictions

**P1 (directional, the geometry gives a signed prediction).**
`S_equiv_c8` sits vastly farther from `R*` in parameter distance than
`S_transverse` -- an 8x mass error against a 1.5x one -- yet lies along a
direction the passive dynamics cannot see. So

    P_R*(pi_equiv_c8)  >  P_R*(pi_transverse)

by at least **20 percentage points**, on seed means. If parameter magnitude
governed transfer this must come out the other way round, so the prediction is
falsifiable in a way that matters.

**P2 (non-directional, deliberately).**
`S_twinA` and `S_twinB` are indistinguishable to short-window trajectory RMSE
for the frozen policy (1.5% apart) yet gave 0.0% and 100.0%. As TRAINING
simulators, they should produce policies whose transfer differs by at least
**30 percentage points** on seed means. No direction is predicted -- the
geometry gives no principled reason to say which twin trains the better policy,
and inventing one after the fact would be unfalsifiable.

**P3 (control).** `S_nominal` should transfer at least as well as
`S_transverse`. If a policy trained in the nominal simulator is beaten by one
trained in a knowingly wrong simulator, something is wrong with the protocol
rather than with the theory.

## Statistics

Seed is the experimental unit, not the episode. Report the mean over seeds per
simulator with the per-seed spread; bootstrap over seeds for the P1 and P2
comparisons. With 2-3 seeds per arm the power is low and that is stated up
front: these are effect sizes of 20-30 points, not 5-point differences, and any
comparison whose seed spread overlaps the claimed gap is reported as
inconclusive rather than as support.

## Kill conditions

* **P1 fails** if `pi_transverse` transfers as well as or better than
  `pi_equiv_c8`. That would directly contradict the equivalence-direction
  account and must be reported as such.
* **P2 fails** if the twin-trained policies transfer within 30 points of each
  other. The twin result would then be a property of one frozen policy and not
  of the simulators, which materially weakens the paper's Claim 5.
* **P3 fails** if the nominal arm is not competitive, indicating a protocol
  fault.

If H3 fails, the honest outcome is that simulator-error geometry governs the
robustness of a fixed policy but does not measurably change what RL learns --
which is a narrower claim than the paper wants, and would be reported as the
result.
