# Pre-registration — overnight programme, 2026-09-11 → 12

**Committed before any of `results/T1..T6/` exists.**

## Design principle

Seven hypotheses have failed in this project (H2's two metrics, H3 P1, H4, H5,
H6, H7, H9). Every one tried to fit a *tractable summary* of a nonlinear
system: a scalar score, a linear subspace, a quadratic form, a stress average.
Two succeeded (H8, H10) and both rested on an **exact structural object** — a
proved symmetry — rather than on a fitted one.

So everything below is built on the exact scaling group derived in
`dynamics/symmetry_group.py`, not on the estimated metric `G` that H9 showed
does not extrapolate.

**Baselines.** All GPU work uses the rebuilt (density-corrected) asset and
`CORR_s2/model_999`, which scores 100.0% on it. Every sweep includes an
explicit `c = 1` control — the omission of one is what voided the first H10.

---

## T1 — Is the derived group the WHOLE null space?

The group is 3-dimensional. That is a completeness claim about *scaling*
symmetries, but it does not rule out other behaviourally-null directions that
are not scalings.

> **Claim.** Every behaviourally null direction in parameter space lies in the
> derived group. There are no accidental null directions.

**Method.** 24 random unit directions in the 8-D block
`[m1,m2,m3,I1,I2,I3,m_cart,kv]`, each applied at a magnitude matched to a
displacement already known to be fatal, plus the group direction and the c=1
control. Every direction gets the clamp scaled in proportion to its `kv`
component, so authority is never the reason a direction fails.

**P1.** The group direction scores ≥ 90%.
**P2.** At most 1 of 24 random directions scores ≥ 90%.
**KILL.** If ≥ 4 random directions are null, the group is not the whole null
space and a much larger equivalence structure exists that we have not
characterised — which would be a bigger result than the theorem, and must be
reported as such rather than buried.

---

## T2 — Is the symmetry policy-independent?

Everything measured in this project is policy-dependent, catastrophically so:
77 points of transfer spread at identical settings, and in-simulator success
*anti*-correlated with transfer (ρ = −0.570) among working policies.

> **Claim.** The symmetry is a property of (plant, interface) and not of the
> controller, so *every* competent policy is invariant under it.

**Method.** Every policy that reaches ≥ 90% on the corrected asset, evaluated
at c = 1 and under the full generator at c = 16 and c = 64.

**P1.** Every competent policy retains ≥ 90% under the generator.
**P2.** The spread of (c=64 success − c=1 success) across policies is ≤ 10
points — i.e. the invariance itself does not vary by seed.
**KILL.** If any competent policy loses > 25 points, the symmetry is not
policy-independent and the theorem's practical reading is much weaker.

---

## T3 — Does the EXACT quotient distance predict transfer?

H9 ranked the 38-condition suite by `θᵀGθ` with a *fitted* metric and lost to
trajectory RMSE (−0.322 vs −0.532). Its diagnosed failure was that `G` was
fitted at 2% and the suite reaches +150%.

> **Claim.** Replacing the fitted metric with the exact group fixes it.
> Distance to the **symmetry orbit** — the component of the parameter error
> that the group cannot absorb — predicts transfer better than trajectory
> fidelity.

**Method.** For each expressible condition, project `log θ` onto the group's
orthogonal complement and use the residual norm. CPU only.

**P1.** |ρ| ≥ |ρ(traj_rmse)| + 0.10 on the same conditions.
**P2.** ρ < 0.
**KILL.** P1 fails ⇒ even the exact object does not beat conventional
fidelity, and the honest headline is that trajectory fidelity is simply the
right tool for this problem.

---

## T4 — Does the anisotropy replicate on the corrected robot?

The landscape result (ratio 99.9%, scale 0.0%) was measured on the *uncorrected*
asset. The corrected robot is materially different: cart-to-links mass ratio
2.20 vs 1.20, `l_com` up 53.6% on link 1, tip link less than half its old mass.

> **Claim.** The anisotropy is a property of this class of plant, not of the
> particular wrong numbers it was discovered on.

**Method.** The full 6 × 5 landscape `m = m0·[c(1+δ), c, c]`, on the corrected
asset with `CORR_s2`, with authority telemetry per cell.

**P1.** Ratio distortion explains ≥ 90% of variance; scale ≤ 5%.
**KILL.** If scale explains > 20%, the headline result does not survive
correcting the model it was found on, which would be serious.

---

## T5 — Is randomisation along the orbit free?

H5 found domain randomisation *hurt* (66.3% → 31.5% box, 9.9% geom), but used
the fitted geometry and a 7-D block.

> **Claim.** Randomising strictly *along* the exact symmetry orbit costs
> nothing at all, because the training distribution is then a single plant
> presented in different coordinates.

**Method.** Two arms, 2 seeds each, equal budget: `orbit` randomises only along
the group; `transverse` randomises only in its complement. Both deploy into the
frozen `R*`.

**P1.** `orbit` ≥ 90% of the no-randomisation baseline.
**P2.** `orbit` > `transverse`.
**KILL.** If randomising along an exact symmetry still degrades training, then
DR damage is not about which directions are sampled and H5's failure has a
different cause than geometry.

---

## T6 — Does the group generalise to another robot class?

Symbolic only, CPU. Run the same solver on: a cart-pole under **torque**
command; a 2-link arm with payload under **joint-position** command; and the
triple pendulum under **force** command.

> **Claim.** The solver returns a non-trivial group for each, and the group
> *changes with the interface* — which is the interface hypothesis (H4) in the
> form that might actually survive, since H4 tested behaviour and this tests
> structure.

**P1.** Each system yields a group of dimension ≥ 1.
**P2.** The triple pendulum's group under force command **differs** from its
group under velocity command.
**Reported either way**; this is the generality probe and it has no kill
condition because a null result is informative.

---

## Engineering rules, from failures this week

1. Sequential execution; **no "is python running" guards** — a process hung in
   `simulation_app.close()` blocked two queues for 159 minutes.
2. Every sweep carries an explicit control condition.
3. Verify output files, never exit codes.
4. Results written incrementally so a crash loses one condition, not a night.
