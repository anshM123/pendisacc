# The geometry of the reality gap

**Simulator error is not ordered by magnitude.** On this system a 150% error in
every link mass is harmless while a 50% error in *one* link is fatal; an
actuator with matched bandwidth and *faster* rise time takes transfer from 100%
to 0%; and success is not monotone in gravity. Two simulators whose trajectory
error differs by 1.5% produce transfer of 0% and 100%.

What decides transfer is the **direction** of the error relative to (i) exact
dynamical equivalences of the plant, (ii) the authority of the control
interface, and (iii) the learned policy's failure boundaries.

Date: 2026-09-10. Baseline frozen at `1c0d509`; see `configs/FROZEN_BASELINE.md`.

> **Two pre-registered hypotheses were tested and REJECTED** before arriving
> here, and both are reported rather than buried (§4). A scalar
> policy-conditioned fidelity measure did not beat conventional trajectory
> fidelity. That failure is what motivated looking at geometry directly.

---

## 1. Error magnitude does not order transfer

All measured in Isaac, 256 episodes per condition, one frozen policy.

| model error | magnitude | transfer |
|---|---|---:|
| every link mass x2.5 | +150% | **100.0%** |
| every link mass x10 | +900% | **100.0%** |
| **link 1 alone x1.5** | **+50%** | **0.0%** |
| link 1 alone x5 | +400% | 0.0% |
| force clamp cut to 40 N (a quarter) | -89% | 100.0% |
| joint viscous damping 0.08 | large | 100.0% |
| **transport delay 20 ms** | **tiny** | **0.0%** |
| **joint Coulomb friction 0.008** | **tiny** | **3.5%** |

A uniform error twenty times larger than the asymmetric one is harmless, while
the smaller one is fatal. Gravity is not even monotone: 9.90 gives 100%, 10.50
gives 51.2%, 11.00 gives 80.9%.

## 2. Some errors are exact dynamical equivalences

`dynamics/symmetry.py` proves, symbolically rather than numerically, that for a
planar N-link chain on a **kinematically prescribed** base, scaling every link
mass and inertia by a common factor multiplies the passive equations of motion
by that factor, which then divides out:

| N | uniform scaling | one link only | + joint friction |
|---|---|---|---|
| 1 | **EXACT** | — | fails |
| 2 | **EXACT** | fails | fails |
| 3 | **EXACT** | fails | fails |
| 4 | **EXACT** | fails | fails |

"EXACT" is `simplify(residual(c·m, c·I) − c·residual(m, I)) == 0`, an algebraic
identity. Numerically the induced change in the angular accelerations is ~1e-15
at scales from 1.3x to 100x, against 7.5% for a single-link change.

The two counterexample columns show the hypotheses are load-bearing: uniformity,
and the absence of any non-scaling generalised force on the passive coordinates.

**The base mass never appears.** Once the base is prescribed it is outside the
passive equations entirely, so the equivalence is a property of the *actuation
interface* as much as of the mechanics. Under torque command the base is not
prescribed, the cancellation fails, and the same mass error stops being free.

## 3. The interface gates the equivalence — causally

A real velocity servo does not prescribe the base exactly; its authority is
finite. Frozen policy weights, only the drive changed:

| | kv 400, clamp 349 N | kv 4000, clamp 7000 N |
|---|---:|---:|
| uniform mass x20 | **0.0%** | **100.0%** |
| uniform mass x30 | 0.0% | **100.0%** |
| **asymmetric [5,1,1]** | 0.0% | **0.0%** |
| uniform mass x5 (control) | 100.0% | 100.0% |

Lifting the gain alone leaves 0.0%; lifting the clamp alone leaves 0.0%. The
two limits bound each other, `F = min(clamp, kv·(v_max − ẋ))`, and during
swing-up ẋ reaches 3.5 of 4.0 m/s so a large clamp is unusable without gain.
**Together they restore transfer completely, without retraining.**

Authority restores an *intact* symmetry and cannot manufacture a broken one —
the asymmetric error stays at 0.0% under 20x force and 10x gain.

Getting here required four pre-registered mechanisms, three of which were
falsified: a force-clamp threshold (wrong location), velocity-loop stiffness
(kv 400→4000 changed nothing), force saturation alone (clamp 349→7000 changed
nothing), and absolute impulse deficit (does not transfer across clamps). The
first estimate failed because the force demand was computed over an open-loop
state distribution rather than the trajectory the policy actually flies — 22.7 N
against a true 98.7 N. *A closed-loop quantity was estimated on an open-loop
distribution*, which is the paper's own thesis in miniature.

## 4. Scalar fidelity cannot certify transfer

Two pre-registered attempts to build a policy-conditioned scalar that beats
conventional fidelity **failed** (`PREREGISTRATION.md`, `results/h2_test.json`).
Over 38 heterogeneous conditions:

| predictor | ρ vs measured success |
|---|---:|
| **short-window trajectory RMSE** | **−0.506** |
| \|phase error at λ_max\| | −0.505 |
| `D_SW` (stability-weighted) | −0.415 |
| `R_TC` (task-margin projected) | −0.406 |
| raw model gap | −0.390 |

Conventional trajectory fidelity won. `R_TC` failed all four frozen kill
criteria. What *did* hold: weighting the same discrepancies by closed-loop
amplification moved ρ from −0.390 to −0.415…−0.698 depending on the family, so
amplification carries real signal — it just does not beat a good simple
baseline.

But the winner cannot **certify** transfer. With the tolerance frozen in
advance (10% relative, ≥50-point gap, plausible conditions only):

| | trajectory RMSE | transfer |
|---|---:|---:|
| `act_2nd_z05` — 2nd-order actuator, ζ=0.5 | 1.1624 | **0.0%** |
| `dis_jdamp_004` — joint viscous damping | 1.1448 | **100.0%** |
| difference | **1.5%** | **100 points** |

Different model families, indistinguishable fidelity, opposite outcomes. Four
such pairs clear the threshold. So trajectory fidelity is **informative
statistically and ambiguous locally** — a sharper claim than "insufficient",
and one our own data supports rather than contradicts.

## 5. Does the geometry change what RL learns? Partly, and less than predicted

Everything above is one frozen policy across many simulators. The sim-to-real
question is `S_i -> pi_i -> R*`: train *in* each simulator, deploy into a common
reality. 15 policies, six training simulators, one deployment target `R*` fixed
in `PREREGISTRATION_H3.md` before any of them existed.

| training simulator | seeds | success in `R*` (per seed) | mean |
|---|---:|---|---:|
| `S_nominal` | 3 | 100.0, 71.1, 23.0 | 64.7% |
| `S_twinB` — joint damping 0.004 | 3 | 100.0, 73.8, 22.7 | **65.5%** |
| `S_transverse` — mass [1.5,1,1] | 2 | 7.4, 2.3 | 4.9% |
| `S_equiv_c8` — uniform x8 | 2 | 2.0, 0.0 | **1.0%** |
| `S_twinA` — 2nd-order actuator, ζ=0.5 | 3 | 2.0, 0.8, 0.0 | **0.9%** |
| `S_delay` — 12 ms dead time | 2 | 1.6, 0.0 | 0.8% |

**P1 failed, and failed in the direction opposite to the prediction.** The
equivalence-direction simulator (uniform x8) trained *worse*-transferring
policies (1.0%) than the transverse one (4.9%). The pre-registration named this
as a kill condition and it is reported as one.

The likely reason sharpens the theory rather than rescuing it, but it is
**post-hoc and is labelled as such**: at c = 8 the force demand is ~8x nominal,
so *training* at c = 8 happens in a force-saturated regime. §3 already showed
the symmetry is only realisable while the drive has authority. A policy
*evaluated* at c = 8 with authority restored scores 100%; a policy *raised*
there learns a strategy fitted to saturation and does not come back. So

> equivalence for a fixed policy is not equivalence for training.

That is a real distinction and it is not one the paper predicted. Testing it
properly requires an authority-matched c = 8 training arm, which has not been
run.

**P2 was supported, by 64.6 points** — the two trajectory-fidelity twins, 1.5%
apart in RMSE, trained policies transferring at 65.5% and 0.9%. But the honest
reading is weaker than the number: `S_twinB` is joint damping 0.004, a nearly
negligible perturbation whose seed pattern (100.0, 73.8, 22.7) is almost
identical to nominal's (100.0, 71.1, 23.0), because the same seed under a
negligible ξ gives nearly the same policy. So P2 is "one twin is near-nominal
and the other is destructive", not "two comparable rivals". The fidelity
measure still cannot separate them, which is the claim — but the claim is
carried by twinA being severe.

**P3 was supported.** Nominal (64.7%) beat transverse (4.9%), so the protocol
is not at fault.

## 6. In-simulator performance carries no information about transfer

Exploratory — this axis was not named in `PREREGISTRATION_H3.md` and no
threshold was frozen for it. It is also the most uncomfortable number here.

Every one of the 15 policies was selected the only way a practitioner without
access to reality can select: best measured success in its **own** training
simulator.

| | mean | range | spread |
|---|---:|---|---:|
| own training simulator | 99.7% | 98.0 – 100.0 | **2.0 points** |
| deployment into `R*` | 27.1% | 0.0 – 100.0 | **100.0 points** |

Spearman ρ(own-sim, `R*`) = **−0.090**, permutation *p* = 0.749, *n* = 15.

The controlled version removes the simulator from the comparison entirely.
Within the nominal arm the simulator, protocol, reward, architecture and
iteration count are identical; the random **seed** is the only difference:

| policy | own-sim | in `R*` |
|---|---:|---:|
| `S_nominal_s1` | 100.0% | 100.0% |
| `S_nominal_s2` | 99.2% | 71.1% |
| `S_nominal_s3` | 100.0% | 23.0% |
| **spread** | **0.8 points** | **77.0 points** |

Seed alone moves transfer by 77 points while moving the selection criterion by
0.8. Two consequences. First, methodological: the quantity everyone selects on
is uninformative about the quantity everyone wants, so any single-seed
sim-to-real comparison — including several of ours — is underpowered by
construction. Second, for this paper: a 77-point within-arm spread is 2.5x the
30-point threshold P2 was tested against, which is why the seed-spread rule was
frozen in `PREREGISTRATION_H3.md` in advance.

## 7. What is not yet established

* **Does the geometry change what RL learns?** Everything above is one frozen
  policy across many simulators. The sim-to-real question is
  `S_i → π_i → R*`. Pre-registered in `PREREGISTRATION_H3.md`; 15 policies
  training as of 2026-09-10.
* **Is the geometry useful?** No geometry-aware randomisation or policy
  selection experiment has been run.
* **Does it survive reality?** The hardware is not built. Every number here is
  simulation.
* The twin pairs come from the discovery set and need fresh confirmation.

---

# Appendix: the baseline these results are measured against

## 8. What "success" means here

Reward is not the metric. It misled this project three separate times — most
starkly when a run reported reward 108 and rising while measuring **0%** actual
success. Success is therefore defined physically and counted
(`experiments/evaluate.py`). An episode succeeds only if **all three** hold:

1. it never terminated early — no rail excursion, no fall;
2. the tip reached upright (normalised height > 0.9) at some point;
3. the tip is **still** upright over the final quarter of the 12 s episode,
   for more than 95% of that window.

Every evaluation episode starts from **dead hang**. There are no easy starts in
the measurement, regardless of what the training distribution contains. That
separation is deliberate: it is what lets an easier *training* curriculum be
used without it flattering the *reported* number.

The nominal start is deliberately narrow — joint 1 at `pi ± 0.05` rad, joints 2
and 3 at `± 0.02` rad, all joint rates zero, cart exactly centred and at rest,
and no sensor noise. That is a **capability** test, not a robustness test, and
it is why the numbers in §1 carry no confidence interval. §4 repeats the
measurement over an actual distribution.

## 9. A representative trajectory

From `results/rollout_success.npz` (seed 3, `model_700`), rendered in
`figures/swingup_success.gif` using the real CAD outlines:

| quantity | value | limit |
|---|---:|---:|
| time from dead hang to upright | **1.57 s** | — |
| tip height once caught | **+1.000** | 1.000 = perfectly upright |
| fraction of final 3 s upright | **100.0%** | — |
| cart travel used | 0.919 m | 1.2 m usable |
| peak cart speed | 3.47 m/s | 4.0 m/s rated |

Both figures are one-off swing-up transients. Once caught, the cart settles into
a 0.247 m band at under 0.91 m/s, and **no episode in any evaluation ever hit the
rail limit** (0 terminations in 3072). Peak speed is the tighter of the two
margins (87% vs 77%), and required force is an order of magnitude inside the
drive's capability (§7).

## 10. Robustness — over an actual distribution

§1 is a capability test from a near-fixed start. This is the same policy
measured over **independent random draws**, which is what a rate with a
confidence interval requires:

| perturbation | range (uniform) |
|---|---|
| every link angle | ± 0.10 rad (±5.7°) |
| every link rate | ± 0.20 rad/s |
| cart position | ± 0.10 m |
| cart velocity | ± 0.10 m/s |
| sensor noise on every observation | Gaussian, σ = 0.01 |

| seed | checkpoint | success | failures | early termination | ever upright | hold |
|-----:|---|---:|---:|---:|---:|---:|
| 1 | `model_800` | 99.80% | 2 | 0.20% | 100.00% | 99.87% |
| 2 | `model_700` | 99.71% | 3 | 0.29% | 100.00% | 99.87% |
| 3 | `model_700` | **99.02%** | 10 | 0.98% | 100.00% | 99.87% |
| | **pooled** | **99.51%** | **15 / 3072** | | | |

**Pooled 3057/3072 = 99.51%, Wilson 95% CI [99.20%, 99.70%].** The draws within
a seed are independent, so this is far better founded than the §1 numbers.

> One nuance before quoting the interval. It pools three policies whose true
> success probabilities visibly differ (99.80 / 99.71 / 99.02), and a vanilla
> binomial interval treats them as draws from one common Bernoulli parameter.
> For descriptive reporting 3057/3072 is fine; for inference across PPO seeds
> the seed belongs in the hierarchy, and the main transfer experiments should
> use seed-stratified or hierarchical bootstrap intervals rather than
> pseudo-replicating thousands of environments as though seed variability did
> not exist. The **worst seed, 99.02%**, is the safe figure to quote.

Two things the failure mode makes clear. **Every failure is a rail excursion** —
`ever reached upright` is 100.00% in all three seeds, so the policy never fails
to swing up or to catch; the 15 losses are all cart-bound terminations during
the transient. And **hold fraction is unchanged at 99.87%**, identical to the
nominal case, so once caught the capture is not degraded by the perturbation or
the sensor noise at all.

Reading this against §1: perfect nominal success and ~99.5% perturbed success
is the expected shape. The gap is the honest measure of margin, and it points at
rail headroom during swing-up rather than at the controller.

## 11. Sensitivity to the simulator itself

Section 4 randomises the initial condition and the sensors. It does **not**
randomise `xi`, the simulator. Doing so changes the picture, and not
flatteringly.

**The policy is tuned to the actuator it trained on.** Evaluated in Isaac at
actuator time constants it never saw, 256 episodes each:

| actuator `tau` | 20 ms | 50 ms | 80 ms | **100 ms** | 
|---|---:|---:|---:|---:|
| success | 1.6% | 57.0% | 93.8% | **100.0%** |

Success falls away in *both* directions from the trained value. **Increasing
the actuator bandwidth fivefold takes success from 100% to 1.6%.** Faster is not
automatically better here, because the policy is itself part of the closed loop
-- which is the point: this is not a robustness curve with a comfortable
plateau, it is evidence that the policy exploits the particular actuator
dynamics it was trained against. Whatever the real drive turns out to be, it
will not be exactly 100 ms.

**Model form matters more than parameters.** Replacing the first-order lag with
a second-order actuator of the *same* time constant collapses success to 0%,
while adding Stribeck friction, a 20 mm/s deadband, an 8 ms transport delay and
+-4% mass errors on top of a first-order drive changes nothing:

| pseudo-reality | success |
|---|---:|
| nominal | 100% |
| + Stribeck friction (cart and joints) | 100% |
| + Stribeck **and a second-order drive, same tau** | **0%** |

A first-order lag contributes at most 90 degrees of phase; a second-order one
reaches 180. Same nominal bandwidth, different closed loop. No amount of
domain randomisation *over tau* covers a change in the actuator's order, which
is exactly why a simulator population built from parameter noise alone would be
misleading.

These two results are the reason the project exists, and they say the baseline
in Sections 1 and 4 should be read as a *nominal* capability result, not as
evidence of transferable control.

### Status of the standalone analysis tool

`dynamics/closed_loop.py` reproduces the control path outside Isaac so the
variational analysis runs on a CPU. Cross-checked against Isaac at actuator
settings the policy never trained on:

| `tau` | standalone (n=48) | Isaac (n=256) | difference |
|---|---:|---:|---:|
| 20 ms | 4.2% | 1.6% | +2.6 pts |
| 50 ms | 37.5% | 57.0% | -19.5 pts |
| 80 ms | 66.7% | 93.8% | -27.1 pts |
| 100 ms | 100.0% | 100.0% | 0.0 pts |

The ordering is exact over these four points, but the tool is **systematically
pessimistic** by 20-27 points in the middle of the range, agreeing only at the
extremes; the bias persists at n=48, so it is not sampling noise.

The honest statement of its status: *the standalone model reproduces the
monotonic ranking over the tested first-order `tau` sweep, but is not calibrated
for absolute success and is not yet validated across model-form changes.* Four
scalar-`tau` points do not establish that it will correctly rank friction,
dead time, second-order actuators, mass error, compliance, or combinations --
which are exactly what twin-finding needs. It is therefore used for cheap
candidate DISCOVERY, and every publishable comparison is confirmed in Isaac.

## 12. Conditions

These are not idealised-actuator results.

| | |
|---|---|
| servo lag `tau` | **100 ms**, first-order, on the commanded cart velocity |
| actuator bandwidth | 10 rad/s |
| plant instability `lambda_max` | **15.54 rad/s** (64.4 ms divergence time) |
| actuator pole / `lambda_max` | **0.64** — see the note below |
| control rate | 250 Hz (sim dt 2 ms, decimation 2) |
| action | commanded cart velocity, clipped to ±4.0 m/s in physical units |

> **On the 0.64 ratio.** Comparing `1/tau` against `lambda_max` is a useful
> heuristic, not a stabilisability threshold — a first-order lag is minimum
> phase and adds no fundamental obstruction, so nothing forbids control at
> ratios below 1. The defensible statement is: *the policy succeeds despite a
> deliberately pessimistic 100 ms first-order actuator time constant, whose pole
> magnitude is smaller than the plant's dominant open-loop unstable eigenvalue.*
> §7 shows an LQR achieving the same thing with modest authority, which is what
> makes the claim concrete rather than rhetorical.

The 100 ms figure is a deliberate worst case, roughly double the ~50 ms typical
for drives of this class, adopted because the real drive has not been measured.
It is a *lag*, not a transport delay; `servo_transport_delay_s` is 0 and
unmeasured. A first-order lag contributes at most 90° of phase and is
compensable with full state feedback; a dead time is not. **If one thing is
measured on the real drive, it should be dead time, not settling time.**

## 13. The plant these results are about

Derived from the CAD by `tools/sw_dump_assembly.py` and validated end-to-end by
`tools/validate_asset.py` (`results/asset_validation.json`, all checks pass):

| check | result |
|---|---|
| rigid-body masses, Isaac vs CAD | agree to 3e-8 relative |
| unstable growth rate, analytic vs Isaac | **15.5378 vs 15.5223 rad/s (0.10%)**, log-fit r² = 0.99999985 |
| energy conservation, passive 3 s | 0.033% drift |
| upright trajectory, 50 ms | 0.14% of the angle scale |
| hanging trajectory, 3 s | 1.43% of the oscillation amplitude |

Key masses: cart **0.79053 kg** (0.42659 kg CAD + **0.36394 kg** reflected
drivetrain inertia, `J_rotor / r_pulley²` — 46% of the moving mass, and absent
from the CAD entirely). Links 0.17752 / 0.10026 / 0.07747 kg, joint spacing
0.24999 / 0.25000 m.

> **Caveat on the link masses.** Every link body in the CAD carries density
> exactly 1000 kg/m³, which is SolidWorks' default for a part with no material
> assigned. The physical links are **PETG prints**; the CAD density is a
> placeholder, and what is unsettled is the final manufactured mass, COM and
> inertia, not the material. Final published mass properties will be obtained
> from the completed parts rather than inferred from a nominal density.
> `tools/material_sensitivity.py` sweeps the density anyway:
> `lambda_max` moves only 6.7% even for steel, because a pendulum's divergence
> rate is set by geometry rather than mass, and required force stays far inside
> the drive. So the *control* consequence is small — but the published plant
> parameters should come from **weighing the actual links once built**, with COM
> and inertia measured or estimated per link, not from any assumed density.

## 14. Actuator headroom

`tools/actuator_bandwidth.py` designs an LQR on the augmented plant (link
states + drive lag state, commanded cart velocity as input) and asks what it
costs to catch a 5° lean:

| `tau` | bandwidth | BW / λ | catch time | peak force | peak `v_cmd` |
|---:|---:|---:|---:|---:|---:|
| 10 ms | 100 rad/s | 6.44 | 0.59 s | — | 1.17 m/s |
| **100 ms** | **10 rad/s** | **0.64** | **0.60 s** | **8.7 N** | **1.29 m/s** |
| 400 ms | 2.5 rad/s | 0.16 | 0.67 s | — | 1.46 m/s |

Against a drive rated at **99.7 N** (349.5 N peak) and **4.0 m/s**. The
requirement degrades gently with lag because holding a lean needs only ~1 m/s²
and `v_cmd − v_ref = a·tau`. Force is not the constraint at any plausible lag.

## 15. What it took — four defects, each found by measurement

Every one of these was diagnosed only after the reward curve had already lied
about it. The order matters: each fix was necessary and none was sufficient.

| # | defect | evidence | after the fix |
|---|---|---|---|
| 1 | **Suicide was cheaper than surviving.** Isaac Lab scales every reward by `step_dt`, so hanging for a full episode cost −69.6 while crashing into the rail cost −1.0 — ~70× cheaper. | seed 2 ended **100%** of episodes on the rail for 550 consecutive iterations, mean length 63 of 3000 steps | early terminations 100% → **2.7%**; success still 0% |
| 2 | **Action std had no cost.** `clip_actions` clips before `env.step`, so `action_l2` reads the *clipped* action and is blind to σ above the clip, while entropy keeps paying for more. | σ → **47.2**; every sampled action saturates, so the policy trains as a bang-bang controller | holding fixed; seed 1 **100%**, seed 2 **0%** |
| 3 | **A rotational limit cycle.** With no bound on link energy the policy spun the chain instead of stopping. | `tools/diag_capture.py`: median `Σω²` = **11660** (~62 rad/s per link), passing *through* upright **15.3%** of steps | seed-independent; **99.2%** at iteration 400, vs 1500 iterations before |
| 4 | **Std broke loose mid-run.** Effort weight −0.05 slowed the runaway but did not stop it. | at σ 1.80: reward 212 → 122, capture 6.91 → 3.42, rail exits 1.9% → **26.9%**, success 99.2% → **30.5%** | σ held at 0.72–0.93 in all three seeds |

Two attempts that **failed** and are recorded because they are informative:
widening the capture Gaussian's *angle* axis (0.25 → 1.0) moved the capture
reward from 0.0000 to 0.0006 and changed nothing, because the term is a product
and `exp(−Σω²/σ_v²)` is annihilated at `Σω²` = 11660 for any sane `σ_v`. Both
failures came from the same mistake — sizing a reward term against an *assumed*
operating point (4 rad/s) instead of a measured one (62 rad/s).

## 16. Known limitations

**Checkpoint selection is required, not a convenience.** Within a single run,
performance oscillates violently:

| seed | per-checkpoint success (256 episodes, every 100 iterations) | ≥95% |
|---|---|---:|
| 1 | 0, 0, 0, 0, 0, **97.7**, 1.2, 41.4, **100**, 0, 74.2 | 2 / 11 |
| 2 | 0, 0, 0, 0, 0, 0, **98.8**, **100**, 23.0, **100**, 24.2 | 3 / 11 |
| 3 | 0, 0, 0, 0, 0, **100**, **100**, **100**, **99.6**, **100**, 13.3 | 5 / 11 |

Some individual checkpoints collapse entirely — seed 1's `model_900` terminates
100% of episodes. **The last checkpoint must never be used.** Seed 3 shows a
contiguous plateau (500–900), seeds 1 and 2 show isolated spikes.

The cause is a side effect of the reverse curriculum: with starts uniform over
the circle, ~17% of episodes begin near upright and dominate the training
reward, so dead-hang skill becomes a minority of the objective and drifts
freely. `model_900` of seed 1 reports training reward 165 while measuring 100%
early termination from dead hang. Selecting on measured success is what makes
this safe — but the honest description of the method is *train, then select on a
dead-hang measurement*, not *train and take the result*.

**The obvious next improvement** is to anneal the start distribution back toward
hanging once the catch is learned, keeping ~5% easy starts to prevent
forgetting. That should turn the spikes into a flat top and remove the reliance
on selection. Not done.

**Not yet established:** anything about transfer. These are single-simulator
results. The simulator population, nonlinear fingerprints, twin search and
pseudo-realities are all still empty.

## 17. Reproducing

```powershell
# three seeds, 1000 iterations each (~35 min per seed on an RTX 5070 Ti)
powershell -ExecutionPolicy Bypass -File tools\train_seeds_reliable.ps1

# score every checkpoint of a run by measured dead-hang success
.\run.cmd experiments\evaluate.py --run logs\rsl_rl\tip_swingup\<dir> --stride 100 --num_envs 256

# re-test the winner at higher precision
.\run.cmd experiments\evaluate.py --checkpoint <path\model_N.pt> --num_envs 1024

# watch it in the Isaac Sim renderer
.\run.cmd experiments\play.py --task TIP-SwingUp-Play-v0 --experiment tip_swingup --checkpoint <path\model_N.pt>
```

Checkpoints behind the headline table:

| seed | path |
|---|---|
| 1 | `logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt` |
| 2 | `logs/rsl_rl/tip_swingup/2026-09-05_21-52-24_rel2/model_700.pt` |
| 3 | `logs/rsl_rl/tip_swingup/2026-09-05_22-45-54_rel3/model_700.pt` |

Raw evaluation output: `results/eval_rel{1,2,3}.json` (sweeps),
`results/eval_rel{1,2,3}_confirm.json` (1024-episode confirmations),
`results/diag_capture.json`, `results/asset_validation.json`.
