# Swing-up results

**A learned policy swings the triple pendulum up from dead hang and holds it, in
3072 of 3072 nominal simulations, on every seed tried — under a deliberately
pessimistic 100 ms first-order actuator time constant whose pole magnitude is
smaller than the plant's dominant open-loop unstable eigenvalue.**

Date: 2026-09-05. Commit: `1c0d509`.

---

## 1. The headline

| seed | checkpoint | episodes | successes | success rate | early termination | hold fraction |
|-----:|------------|---------:|----------:|-------------:|------------------:|--------------:|
| 1 | `model_800` | 1024 | 1024 | 100.00% | 0.00% | 99.87% |
| 2 | `model_700` | 1024 | 1024 | 100.00% | 0.00% | 99.87% |
| 3 | `model_700` | 1024 | 1024 | 100.00% | 0.00% | 99.87% |
| | **pooled** | **3072** | **3072** | **100.00%** | **0.00%** | **99.87%** |

The figure to quote is the **minimum across seeds**, which is also 100.00%.

> **Do not attach a binomial confidence interval to this.** The evaluation start
> is near-fixed (§2), so 1024 parallel environments are close to 1024 replicas of
> one trajectory rather than 1024 independent Bernoulli trials. "3072/3072
> nominal simulations succeeded" is the defensible claim; a rule-of-three lower
> bound is not, because the independence assumption fails. For a rate with a
> distribution behind it, see the randomised evaluation in §4.

The three runs are independently seeded and reach an identical hold fraction to
four decimal places, which suggests the metric is measuring a property of the
converged behaviour rather than run-specific noise.

## 2. What "success" means here

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

## 3. A representative trajectory

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

## 4. Robustness — over an actual distribution

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

## 5. Sensitivity to the simulator itself

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

## 6. Conditions

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

## 7. The plant these results are about

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

## 8. Actuator headroom

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

## 9. What it took — four defects, each found by measurement

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

## 10. Known limitations

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

## 11. Reproducing

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
