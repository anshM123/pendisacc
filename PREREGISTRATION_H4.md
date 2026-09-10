# Pre-registration — H4: do equivalence classes belong to the plant or to the interface?

**Committed before any force-interface policy is trained.** `git log` at this
commit contains no `logs/rsl_rl/tip_swingup/*_H4_*` run. The only training run
with `--interface force` at this commit is `IFACE_SMOKE`, a 12-iteration
construction test whose checkpoints are excluded by name from every analysis
below.

## What is already established, and what is missing

`dynamics/symmetry.py` proves symbolically, for planar N-link chains, N = 1..4:
uniform scaling of every link's mass and inertia multiplies the passive
Euler–Lagrange equations by a common constant, which divides out. The passive
trajectory is unchanged, exactly.

The proof has a hypothesis that is easy to skip past: the base coordinate must
be **kinematically prescribed**. That is not a statement about the mechanics.
It is a statement about the actuator. A stiff velocity loop with authority to
spare prescribes the base motion; a force command does not.

`tools/interface_symmetry.py` (commit `ffd6415`) measures the consequence
open-loop, at equal parameter distance ‖Δm‖ = 0.109 kg:

| direction | velocity interface | force interface |
|---|---:|---:|
| uniform `[c,c,c]` | 1.4e-14 | 7.5e-02 |
| transverse `[c,1,1]` | 1.2e-01 | 1.3e-01 |
| transverse `[1,1,c]` | 2.0e-01 | 2.0e-01 |
| **ratio worst/best** | **1.4e+13** | **2.6** |

So in the *equations*, direction is everything under one interface and nearly
irrelevant under the other. What is missing is whether this survives into
**learned closed-loop control**, which is the only version that matters for
sim-to-real. Nothing in the repo tests it.

## H4

> The anisotropy of the reality gap is a property of the action interface. A
> policy commanding cart velocity should be nearly immune to uniform inertial
> error and highly sensitive to a transverse error of the same parameter
> magnitude. A policy commanding cart force, on the same plant, should be
> comparably sensitive to both.

## Design

Two arms. **Only the commanded variable differs.** Plant, servo lag (τ = 100 ms),
transport delay, force clamp (349.5 N), speed limit (4.0 m/s), observation,
reward, episode length, PPO hyperparameters, iterations (1000) and env count
(4096) are identical — enforced in code by `experiments/interfaces.py`, which
derives the force term's servo parameters from the velocity term's rather than
restating them.

| arm | seeds | provenance |
|---|---:|---|
| `velocity` | 3 | **reuse** `C5_S_nominal_s1..s3` — trained 2026-09-09/10 under the frozen protocol, *before this hypothesis was written* |
| `force` | 3 | trained now, `--interface force` |

Reusing the velocity arm is deliberate and is the stronger choice: those
policies cannot have been shaped by H4, because H4 did not exist when they were
trained.

**Authority is matched, not merely mentioned.** The force arm's action is scaled
to the drive's *peak* force and clamped there, so it can command at least as
much force as the velocity loop could ever have asked for. Under-powering the
force arm would confound interface with authority — which is exactly the
confound that produced the P1 failure in `PREREGISTRATION_H3.md`, and it is not
repeated.

**What cannot be equalised, stated up front.** A velocity command of 1.0 and a
force command of 1.0 are different physical requests; both are normalised to
±1 before the `action_l2` penalty sees them, but the mapping to physics
differs. This is inherent to changing an interface and is not removable.

## Measurement

Every policy is swept over link-mass errors at **matched Euclidean distance in
parameter space**, so that magnitude is held fixed and only direction varies:

* uniform `[c,c,c]` for c ∈ {1.0, 1.5, 2.5, 4.0, 8.0}
* transverse `[c',1,1]` and `[1,1,c'']` with c', c'' chosen per c so that
  ‖Δm‖ matches the uniform case exactly

256 episodes each, `TIP-SwingUp-Play-v0`, dead-hang starts.

Define, per policy, at the matched pair for **c = 2.5**:

    A = success(uniform) - mean success(the two transverse directions)     [points]

`A` is the **anisotropy**: how much the direction of an error matters when its
magnitude is held fixed. Reported as the mean over seeds with the per-seed
spread.

## Predictions

**Q1 (primary, directional).**

    A_velocity - A_force  >=  40 points

**Q2 (secondary).** `A_velocity >= 40` on its own — the velocity arm must
actually exhibit anisotropy, not merely exceed a flat force arm.

**Q3 (control).** Both arms reach >= 80% at c = 1.0 in their own interface.

## Kill conditions

* **H4 fails** if `A_force >= A_velocity`. The anisotropy would then not be a
  property of the interface, and the interface account — the paper's central
  generalisation beyond this one robot — is wrong.
* **Q2 fails** if the velocity arm is not itself anisotropic, which would mean
  the open-loop result does not survive into closed loop at all.
* **Reported INCONCLUSIVE, not supported**, if either:
  * any arm fails the Q3 competence gate (a force policy that never learned
    swing-up transfers badly everywhere for reasons that have nothing to do
    with geometry); or
  * the force arm scores < 20% at *every* swept condition including c = 1.0,
    making `A_force` a difference between two floors; or
  * the per-seed spread of `A` within an arm overlaps the claimed
    between-arm gap. Seed is the experimental unit. With 3 seeds the power is
    low and that is stated in advance, as it was for H3 — where the nominal
    arm's seed spread reached 77 points.

## Why this is the load-bearing experiment

Every other result in this project is a fact about one triple pendulum. This
one is a fact about **what kind of object simulator adequacy is**. If it holds,
then "is this simulator accurate enough?" is not answerable from the simulator
and the robot alone — the question is ill-posed until the control interface is
named. If it fails, the geometry story remains true of this plant and stops
generalising, and the paper should say so.
