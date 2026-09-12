# Pre-registration — representing uncertainty in similarity coordinates

**Committed before `xi/hidden/` is generated, before any policy is evaluated on
it, and before `results/PI_sysid/` exists.**

The paper now claims that raw physical parameters are the wrong coordinates for
sim-to-real uncertainty. So far that claim is *motivated* (misordered risk; a
correctly decomposed CAD defect) but not *demonstrated*. This registers the two
experiments that would demonstrate it, and one frozen instrument both depend on.

---

## 0. A design gap found while writing this

`tools/t5_redo.sh` evaluates each trained policy with `evaluate.py --run` and no
`--xi`, i.e. on the **nominal** corrected asset. The overnight pre-registration
said T5 would deploy into a target. So the voided 100-vs-0 result, and the
current re-run as written, measure whether randomisation made *training* fail on
nominal — not transfer. That is still worth knowing, and is reported as such,
but it cannot support a transfer claim. The hidden suite below exists to fix it.

---

## 1. Frozen hidden target suite (instrument)

24 target simulators, generated once by `tools/build_hidden_targets.py` with a
fixed seed, six families of four, each written to `xi/hidden/HT_XX.json` with a
sha256 manifest. The generator refuses to overwrite an existing manifest.

| family | varies | why |
|---|---|---|
| `ratio` | per-link mass/inertia, log-normal, σ 0.25, common scale removed | the direction the landscape says is fatal |
| `ratio_scale` | as `ratio` plus a common scale factor in [0.6, 1.6] | realistic: both parts present |
| `dissipation` | joint viscous and Coulomb friction | not touched by any DR arm |
| `actuator` | lag τ, transport delay | not touched by any DR arm |
| `drive` | loop gain k_v and force clamp, independently | unknown on hardware |
| `build` | the CAD-density bracket (PETG 400–570 kg/m³) with cart ±10% | the real error we already found |

Every target is **plausible for the real build**. No target equals nominal, and
none is chosen by looking at any policy's performance. Families `dissipation`
and `actuator` are deliberately outside what the randomisation arms vary: they
measure whether an arm buys robustness it was not trained for, and they should
show no advantage for either arm.

The suite is committed before any policy is evaluated on it and is never edited
afterwards. A second suite would be registered separately.

---

## 2. Randomisation in similarity coordinates (GPU, after T5)

Three arms on the corrected asset, 5 seeds each, 1000 iterations, frozen PPO.
All randomisation is over the 7 log-coordinates
`[m1, m2, m3, I1, I2, I3, m_cart]`, per environment, width `w = 0.35`.

| arm | sampled | status |
|---|---|---|
| `raw` | isotropic draw, used as is | **new** — added to `train.py` as `--dr raw` |
| `transverse` | same draw, common-scale component removed, rescaled | = T5 transverse arm (running) |
| `orbit` | common-scale component only, rescaled | = T5 orbit arm (running) |
| `none` | no randomisation | CORR_s1–s3 exist; **s4, s5 trained for this** |

**Correction to the arm names.** `orbit` scales all seven inertial parameters
together but holds k_v and the clamp fixed, so it is *not* the exact similarity
direction (that also scales k_v, the clamp and dissipation). It is the
common-scale direction the landscape found nearly neutral empirically (0.7% of
variance). Read `transverse` as "ratio-only" and `orbit` as "scale-only".

**Budget matching.** All three arms are rescaled to the same expected log
displacement `E||d||`, verified per run from `dr_applied.json`. This is budget
matched in *raw* coordinates: `transverse` spends the same raw displacement as
`raw`, but none of it on the common-scale direction the landscape says is
behaviourally irrelevant. That is exactly the comparison the claim is about.

**Primary metric.** Mean success over the 24 hidden targets, per seed; seed is
the experimental unit. Each policy is scored at its **final** checkpoint
(no selection on nominal or on the suite), 256 episodes per target, with
`evaluate.py --run_list`.

**Predictions.**

* **P1.** `transverse` − `raw` ≥ **10 points** on the hidden-suite mean.
* **P2.** `orbit` is within 10 points of `none` on the hidden suite:
  randomising only the common scale buys no robustness.
* **P3.** Any `transverse` advantage is concentrated in families `ratio` and
  `ratio_scale`, and absent (within 5 points) in `dissipation` and `actuator`.

**Kill.** `raw` ≥ `transverse` on the hidden-suite mean ⇒ the representational
claim does not hold for domain randomisation and is withdrawn from the paper.

**Inconclusive.** Per-seed ranges overlap the claimed gap. With a documented
77-point seed spread this is a real possibility and is reported as unresolved,
not as absent.

**Also reported, not predicted:** nominal success per arm (what T5 currently
measures), as a measure of training difficulty.

---

## 3. System identification in similarity coordinates (CPU, runs now)

**Premise, checked before registering.** The exact similarity direction scales
every mass-dimensioned quantity together: link masses and inertias, cart mass,
`k_v`, joint damping, and the force clamp. If `k_v` were known, overall mass
scale would be identifiable from trajectories and there would be nothing to
gain. On this build `k_v` is the drive's internal velocity-loop gain, and
`hardware.yaml` marks it **ASSUMED**, not measured. So "heavier robot with a
stiffer drive" and "lighter robot with a softer drive" are genuinely confounded
in motion data.

**Setup** (`experiments/pi_sysid.py`). This is a CPU analytical model, policy-free.
There are 8 random true plants. Link ratios have σ 0.25. The common scale is in
[0.6, 1.6]. Inertia has independent σ 0.10 on top of the link factor. `k_v` has
σ 0.35 and joint damping has σ 0.30. Excitation is open-loop piecewise-constant
commanded velocity (±1.2 m/s) plus a sine, 1.5 s per trajectory, starting from
a hanging position. Noise: joint angles σ 3.8×10⁻⁴ rad (MT6701), cart σ 1×10⁻⁴ m.
The clamp is removed from the model, and the peak force the true plant demands
is reported against the real 349.5 N.

**Estimators**, both nonlinear least squares on noise-normalised trajectory
error, both started from the same prior draw (σ 0.30 in every log-parameter):

* **raw** fits all 9 log-parameters `[m1..m3, I1..I3, m_cart, k_v, b_joint]`.
* **similarity** fits 8 coordinates with the gauge fixed at `log m_cart = 0`,
  i.e. the plant modulo similarity.

**Metrics** for N ∈ {1, 2, 4, 8, 16}:

* held-out angle RMSE on 4 new trajectories
* raw estimator's error along the generator
* both estimators' error in gauge-invariant (quotient) coordinates
* Fisher information at the truth

**What is and is not a finding.** S1 is a *correctness check*. With every
mass-dimensioned parameter included and the clamp off, dimensional analysis
guarantees an exact null direction. If S1 failed, the code would be wrong, not
the theory.

* **S1 (check).** The smallest Fisher eigenvector has |cos| ≥ 0.95 with the
  all-ones generator, and the second eigenvalue is ≥ 10³ × the smallest.
* **S2.** Held-out RMSE is equal for the two estimators at every N (median
  ratio within 10%). Identifying in similarity coordinates does not predict
  trajectories better, because the discarded direction does not affect them.
* **S3.** The raw estimator's error along the generator does not shrink with N
  (at N = 16 its RMS is ≥ 0.5 × the prior σ). Both estimators' quotient error
  shrinks with N (N = 16 RMS ≤ 0.5 × N = 1 RMS).

The honest content of S2 + S3: in raw coordinates, the absolute masses and
`k_v` a practitioner reads off are set by the prior, not by the data, while the
fit looks converged. This restates base-parameter identifiability (Gautier &
Khalil 1990) for a learned-controller setting. It is a representation argument,
**not a new identifiability result**, and it will be written as such. No claim
of needing fewer trajectories is registered.

**Kill.** If S2 or S3 fails with S1 passing, the representation argument for
identification is withdrawn.

**Stated limitation.** If `k_v` is measured on the hardware, the confound goes
away, and so does any identification argument. That would be reported as a
condition on the claim.

---

## Addendum (2026-09-12, before any section-2 training): section 2 not run

A CPU proof (`tools/pi_prove.py`, `results/PI_prove/prove.json`), run with no
Isaac and before any randomisation policy was trained, shows the section-2
comparison cannot test the representational claim:

* **Exact gauge.** Scaling masses, inertias, cart, k_v and F_clamp together by
  c in {0.25, 0.5, 2, 4} changes a 12 s closed-loop trajectory by at most
  7.3e-12.
* **Raw DR is the same as quotient DR without rescaling.** A raw log-box draw
  and its quotient projection give trajectories within 2.8e-7 over 24 draws.
* **Equal-budget quotient DR is only a width change.** It equals raw DR with
  the quotient part widened by 1.107x.
* **The inertials-only common scale (drive held fixed) is also behaviourally
  neutral** for the frozen policy: 100% success at every c. So
  `transverse` vs `raw` is likewise a ~1.08x width change.

The section-2 arms are therefore width variants of one distribution, not
different representations. They are withdrawn rather than run.

Caveat: the CPU model reproduces nominal success for CORR_s1 (100%) but not
CORR_s2 or CORR_s3 (0%). The two identities above are algebraic and hold for any
policy. The sensitivity map (E) is for CORR_s1 only.
