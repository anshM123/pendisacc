# Pre-registration — is the transfer boundary fractal?

**Committed before any map, pair set, or exponent exists.**

Question: for a frozen learned swing-up policy, is the set of simulator
parameters on which it succeeds separated from the set on which it fails by a
fractal boundary? If so, the uncertainty exponent α of that boundary sets how
much better system identification buys certainty about transfer.

Closest prior art (checked 2026-09-14, not exhaustive): fractal basin boundaries in
controlled pendulums (state space); "Fractal Landscapes in Policy Optimization"
(NeurIPS 2023, policy-parameter space); "The boundary of neural network
trainability is fractal" (2024, hyperparameter space). None located treats
**simulator-parameter space of a frozen policy's transfer**. That is the only
claim this study can make.

## Instrument (`experiments/fractal_map.py`)

- Policy: `CORR_s1` model_800 (corrected asset; the policy used for the T4 landscape).
- Task `TIP-SwingUp-Play-v0`, velocity interface, 12 s, success = no early
  termination, tip reached upright, tip upright > 95% of the final quarter.
- **Fixed initial condition for every environment**: link 1 at π + 0.03 rad,
  links 2–3 +0.01 rad, zero velocities, cart at 0. The play task normally
  randomises these; randomness in the IC would make any map speckled for a
  reason unrelated to the boundary.
- Parameter plane: x = log(m1 / m1,nom), y = log(m3 / m3,nom) with m2 and the cart fixed
  (so x, y are log(m1/m2), log(m3/m2) up to constants); inertia scaled with mass.
  Window x, y ∈ [−0.6, 0.6].

## Noise floors (measured, not assumed)

- **d0** within-run: 128 grid cells duplicated in other env slots; disagreement rate.
- **r0** run-to-run: the full 64×64 grid run twice in separate processes; disagreement rate.
- **p0** pair floor: in the exponent run, 600 pairs at ε = 0.

## G1 — map

64×64 grid (4096 environments). Reported: success fraction, d0, r0, image.

## G2 — uncertainty exponent

600 random base points in the window; for ε ∈ {0.3, 0.1, 0.03, 0.01, 0.003, 0.001}
(log units) each paired with θ + ε·(cos φ, sin φ). f(ε) = fraction of pairs with
different outcomes. f_c(ε) = f(ε) − p0. α = slope of log f_c vs log ε over the ε values
where f(ε) exceeds p0 by ≥ 3 standard errors. D = 2 − α.

**Threshold-jitter control.** A flip whose two hold fractions both lie within
0.05 of the 0.95 cut is counted as threshold-straddling. If > 50% of flips at the
smallest usable ε straddle, small α is attributed to the success cut, not the
dynamics.

## Decision rule

- **KILL** if any of:
  - α ≥ 0.8
  - fewer than 3 ε values are usable above the noise floor
  - r0 ≥ 5%, so the simulator is too nondeterministic to map
  - the threshold-jitter control fails
- **CONTINUE** if 0.5 < α < 0.8: report as a rough (non-smooth) boundary and
  continue only with the zoom test.
- **LOCK IN** only if all of:
  - α ≤ 0.5, with ≥ 4 usable scales
  - the zoom window reproduces α within ±0.15. The zoom is half-width 0.06,
    centred on the grid cell with the most disagreeing 4-neighbours; ties go to
    the lowest index.
  - a second policy (`T5_orbit_s1`, final checkpoint) gives α ≤ 0.6 on the
    full window

Only then: finite-time Lyapunov exponents, delay × mass plane, cartpole control.

---

## Addendum 1 (2026-09-14 19:36, before the run it governs)

Outcome of the registered lock-in so far:
- **Full window:** α = 0.282, 6 usable scales. Passes.
- **Second policy (T5_orbit_s1):** α = 0.271. Passes.
- **Zoom: FAILS.** α = 0.041 against the required 0.28 ± 0.15.

The registered zoom centre was an isolated success island inside the failure
region (base success 1–2%). **The fractal claim is therefore NOT locked in, and
this addendum does not change that.**

A new, separately reported test (**Z2**) is registered here. The only thing
that changes is the centre rule; half-width, pairs, ε set and the ±0.15
criterion are unchanged.
- **Centre rule:** take the largest 4-connected success component of grid_A.
  Among its cells that have a failing 4-neighbour, pick the one closest (in
  (x, y)) to the mean position of all such cells.
- **Report:** Z2 is reported alongside the failed zoom, never instead of it.
  If Z2 passes, the claim is at most "the full-window exponent reproduces at
  the main boundary but not at isolated islands".

Also recorded, exploratory and post hoc: on the full window, "power law + constant floor"
(β 0.57, floor 4.8%) fits better than a pure power law (χ² 2.5 vs 8.3).
`results/fractal/pairs_small_EXPLORATORY.json` (ε 1e-3 … 1e-5) tests for that floor.

**Z2 outcome (19:37):** α = 0.315 over 6 usable scales at centre (0.124, 0.029);
|0.315 − 0.282| = 0.03 ≤ 0.15 → **Z2 passes**. Reported with zoom-1's failure.
Exploratory small-ε run: f plateaus near the simulator's slot-noise floor for
ε ≤ 3e-4, so the measurable scaling range is ε ∈ [1e-3, 0.3] (~2.5 decades).

---

## Addendum 2 (2026-09-14, before the runs it governs)

Registered here, with the same pair design, ε set, noise floor and α estimator as G2.

**P2 — actuator plane.** x = log m1 scale, y = log servo-lag scale (τ = 0.1 s · e^y,
set per environment). Policy CORR_s1 model_800, window ±0.6.
- *Why τ and not delay:* transport delay is quantised to whole 4 ms control steps
  in this simulator, so ε-scaling in delay is impossible.
- *Prediction:* α_P2 ≤ 0.5 with ≥ 4 usable scales, i.e. the rough boundary is not a
  mass-only artefact.
- *Failure:* α_P2 ≥ 0.8, or < 3 usable scales.

**C1 — control system.** Stock Isaac-Cartpole-v0, trained 150 iterations with
stock settings, frozen.
- *Setup:* pole IC 0.25 rad; x = log pole-mass scale, y = log cart-mass scale.
  Window ±1.5, widened once to ±3 if the grid is > 98% or < 2% success. 292 pairs
  per ε (4096-env limit).
- *Prediction:* the cartpole boundary is smooth, α_C1 ≥ 0.8. If α_C1 ≤ 0.5 the
  roughness is not specific to the triple pendulum, and that is reported as such.
- *No boundary:* if the window has no boundary after widening, C1 is reported as
  uninformative.

**FTLE (diagnostic, no threshold).** 256 twin pairs at ε ∈ {0, 1e-5, 1e-4, 1e-3} in
log m1; median log separation vs time; growth rate between separations 1e-5 and 1e-2.

**C1 outcome (19:44):** the grid was 98.9% success at ±1.5, so the window was
widened to ±3 (85.7% success).
- **Pairs:** flips 3.1% at ε = 0.3, 1.0% at 0.1, 0.7% at 0.03, and 0 at ε ≤ 0.01;
  floor 0.
- **α criterion:** only 1 usable ε, so α is inestimable and C1 is **uninformative
  on α as registered**.
- **Descriptive:** the boundary is 2.9% of grid cells (triple pendulum: 21–31%), with
  no isolated cells and no noise floor. That is qualitatively consistent with the
  smooth-boundary prediction.
- **Caveat:** the task is not difficulty-matched (a 5 s balance from 0.25 rad) and
  292 pairs per ε.

**P2 outcome (19:42):** α = 0.246 over 6 usable ε. Prediction holds.

---

## Addendum 3 (2026-09-15, before any run it governs): final replication tests

All tests use the G2 α estimator, with a bootstrap 95% CI over pairs (1000
resamples, usable-ε set fixed from the point estimate). Success criterion,
episode length, policy (CORR_s1 model_800), ε set {0.3, 0.1, 0.03, 0.01, 0.003, 0.001}
and seed 20260914 are unchanged unless stated. No window or threshold is changed
after outcomes.

**R1 — independent dynamics (CPU).**
- *Setup:* the corrected analytical model (`dynamics/closed_loop.py`, explicit 0.4 ms
  substeps, no PhysX), with the **same 600 × 7 parameter pairs** as G2 (same RNG
  layout). IC in absolute angles: θ1 = π + 0.03, θ2 = θ1 + 0.01, θ3 = θ2 + 0.01. Success
  criterion as Isaac (|x| ≤ 0.6 throughout, tip > 0.9 at some time, tip > 0.9 for
  > 95% of the final quarter).
- *Supports:* α ≤ 0.6 with ≥ 4 usable ε, and CI upper bound < 0.8.
- *Weakens the paper to Isaac-specific:* α ≥ 0.8.
- *Ambiguous:* anything else; reported as such.

**R2 — initial conditions (Isaac).** Five fixed ICs frozen here, as (link-1 absolute
offset from hanging, joints 2–3 relative offset):
- IC0: (+0.030, +0.010) (= G2)
- IC1: (−0.030, +0.010)
- IC2: (+0.045, −0.015)
- IC3: (−0.045, +0.015)
- IC4: (+0.010, −0.020)

Pairs on the m1 × m3 plane, window ±0.6.
- *Supports:* median α < 0.5 and ≥ 4 of 5 ICs with α < 0.8.
- *Reporting:* every IC is reported, including any with < 3 usable ε.
- *Mechanism check:* ICs are now applied through a per-environment IC table.
  IC0 must reproduce G2's outcomes exactly; if not, that is reported as a failure
  of the new mechanism.

**R3 — state-space outcome boundary (Isaac).** Physics nominal.
- *Plane:* x = initial link-1 angle offset added to IC0, y = initial joint-2 relative
  offset added to IC0 (rad).
- *Window:* grid 64×64 at half-width 0.05 rad. Widen ×4 (0.2, then 0.8) at most
  twice while grid success is > 98% or < 2%. Then pairs in the final window,
  same ε set in rad.
- *Supports:* α ≤ 0.6 with ≥ 4 usable ε.
- *Smooth:* α ≥ 0.8.
- *Uninformative:* the window is still degenerate after two widenings.

**R4 — numerical resolution (Isaac).** Primary plane, pairs, with physics dt
halved (1/1000 s, decimation 4, so the control rate stays 250 Hz and the policy
is unchanged).
- *Supports:* α within ±0.15 of 0.282 with ≥ 4 usable ε. The ε = 0 floor is
  reported and may move.
- *Indicates the exponent depends on integration:* |α − 0.282| > 0.15.

**Reporting.** Every α goes in one table with CI and C_½ = 2^(1/α). C_½ is valid
only over the resolved ε range.

**R2 outcome (16:35):** α = 0.28, 0.29, 0.33, 0.32, 0.33 for IC0–IC4 (median 0.32),
each with 6 usable ε. 5 of 5 are below 0.8. **Supports.** The IC-table mechanism
reproduced G2's 8400 outcomes exactly.

**R3 outcome (16:39):** the initial-state grid had 100.0% success at ±0.05 and at
±0.2, and 99.9% at ±0.8 rad, so it was still degenerate after two widenings.
**Uninformative as registered.** The pairs run at ±0.8 was also 99.9% success.
Reported as a contrast: the swing-up is insensitive to the initial state over this
range.

**R4 outcome (16:41):** α = 0.376, CI [0.303, 0.477], 5 usable ε, floor 0.83%.
|0.376 − 0.282| = 0.094 ≤ 0.15, so R4 **supports** as registered.
Recorded as exploratory: with physics dt halved, 12.2% of individual outcomes on the
same parameter pairs change, yet the exponent stays in the same regime.
