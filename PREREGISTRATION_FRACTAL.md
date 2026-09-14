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
