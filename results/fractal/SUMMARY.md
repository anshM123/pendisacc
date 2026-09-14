# Transfer-boundary study: results to date (2026-09-14)

Pre-registration: `PREREGISTRATION_FRACTAL.md` (gate, addendum 1, addendum 2), each
committed before the runs it governs. Every number below comes from a committed JSON
file in `results/fractal/`.

## Setup
- **Policies:** frozen PPO swing-up. CORR_s1 model_800 is the primary; T5_orbit_s1 model_999 is the second policy.
- **Episodes:** Isaac Lab, 12 s, one fixed initial condition shared by every environment.
- **Success:** no early termination, the tip reaches upright, and the tip is upright for more than 95% of the final quarter.
- **Uncertainty exponent α:** f(ε) is the fraction of random parameter pairs, a distance ε apart in log parameters, whose outcomes differ.
  The ε = 0 floor f₀ is subtracted, and α is the slope of log(f − f₀) against log ε over the usable ε (≥ 3 standard errors above the floor).

## Results

| Test | Plane / policy | α | Usable ε | f(ε = 0.001) | Floor f₀ | Verdict |
|---|---|---|---|---|---|---|
| G2 full window | m1 × m3, CORR_s1 | **0.282** | 0.3 … 0.001 | 5.2% | 0.33% | pass (α ≤ 0.5) |
| Zoom 1 (registered rule) | isolated success island | 0.041 | 5 | 2.8% | 0.5% | **FAIL** (needed 0.28 ± 0.15) |
| Z2 zoom (addendum 1) | main boundary | **0.315** | 6 | 10.5% | 2.0% | pass |
| Second policy | m1 × m3, T5_orbit_s1 | **0.271** | 6 | 7.2% | 0.33% | pass (≤ 0.6) |
| P2 actuator plane (addendum 2) | m1 × servo lag τ, CORR_s1 | **0.246** | 6 | 9.3% | 2.2% | pass (≤ 0.5) |
| C1 cartpole control (addendum 2) | pole mass × cart mass | pending | | | | predicted smooth (α ≥ 0.8) |

**Noise and repeatability**
- **Run-to-run:** 0.0%. The same grid in a separate process is identical.
- **Within one run:** identical parameters in different environment slots disagree 0.3–2.3%.
- **Success-cut jitter:** ≤ 3% of flips have both hold fractions within 0.05 of the 0.95 cut.

**Scaling range (exploratory)**
- For ε ≤ 3e-4 the flip rate levels off at 2–2.5%, against a 1.0% floor.
- The measurable scaling range is therefore ε ∈ [1e-3, 0.3], about 2.5 decades.

**Separation growth (diagnostic)**
- Median twin separation scales with ε and grows modestly.
- Twins that end > 0.3 apart diverge at 1.5–2.4 s⁻¹. That is 4% at ε = 0 (slot noise) and 10% at ε = 1e-3, reaching 0.1 in about 2 s.
- Over 12 s this amplifies solver-level noise to order one, consistent with the floor.

## What can be claimed
- **Measured exponent.** A frozen learned swing-up policy's success/failure boundary in
  simulator-parameter space has uncertainty exponent α ≈ 0.25–0.32 over 2.5 decades.
  This is reproduced across two parameter planes (mass–mass, mass–actuator lag), a
  second policy, and a zoom on the main boundary.
- **Consequence.** Near that boundary, halving the chance of mispredicting transfer
  needs about 2^(1/0.28) ≈ 12× better parameter precision.
- **Mechanism.** Outcome-divergent trajectories separate exponentially at about 2 s⁻¹,
  so the simulator's own numerical noise sets a floor below ε ≈ 3e-4.

## What cannot be claimed
- **"Fractal" in the strict sense.** 2.5 decades, limited by simulator precision, and
  the registered zoom-1 test failed. Isolated success islands behave as independent
  coin flips (α ≈ 0).
- **Chaos.** Finite-time divergence is shown only for the subset of divergent twins.
- **Hardware.** Nothing here has been tested on hardware.
- **Novelty.** Not established. The closest known work finds fractal boundaries in
  policy-parameter space (NeurIPS 2023) and hyperparameter space (2024). A dedicated
  literature pass is still needed.
