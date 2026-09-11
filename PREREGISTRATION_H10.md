# Pre-registration — H10: does the derived generator beat the measured one?

**Committed before `results/h10/` exists.**

## The prediction under test

`dynamics/symmetry_group.py` solves for the scaling symmetry group of the
3-link chain on a velocity-loop-driven cart and returns, at fixed control rate,
the generator

    m1, I1, m2, I2, m3, I3, m_cart, kv     all scaled by the same factor c

This was **derived, not guessed and not measured**. It is strictly larger than
the direction found empirically in `results/geom_confirm.json`, which scaled
only `m_cart` and `kv`. That partial direction scored 97.7% at c = 16 and
**0.0% at c = 32**.

The symbolic result says the partial direction was only approximately null —
close because the cart dominates the light links — while the full generator is
*exactly* null.

## The clamp is part of the prediction

The drive force is `clip(kv (v_cmd − ẋ), ±F_clamp)`. Under the generator `kv`
scales by c, so the required force scales by c. `F_clamp` is a saturation, not
a term in the equations, so the symbolic search cannot see it — but a
symmetry that demands more force than the drive can deliver is not realisable.
This is the same authority boundary §3 established for the inertial symmetry,
arriving in a new place, and it gives the experiment its sharpest contrast.

## Conditions

Frozen baseline policy, 256 dead-hang episodes each.

| id | `mass_scale` | `cart_mass_scale` | `kv` | `f_clamp` |
|---|---|---|---|---|
| `partial_c32` | 1 | 32 | 12800 | 349.5 |
| `full_c32` | 32 | 32 | 12800 | 349.5 |
| `full_c32_clamp` | 32 | 32 | 12800 | 11184 |
| `full_c16_clamp` | 16 | 16 | 6400 | 5592 |
| `full_c64_clamp` | 64 | 64 | 25600 | 22368 |
| `partial_c32_clamp` | 1 | 32 | 12800 | 11184 |

## Predictions

**P1 (the derived generator is realisable where the measured one was not).**
`full_c32_clamp` ≥ 90%, against `partial_c32` = 0.0% already measured.

**P2 (the clamp is what breaks it, not the mass).**
`full_c32` ≤ 25%, i.e. without scaling the clamp the same generator fails.

**P3 (the full generator is genuinely better than the partial one).**
`full_c32_clamp` − `partial_c32_clamp` ≥ 20 points. If scaling the clamp alone
rescues the partial direction too, then the link terms in the generator are
not doing any work and the symbolic result adds nothing to the measurement.

**P4 (it holds far out).** `full_c64_clamp` ≥ 90% — a 6400% error in every
inertial parameter in the machine, transferring perfectly.

## Kill conditions

* **P1 fails** ⇒ the derived generator is not realisable and the symbolic
  method does not produce usable predictions, regardless of its internal
  validation.
* **P3 fails** ⇒ the symbolic search told us nothing the measurement had not
  already found, and should be reported as a formalisation rather than a
  discovery.

## What is already known and must not be re-derived as if new

`cart_only_x16` = 1.2%, `both_x16` = 97.7%, `both_x32` = 0.0%
(`results/geom_confirm.json`). P1 and P2 are being tested against numbers that
already exist, which is the only reason this is a sharp experiment rather than
a sweep.
