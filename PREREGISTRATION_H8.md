# Pre-registration — H8: the CAD defect is a prediction, not a bug report

**Committed before `results/h8/` exists.**

## What was found

Every one of the 16 CAD bodies composing `link1`, `link2` and `link3` carries
SolidWorks' default density of exactly 1000 kg/m³. Of 130 bodies in the
assembly only 8 have a real material, and all 8 are linear rails or rail
bearings in the static base. Every inertial parameter the pendulum dynamics
depend on is therefore fictitious
(`results/cad_density_audit.json`).

The error is not a uniform scale factor. Steel shafts, bearings and collars are
modelled 7.9× too light; PETG arms printed at 25% infill are modelled ~2.1× too
heavy. Correcting both gives link mass factors

    [0.808, 1.060, 0.480]     (central PETG estimate, 480 kg/m³)

which decompose into

    overall scale        c = 0.744
    residual ratio         [1.086, 1.426, 0.646],  distortion 2.21x

## Why this is a test and not merely a correction

`results/h4_grid_velocity.json` establishes, over 90 cells and three seeds,
that overall mass scale explains **0.0%** of transfer variance across a 5×
range while ratio distortion explains **99.9%**. That result was measured
before the CAD defect was known.

It therefore makes a signed, quantitative prediction about this defect, and
the prediction is falsifiable in a way that matters: the defect is **mostly
scale by magnitude** (a 26% mass reduction) but **mostly ratio by effect**.
A naive reading would expect a 26% mass error to be survivable. The landscape
says the 2.21× ratio component will destroy transfer and the 0.744 scale
component will cost nothing.

## Design

The frozen baseline policy, unchanged, evaluated over 256 dead-hang episodes
per condition:

| id | link mass factors | what it isolates |
|---|---|---|
| `nominal` | `[1, 1, 1]` | control |
| `scale_only` | `[0.744, 0.744, 0.744]` | the component predicted FREE |
| `ratio_only` | `[1.086, 1.426, 0.646]` | the component predicted FATAL |
| `corrected_mid` | `[0.808, 1.060, 0.480]` | the actual defect, PETG 480 |
| `corrected_lo` | `[0.732, 0.987, 0.400]` | PETG 400, bracket |
| `corrected_hi` | `[0.893, 1.143, 0.570]` | PETG 570, bracket |

Inertia is scaled with mass on each link, as elsewhere in this project.

## Predictions

**P1 (the scale component is free).** `scale_only` ≥ 90%.

**P2 (the ratio component is fatal).** `ratio_only` ≤ 25%.

**P3 (the defect behaves like its ratio part, not its magnitude).**
`|corrected_mid − ratio_only| ≤ 15` points, i.e. removing the scale component
from the real defect changes little.

**P4 (robust to the PETG bracket).** All three `corrected_*` conditions fall
on the same side of 50%.

## Kill conditions

* **P1 fails** if a 26% uniform mass reduction measurably hurts. The landscape's
  central claim — scale is free — would then not hold at this operating point.
* **P2 fails** if the ratio component is survivable. The anisotropy result
  would not transfer to a defect that arose naturally rather than by design,
  which is the whole point of testing it here.
* **P3 fails** if the real defect behaves unlike its ratio component. The
  decomposition would then not be the right way to read a real model error.

## What follows either way

If the predictions hold, the honest headline is not "we found a CAD bug". It is
that **a real, naturally occurring simulator defect was decomposed by a
previously measured geometry into a free component and a fatal one, and the
decomposition predicted the outcome before it was measured.** That is the first
time in this project that the geometry has been used to predict rather than to
describe.

If they fail, the landscape does not generalise beyond the perturbations it was
measured on, and that is a serious limitation on everything built from it.

## Stated in advance

The PETG effective density is bracketed 400–570 kg/m³ rather than known. The
steel figure (7850) is not in doubt. No link has been weighed; the build is not
assembled. `configs/robot/hardware.yaml` carries a `links_weighed_kg: null`
field for exactly this reason, and until it is filled these corrected masses
remain an estimate of the truth rather than the truth.
