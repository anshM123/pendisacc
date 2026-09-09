# Frozen baseline — do not retrain

Frozen 2026-09-08. Everything downstream (simulator population, variational
analysis, twin search, transfer prediction) is measured against THIS policy and
THIS closed loop. Changing any of it invalidates the comparison.

## Actuator architecture

`a_stepdir_position` — Teensy → STEP/DIR → CNC4PC C34SOA6-RS → A6-RS in
POSITION mode. The RL action is a commanded **cart velocity**, not a torque.

RS485 torque mode was considered, briefly frozen, and **reversed**. It stays
reversed. The reasons are in `configs/robot/hardware.yaml`: Modbus RTU needs a
register write + CRC + reply per command, which is poor for a 250 Hz loop,
while STEP/DIR is a bare pulse train the drive accepts to 4 MHz; and the A6-RS's
documented *continuous* torque reference path is the analog input, not Modbus.
Training against raw torque would produce a policy the hardware cannot execute.

## Policies

| seed | checkpoint | nominal | randomised |
|---|---|---:|---:|
| 1 | `logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt` | 1024/1024 | 99.80% |
| 2 | `logs/rsl_rl/tip_swingup/2026-09-05_21-52-24_rel2/model_700.pt` | 1024/1024 | 99.71% |
| 3 | `logs/rsl_rl/tip_swingup/2026-09-05_22-45-54_rel3/model_700.pt` | 1024/1024 | 99.02% |

Seed 3 `model_700` is the reference policy for the analysis.

## Frozen protocol

Reward terms and weights, PPO hyperparameters, observation layout, control rate
(250 Hz), episode length (12 s), 1000 iterations, and selection by measured
dead-hang success. See `RESULTS.md` for what each was and why.

## What is NOT frozen

The simulator parameters `xi`. That is the experimental variable.
