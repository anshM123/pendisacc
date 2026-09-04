# Beyond Trajectory Fidelity

**Which nonlinear structures of a robot–policy closed loop must be preserved between
simulation and reality for learned control to transfer?**

Triple inverted pendulum on a cart. Isaac Lab supplies the controlled simulation
population, PPO acts as both controller and adversarial probe of exploitable
simulator physics, and nonlinear dynamics explains which policies survive reality.

Target: ICRA 2027, paper deadline **2026-09-15 23:59 PST** (supplementary video
window 09-17 → 09-22).

---

## Frozen stack

Pinned deliberately; do not chase upstream releases before submission.

| Component | Version | Note |
| --- | --- | --- |
| Python | 3.11.16 | **forced** — Isaac Sim 5.1.0 ships cp311 wheels only |
| Isaac Sim | 5.1.0 (pip) | latest *stable*; 6.0.x is cp312 |
| Isaac Lab | 2.3.2 | source, editable, at `C:\Users\anshm\Downloads\IsaacLab-main\IsaacLab-main` |
| PyTorch | 2.7.0+cu128 | cu128 required for Blackwell / RTX 50-series (sm_120) |
| RL | rsl-rl-lib 5.0.1 | skrl + sb3 also installed for baselines |
| Env | `env_isaaclab/` | `env_cad/` is a separate py3.12 env for SolidWorks COM |

`OMNI_KIT_ACCEPT_EULA=YES` must be set for any non-interactive Isaac run.

Reinstall from scratch: `tools/install_isaaclab.ps1`.

### Why not the plan's Isaac Lab 3.0.0-beta2 + Sim 6.0.1

The project plan pinned the beta. It was dropped because 2.3.2 was already on disk,
is the latest stable release, has the better-documented Windows path, and nothing in
this project needs a 6.0 feature. A beta stack 13 days from a deadline is schedule
risk with no scientific upside. Isaac Sim 6.0.1 remains available as a PhysX-version
robustness check later, since it is a separate venv.

---

## Hardware reality

| | |
| --- | --- |
| GPU | RTX 5070 Ti Laptop, **12 GB VRAM** |
| RAM | 31.4 GB |
| Shared | GT LiDAR lab RTX 4090, 24 GB |

Note the VRAM: Isaac Sim's stated minimum is 16 GB, so the laptop is **below the
official floor**. The scene here is tiny and headless, so it is expected to work,
but env-count scaling must be benchmarked rather than assumed. Laptop = think and
filter; 4090 = execute finished batches (target 25–40 GPU-hours total). **Superseded
by the measured benchmark below — the laptop turned out to be roughly half a 4090.**

---

## The machine, as measured from CAD

Extracted from `Major Assembly.SLDASM` — **not** from the `.x_t`, and not hand-typed.

Assembly frame: **y is vertical** (gravity along −y), **z is the rail axis**
(1.524 m = 60 in travel), links swing in the **y–z** plane about axes parallel to **x**.

Three MT6701 magnetic encoders, one per revolute joint, each with a ball bearing
and set-screw collar. Pivot spacing came out at **0.249995 m** and **0.249998 m**,
i.e. 250 mm by design — strong evidence the joint extraction is correct.

| body | CAD parts | mass (kg) | L (m) | l_c (m) | I about proximal x (kg·m²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (static) | 72 | 7.622661 | — | — | — |
| cart | 42 | 0.144809 | — | — | — |
| link1 | 7 | 0.177516 | 0.24999 | 0.10794 | 3.945e-3 |
| link2 | 8 | 0.100256 | 0.25000 | 0.16525 | 3.628e-3 |
| link3 | 1 | 0.077468 | free end | 0.15787 | 2.840e-3 |

Moving mass 0.500 kg against a 7.62 kg static frame.

---

## CAD pipeline

```
Major Assembly.SLDASM
   │  tools/sw_dump_assembly.py        (SolidWorks COM, read-only)
   ▼
assets/triple_pendulum/cad/bodies_raw.csv     130 leaf bodies, 69 unique parts
   │  configs/robot/body_grouping.yaml         5 rigid bodies
   │  tools/aggregate_bodies.py                parallel-axis composition
   ▼
configs/robot/triple_pendulum_params.yaml     <- single source of truth
```

The `.x_t` Parasolid export is **not** used for physics. It is a flat solid dump with
no assembly structure, no joints and no mass properties, and Isaac Sim has no reliable
Parasolid importer. Mass properties come from SolidWorks itself; geometry is visual
only (see the collision note below).

### Self-validation

`sw_dump_assembly.py` does not trust its own transform convention. It composes every
body under both `R` and `Rᵀ` and keeps whichever reproduces the assembly centre of mass
that SolidWorks reports:

```
convention R   : com err = 6.425e-03 m
convention R^T : com err = 1.370e-06 m      <- chosen
mass summed = 8.122710 kg   solidworks = 8.122671 kg   rel err = 4.8e-06
```

Any future CAD change re-runs this check automatically. `dump_meta.yaml` records the
residuals for provenance.

### SolidWorks COM notes (hard-won)

`IAssemblyDoc.GetComponents` returns **late-bound** dispatch objects. On those:
`GetPathName`, `IsSuppressed`, `GetChildren` come back as already-evaluated
*properties*, not callables; `GetModelDoc2` and `Transform2` raise
`DISP_E_MEMBERNOTFOUND`; `CastTo("IComponent2")` fails because the CoClass exposes no
type info. Workarounds actually used: open each unique part **by path** for mass
properties, and read placement from **`GetXform`** (16 doubles) instead of `Transform2`.
Run `tools/sw_probe_component.py` if a future SolidWorks release shifts this again.

---

## Layout

```
assets/    CAD dump, meshes, generated USD
configs/   robot params, physics levels, simulator population, experiments
dynamics/  analytical model, linearisation, Poincare, Floquet, continuation, basins
source/    Isaac Lab external project (manager-based env, mdp terms, actuators, agents)
experiments/ population generation, training, transfer evaluation, twin search
hardware/  state interface, policy runner, safety, logging
tools/     CAD extraction + install/smoke scripts
```

---

## Measured physics

From the analytical model (`dynamics/analytical/validate.py`, all checks pass):

| | |
| --- | --- |
| upright unstable modes | **3**, dominant λ = **+15.51** rad/s |
| fastest divergence time constant | **64.5 ms** |
| hanging oscillation modes | 0.859 / 1.522 / 2.595 Hz |
| energy conservation, 20 s large-angle swing | 1.5e-10 relative |

(λ was +16.30 / 61 ms before the reflected drivetrain inertia was included; adding
the rotor's 0.74 kg reflected mass to the 0.145 kg cart moved it by ~5%.)

**The ~65 ms time constant is the single most important number for hardware.** It
sets the control-rate floor, the latency budget, and how quickly a policy must
react. It also explains why trajectory-level comparison of this system is close to
meaningless: over 1 s, that is ~10⁷ amplification of any discrepancy. The project's
central claim arrives on its own the first time you try to compare two integrators.

## Laptop benchmark

`tools/benchmark.ps1`, cartpole, headless, RTX 5070 Ti Laptop (12 GB):

| num_envs | env-steps/s |
| ---: | ---: |
| 64 | 4,938 |
| 256 | 20,173 |
| 1024 | 79,552 |
| 2048 | 153,760 |
| **4096** | **269,846** |

4096 envs runs fine despite the 12 GB VRAM being under Isaac Sim's stated 16 GB
minimum. Isaac Lab's published desktop-4090 cartpole figure is ~510k env-steps/s,
so this laptop is at roughly **half a 4090**. That materially changes the compute
plan: the laptop is not merely a "think and filter" machine, and the shared 4090
budget can be spent on final multi-seed batches rather than on routine training.

## Actuator model: velocity command, not force

The drive is STEP/DIR into an A6-RS position loop, so a force command is not
something the hardware can execute. The action space is a signed cart VELOCITY
(scale 4.0 m/s = 3000 rpm x 12.732 mm), and the servo's inner loop is modelled as

    F = clamp( D * (v_cmd - v_cart), +-F_max ),   D = 400 N.s/m, F_max = 200 N

Paper-quality replacement, still to write: explicit pulse integration into a
position reference plus the A6 position controller.

### Reflected drivetrain inertia

GT2 2 mm / 40 T gives r = 12.732 mm, so r^2 = 1.62e-4 and the rotor reflects to
the cart as `m = J_rotor / r^2`:

| | J_rotor | reflected mass |
| --- | ---: | ---: |
| 400 W | 0.35e-4 kg m^2 | 0.216 kg |
| 750 W | 1.20e-4 kg m^2 | **0.740 kg** |

Against a **0.1448 kg** CAD cart, the drivetrain DOMINATES the translating
inertia. Every run before this had a cart ~6x too easy to accelerate. It is
added from `hardware.yaml` in both the URDF builder and the analytical model so
the two cannot drift apart.

## Isaac articulation: validated

`tools/rebuild_asset.ps1` runs CAD params -> URDF -> USD -> validation.
`results/asset_validation.json`, all checks pass:

| check | result |
| --- | --- |
| A. 4 DOFs: `cart_slide` + `joint1..3` | pass |
| B. revolute drive stiffness & damping = 0 (genuinely passive) | pass |
| C. link masses vs CAD | pass, rel err ~1e-8 |
| D1. upright, 0.05 s vs analytical | 0.23% |
| D2. hanging, 3 s vs analytical | 1.67% |
| D3. energy drift, passive | 4.5e-4 relative |
| D4. growth rate along dominant eigenvector | **16.286 vs 16.303 (0.10%)**, log-fit R2 = 0.9999998 |

D4 perturbs along the dominant eigenvector so the response is a pure exponential
from t=0. That validates the eigenVECTOR as well as the eigenvalue: a different
mode shape in Isaac would pull the motion off the ray and wreck the fit. The Isaac
plant and the symbolic plant are the same dynamical system.

## The joint-angle convention bug (read before touching either model)

**URDF/Isaac joint angles are RELATIVE to the parent link. The analytical model
uses ABSOLUTE angles from vertical.** They agree to first order near q = 0 and
diverge as the pendulum moves.

This cost a full debugging cycle and looked exactly like a physics error:

| comparison | raw (mixed conventions) | via `rel_to_abs` |
| --- | ---: | ---: |
| small perturbation, 0.3 s | 105.6% | **0.19%** |
| upright, 0.05 s | 886.4% | **2.02%** |

It also silently corrupted the hanging test. `[0, pi, pi, pi]` in *relative*
coordinates puts link2 at an absolute angle of 2*pi -- pointing straight **up** --
a folded configuration that drove the links through the rail and injected 8.1 J
into a 1.1 J system. The correct hanging state is `[0, pi, 0, 0]`.

Everything crossing between the two worlds goes through `dynamics/conventions.py`,
which carries `HANGING_REL`/`UPRIGHT_REL` and a round-trip self-test. Route new
code through it rather than re-deriving the mapping.

Related: the links carry **no collision geometry**. The analytical model has no
contacts, so collision shapes would make the two plants different systems. The
cart's end stops come from prismatic joint limits instead.

## Reward scaling: every term is multiplied by step_dt

`RewardManager.compute()` does:

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

This applies to ONE-SHOT terms too, which is easy to miss. At dt = 4 ms a
`is_terminated` weight of -5 is worth **-0.02** for the whole episode, against a
shaping term that accumulates to ~1.2. Measured consequence on the swing-up task:
**83% of episodes ended by running off the rail**, because doing so was free.

Rule: a one-shot penalty needs `weight ~ desired_cost / dt`. For dt = 4 ms and a
cost of 1.0 reward unit that is `weight = -250`.

Continuous shaping terms are unaffected in spirit -- they just read as
"reward per second" rather than "reward per step", which is the intended design.

## Isaac renderer is unusable on this machine

GUI init and headless `--video` both die in the RTX/hydra path
(`DriverShaderCacheManager` error, then an access violation in `_prepare_ui`).
Root cause is visible in `nvidia-smi`: **BAR1 memory exhausted, 16355 / 16384 MiB**,
with ~30 GPU-accelerated desktop apps open. BAR1 is the PCIe aperture the renderer
needs; compute is unaffected, which is why training runs at 287k steps/s while
anything that renders crashes.

Workaround (and the better path for paper figures anyway):

```
experiments/rollout.py   headless, records absolute link angles to results/*.npz
tools/animate.py         matplotlib animation from that npz -- no Isaac renderer
```

Closing most GPU apps should restore the real GUI if it is ever needed.

## The environment is a standalone install, not a venv

`env_isaaclab/` holds the interpreter at its ROOT (`python.exe` + `python311.dll`
+ `DLLs/` + `Lib/`), the standard layout. It is NOT a venv: `pyvenv.cfg` is
retired and there is no dependency on uv's managed interpreter.

Consequences:

* Run things as `run.cmd <script>` or `env_isaaclab\python.exe <script>`.
* **Every `Scripts\*.exe` is a dead uv trampoline** pointing at the old
  interpreter path; they fail with *"uv trampoline failed to canonicalize
  script path"*. Use `python.exe -m <module>` instead
  (`run.cmd -m pip`, `run.cmd -m isaacsim`). Shims exist for `pip` and
  `isaacsim`.
* Do NOT copy `python*.dll` into `Scripts\` to "fix" the trampolines. That
  shadows the real DLLs and breaks compiled extensions.

## h5py must match Isaac Sim's HDF5

Isaac Sim ships its own `hdf5.dll` (**1.14.6**) and loads it first. h5py 3.16
is built against **HDF5 2.0**, so its `_errors.pyd` binds to the wrong library
and dies with `0xc0000139` ENTRYPOINT_NOT_FOUND -- which takes down the whole
`isaaclab_tasks` extension. Pinned to **h5py 3.12.1** (HDF5 1.14.2), which loads
with only a version warning.

Symptom to recognise: h5py imports fine in plain Python but fails inside Kit.

## Known Isaac-on-Windows gotchas

* `OMNI_KIT_ACCEPT_EULA=YES` is required for any non-interactive run.
* **Kit takes over stdout after launch.** Console output from a script is not a
  reliable record — every check writes its result to `results/*.json` instead.
* **URDF conversion and simulation must not share a process.** Doing both crashes
  with an access violation: the URDF importer loads
  `omni.kit.tool.asset_importer`, and `SimulationContext`'s stage re-init renders
  while those extensions are being torn down. Hence `build_usd.py` (build step)
  and `validate_asset.py` (runtime) are separate.
* Kit often hangs on shutdown, leaving a zero-CPU `python.exe`. The run has
  already finished; check the result JSON rather than the exit code. Those
  zombies also hold their log file open, which makes the *next* run's output
  redirection fail silently — kill them between runs.
* The stock Isaac Lab demo scripts loop on `while simulation_app.is_running()` and
  never terminate headless. Use `tools/smoke_bounded.py` for automated checks.

## Status

- [x] Stack installed and frozen; torch 2.7.0+cu128 sees the 5070 Ti
- [x] CAD → validated rigid-body parameters (link assignment confirmed by author)
- [x] Isaac Lab smoke test + env-count benchmark to 4096 envs
- [x] Analytical model derived and self-validated
- [x] URDF generated from CAD; USD converted
- [x] Articulation validated: DOFs, passive joints, masses, model cross-check
- [ ] Timestep convergence study (plan section 11)
- [ ] Actuator + friction models, parameter registry `xi`
- [ ] PPO balance, then swing-up
- [ ] Simulator population, pseudo-realities, nonlinear metrics

### Open modelling gap

Two `R6Linear_Bearings` instances are suppressed in the CAD, under
`R6Base Assembly-1`. Physically they ride the rail with the cart, so they are
*moving* mass currently missing from `cart` (which reads only 0.1448 kg — light for
a carriage). Cart mass is a `xi` parameter anyway, so this is a known uncertainty
rather than a blocker, but it should be closed before hardware identification.
