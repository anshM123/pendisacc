"""
PPO training entry point for the triple pendulum.

  python experiments/train.py --task TIP-Balance-v0 --num_envs 4096 --headless

Add --video --enable_cameras to record clips during training (slower).
"""

from __future__ import annotations

import argparse
import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a triple-pendulum policy with RSL-RL.")
parser.add_argument("--task", type=str, default="TIP-Balance-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--xi", type=str, default=None,
                    help="path to a JSON file of simulator perturbations -- the TRAINING "
                         "simulator. Same schema as experiments/evaluate.py --xi, so a "
                         "condition can be used interchangeably as a training world or a "
                         "deployment target. Pass a FILE: PowerShell strips the quotes out "
                         "of inline JSON before python sees it.")
parser.add_argument("--interface", type=str, default="velocity", choices=("velocity", "force"),
                    help="which physical variable the policy commands. Changes ONLY the "
                         "action interface -- plant, servo lag, delay, clamp, speed limit, "
                         "observation, reward and PPO settings are untouched. See "
                         "experiments/interfaces.py.")
parser.add_argument("--dr", type=str, default="none", choices=("none", "box", "geom", "orbit", "transverse"),
                    help="link-mass domain randomisation (H5). 'box' is isotropic over all "
                         "three link masses; 'geom' spends the SAME expected budget "
                         "E||dm|| confined transverse to the uniform equivalence direction. "
                         "See PREREGISTRATION_H5.md.")
parser.add_argument("--dr_width", type=float, default=0.20,
                    help="half-width of the box arm. The geom arm's width is scaled up by "
                         "GEOM_WIDTH_FACTOR so the two budgets match.")
parser.add_argument("--video", action="store_true", help="record rollouts during training")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--vulkan", action="store_true",
                    help="use Vulkan instead of D3D12 (broken on this machine)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# ---------------------------------------------------------------------------
# Isaac Lab's experience file sets `vulkan = true` under [settings.app], forcing
# Vulkan on Windows where Kit otherwise defaults to D3D12. On this machine
# (RTX 5070 Ti Laptop / Blackwell, driver 596.13) the Vulkan path dies with an
# access violation inside omni.kit.viewport.window -- both for the GUI and for
# headless video capture. D3D12 works. Verified: with Vulkan the run aborts at
# viewport startup; with this flag it reaches "app ready" and renders.
#
# Override with --vulkan if a future driver fixes it.
if sys.platform == "win32" and not getattr(args_cli, "vulkan", False):
    extra = "--/app/vulkan=false"
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " " + extra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
from importlib import metadata  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401  registers the gym ids
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from interfaces import action_term, apply_interface  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    iface = apply_interface(env_cfg, args_cli.interface)
    print("[train] interface:", args_cli.interface, json.dumps(iface))

    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    # rsl-rl >= 4 uses a new model-config schema; this migrates the legacy
    # RslRlPpoActorCriticCfg into actor/critic model configs. Without it the
    # runner dies with KeyError: 'class_name'.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name

    log_root = os.path.abspath(os.path.join(ROOT, "logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += "_" + agent_cfg.run_name
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print("[train] task     :", args_cli.task)
    print("[train] num_envs :", env_cfg.scene.num_envs)
    print("[train] log_dir  :", log_dir)
    with open(os.path.join(log_dir, "interface.json"), "w", encoding="utf-8") as fh:
        json.dump(iface, fh, indent=1)

    # ---- xi: the TRAINING simulator ------------------------------------
    xi = {}
    if args_cli.xi:
        with open(args_cli.xi, encoding="utf-8") as fh:
            xi = json.load(fh)
        act = action_term(env_cfg)
        for k, attr in (("order", "order"), ("zeta", "zeta"), ("omega_n", "omega_n"),
                        ("tau", "time_constant_s"), ("delay_s", "delay_s"),
                        ("deadband", "deadband")):
            if k in xi:
                setattr(act, attr, int(xi[k]) if k == "order" else float(xi[k]))
        rob = env_cfg.scene.robot
        if "kv" in xi:
            rob.actuators["cart"].damping = float(xi["kv"])
        if "f_clamp" in xi:
            rob.actuators["cart"].effort_limit_sim = float(xi["f_clamp"])
        if "joint_damping" in xi:
            rob.actuators["passive"].damping = float(xi["joint_damping"])
        if "joint_friction" in xi:
            rob.actuators["passive"].friction = float(xi["joint_friction"])
        if "gravity" in xi:
            env_cfg.sim.gravity = (0.0, 0.0, -float(xi["gravity"]))
        print("[train] xi       :", json.dumps(xi))

    # ---- H5: link-mass domain randomisation, per environment ------------
    # Startup-mode randomisation: each of the N parallel environments is given
    # its own sampled link masses, fixed for the run. With 4096 environments
    # that samples the randomisation distribution densely. Applied through the
    # PhysX view after gym.make for the same reason xi is -- a dynamically
    # assigned EventTermCfg does not register, and would silently randomise
    # nothing while reporting a plausible number.
    GEOM_WIDTH_FACTOR = 1.478   # restores E||dm|| after the projection; see
                                # PREREGISTRATION_H5.md for the measurement

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if xi and any(k in xi for k in ("mass_scale", "cart_mass_scale")):
        # after gym.make: masses live in the USD asset, and a dynamically
        # assigned startup EventTermCfg does not register (verified -- the event
        # manager reported only ['reset'] and the perturbation silently did
        # nothing while reporting a plausible success rate).
        _rb = env.unwrapped.scene["robot"]
        _v = _rb.root_physx_view
        _names = list(_rb.body_names)
        _m, _I = _v.get_masses().clone(), _v.get_inertias().clone()
        _want = {"cart": float(xi.get("cart_mass_scale", 1.0))}
        for _i, _sc in enumerate(xi.get("mass_scale", [1.0, 1.0, 1.0])):
            _want["link%d" % (_i + 1)] = float(_sc)
        for _n, _sc in _want.items():
            if _n in _names and _sc != 1.0:
                _b = _names.index(_n)
                _m[:, _b] *= _sc      # inertia scaled by the same factor: a
                _I[:, _b] *= _sc      # density error at fixed geometry
        _idx = torch.arange(_v.count, dtype=torch.int32)
        _v.set_masses(_m, _idx)
        _v.set_inertias(_I, _idx)
        with open(os.path.join(log_dir, "xi_applied.json"), "w", encoding="utf-8") as fh:
            json.dump({"xi": xi, "interface": iface,
                       "masses_in_sim": {n: round(float(_v.get_masses()[0, i]), 6)
                                         for i, n in enumerate(_names)}}, fh, indent=1)

    if args_cli.dr != "none":
        # ---- H5: randomisation over a 7-D parameter block ------------------
        # Coordinates, in dimensionless relative units:
        #   [m1, m2, m3, I1, I2, I3, cart_mass]
        # Mass and inertia move INDEPENDENTLY here, unlike the xi conditions
        # elsewhere which scale them together as a density error. They have to:
        # the null structure couples them, and holding I = f(m) would collapse
        # the very directions this experiment is about.
        #
        # The geom arm projects off the null subspace DISCOVERED by
        # tools/geometry_discover.py, not off a hand-written direction. An
        # earlier version of this block projected off m0/||m0||, the
        # analytically known answer, which made the "geometry-aware" arm a
        # restatement of the theory rather than a test of a method.
        #
        # kv is NOT randomised even though the most-null discovered direction
        # is cart_mass <-> kv. Per-environment joint damping is writable
        # (articulation.write_joint_damping_to_sim) but the source notes it
        # does not update the actuator model's own bookkeeping, so the
        # telemetry and the physics would disagree. Excluded deliberately
        # rather than risked on an 8-hour unattended run; recorded here so the
        # omission is visible in the result.
        import numpy as _np
        _DRN = ["m1", "m2", "m3", "I1", "I2", "I3", "cart_mass"]
        _rb2 = env.unwrapped.scene["robot"]
        _v2 = _rb2.root_physx_view
        _names2 = list(_rb2.body_names)
        _li = [_names2.index("link%d" % (i + 1)) for i in range(3)]
        _ci = _names2.index("cart")
        _m2, _I2 = _v2.get_masses().clone(), _v2.get_inertias().clone()
        _n_env = _m2.shape[0]

        _null = _np.zeros((7, 0))
        # T5: the EXACT symmetry orbit. In log-coordinates the derived
        # generator scales every inertial parameter equally, so within this
        # 7-D block it is the all-ones direction. "orbit" randomises ONLY
        # along it; "transverse" randomises only in its complement. H5 used
        # the FITTED geometry and failed; this uses the proved one.
        if args_cli.dr in ("orbit", "transverse"):
            _u = _np.ones(7) / _np.sqrt(7.0)
            _rng = _np.random.default_rng(args_cli.seed)
            _w = args_cli.dr_width
            _d = _rng.uniform(-_w, _w, size=(_n_env, 7))
            _ref = float(_np.linalg.norm(_d, axis=1).mean())
            if args_cli.dr == "orbit":
                _d = _np.outer(_d @ _u, _u)          # keep ONLY the orbit part
            else:
                _d = _d - _np.outer(_d @ _u, _u)     # remove the orbit part
            _cur = float(_np.linalg.norm(_d, axis=1).mean())
            _d *= _ref / max(_cur, 1e-12)            # equal budget
            _scale = _np.exp(_d)                     # log-coordinates
            for _k2, _b2 in enumerate(_li):
                _m2[:, _b2] *= torch.tensor(_scale[:, _k2], dtype=_m2.dtype, device=_m2.device)
                _I2[:, _b2] *= torch.tensor(_scale[:, 3 + _k2],
                                            dtype=_I2.dtype, device=_I2.device).unsqueeze(-1)
            _m2[:, _ci] *= torch.tensor(_scale[:, 6], dtype=_m2.dtype, device=_m2.device)
            _v2.set_masses(_m2, torch.arange(_v2.count, dtype=torch.int32))
            _v2.set_inertias(_I2, torch.arange(_v2.count, dtype=torch.int32))
            print("[train] dr       : %s  E||dlog|| = %.6f  (ref %.6f)"
                  % (args_cli.dr, float(_np.linalg.norm(_d, axis=1).mean()), _ref))
            with open(os.path.join(log_dir, "dr_applied.json"), "w", encoding="utf-8") as fh:
                json.dump({"mode": args_cli.dr, "width": _w, "coords": _DRN,
                           "orbit_direction": _u.tolist(),
                           "E_norm_dlog": float(_np.linalg.norm(_d, axis=1).mean()),
                           "E_norm_reference": _ref, "n_envs": int(_n_env),
                           "scale_min": float(_scale.min()),
                           "scale_max": float(_scale.max())}, fh, indent=1)
        elif args_cli.dr == "geom":
            _gp = os.path.join(ROOT, "results", "geometry_discover.json")
            if not os.path.exists(_gp):
                raise SystemExit("[train] --dr geom needs results/geometry_discover.json; "
                                 "run tools/geometry_discover.py first")
            # Restricting the FULL null subspace to a coordinate subset does
            # NOT give a null subspace of the restricted system. Doing that
            # here returned "cart_mass alone", because the discovered null
            # direction is cart_mass TOGETHER WITH kv and kv is held fixed.
            #
            # The correct object is the metric of the restricted problem, and
            # because G = E[J^T J] that is exactly the corresponding submatrix
            # of G. Rebuild G from its spectrum, take the submatrix, and
            # eigendecompose that.
            _g = json.load(open(_gp, encoding="utf-8"))
            _V = _np.array(_g["eigenvectors"])
            _wv = _np.array(_g["eigenvalues"])
            _G = _V @ _np.diag(_wv) @ _V.T
            _rows = [_g["params"].index(n) for n in _DRN]
            _Gr = _G[_np.ix_(_rows, _rows)]
            _wr, _Vr = _np.linalg.eigh(_Gr)
            _o = _np.argsort(_wr)
            _wr, _Vr = _wr[_o], _Vr[:, _o]
            # rank 2: with kv fixed the null space is {cart_mass, uniform
            # inertial}, and there is a 12x eigenvalue gap after the first
            # with the analytic symmetry sitting at alignment 0.968 with the
            # first two. Taking 3 would start removing directions an order of
            # magnitude more sensitive, i.e. throwing away useful budget.
            _null = _Vr[:, :2]
            print("[train] dr null subspace (restricted metric): lambda = %s"
                  % ", ".join("%.3e" % x for x in _wr[:3]))
            for _j in range(_null.shape[1]):
                _t = _np.argsort(-_np.abs(_null[:, _j]))[:3]
                print("          n%d: %s" % (_j, ", ".join(
                    "%s %+.2f" % (_DRN[k], _null[k, _j]) for k in _t)))

        # NOTE: this block is for the box and geom arms ONLY. The orbit and
        # transverse arms above apply their own perturbation and return. An
        # earlier version let them fall through to here, which multiplied a
        # fresh isotropic draw on top of the arm's own perturbation and then
        # overwrote dr_applied.json -- so the transverse arm received a
        # strictly larger total displacement than the orbit arm, which is
        # exactly the confound the equal-budget design exists to remove.
        if args_cli.dr in ("orbit", "transverse"):
            pass                                   # already applied above
        else:
          _rng = _np.random.default_rng(args_cli.seed)
          _w = args_cli.dr_width
          _d = _rng.uniform(-_w, _w, size=(_n_env, 7))
          _box_budget = float(_np.linalg.norm(_d, axis=1).mean())
          if args_cli.dr == "geom" and _null.shape[1]:
              _d = _d - (_d @ _null) @ _null.T        # remove the null component
              # projection removes budget; rescale so the two arms spend the same
              _cur = float(_np.linalg.norm(_d, axis=1).mean())
              _d *= _box_budget / max(_cur, 1e-12)
          _budget = float(_np.linalg.norm(_d, axis=1).mean())
          _leak = (float(_np.abs(_d @ _null).mean()) if _null.shape[1] else float("nan"))

          _scale = 1.0 + _d
          if float(_scale.min()) <= 0.0:
              raise SystemExit("[train] dr width %.3f produces a non-positive parameter" % _w)
          for _k, _b in enumerate(_li):
              _m2[:, _b] *= torch.tensor(_scale[:, _k], dtype=_m2.dtype, device=_m2.device)
              _I2[:, _b] *= torch.tensor(_scale[:, 3 + _k],
                                         dtype=_I2.dtype, device=_I2.device).unsqueeze(-1)
          _m2[:, _ci] *= torch.tensor(_scale[:, 6], dtype=_m2.dtype, device=_m2.device)
          _v2.set_masses(_m2, torch.arange(_v2.count, dtype=torch.int32))
          _v2.set_inertias(_I2, torch.arange(_v2.count, dtype=torch.int32))

          print("[train] dr       : %s width %.4f  E||dtheta|| = %.6f "
                "(box reference %.6f)  leakage %.3e"
                % (args_cli.dr, _w, _budget, _box_budget, _leak))
          with open(os.path.join(log_dir, "dr_applied.json"), "w", encoding="utf-8") as fh:
              json.dump({"mode": args_cli.dr, "width": _w,
                         "coords": _DRN,
                         "null_subspace_rank": int(_null.shape[1]),
                         "null_subspace": _null.tolist(),
                         "E_norm_dtheta": _budget,
                         "E_norm_dtheta_box_reference": _box_budget,
                         "E_abs_leakage_into_null": _leak,
                         "kv_randomised": False,
                         "n_envs": int(_n_env),
                         "scale_min": float(_scale.min()),
                         "scale_max": float(_scale.max())}, fh, indent=1)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    t0 = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("[train] wall time: %.1f s" % (time.time() - t0))

    # a stable pointer for downstream scripts, independent of the timestamp
    with open(os.path.join(log_root, "LATEST"), "w", encoding="utf-8") as fh:
        fh.write(log_dir)
    env.close()


def _hard_exit(code: int = 0) -> None:
    """Terminate immediately after Kit shutdown.

    Isaac Sim frequently hangs inside simulation_app.close() on Windows and
    leaves a python.exe spinning at 100% CPU forever. Three such orphans from
    one night's runs were burning ~57,000 CPU-seconds each and starving a
    training job. Results are already on disk by this point, so flush and use
    os._exit to skip the wedged interpreter teardown.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
    simulation_app.close()
    _hard_exit(0)
