"""
Score checkpoints by MEASURED success rate, not by training reward.

Reward is a bad selector here and has misled us twice: one swing-up run climbed
to reward 58 while never lifting the tip above -0.334 (it was farming easy
near-upright starts), and another peaked at 141 then collapsed to -1255 without
the checkpoint file changing. So define success physically and count it.

A swing-up episode SUCCEEDS when all three hold:
  1. it never terminated early (no rail excursion, no fall)
  2. the tip reached upright at some point
  3. the tip is STILL upright over the final quarter of the episode

That is "got up and stayed up", which is what "works 100% of the time" has to
mean. Every environment starts from dead hang, so there are no easy episodes.

  run.cmd experiments\\evaluate.py --run logs\\rsl_rl\\tip_swingup\\<dir> --stride 200
  run.cmd experiments\\evaluate.py --checkpoint <file.pt> --num_envs 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure policy success rate.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--experiment", type=str, default="tip_swingup")
parser.add_argument("--run", type=str, default=None, help="run dir; evaluates its checkpoints")
parser.add_argument("--checkpoint", type=str, default=None, help="single checkpoint")
parser.add_argument("--run_list", type=str, default=None,
                    help="text file of run dirs; evaluates the FINAL checkpoint of each "
                         "in one process (hidden-suite evaluation, PREREGISTRATION_PI.md)")
parser.add_argument("--stride", type=int, default=100, help="evaluate every Nth checkpoint")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--upright", type=float, default=0.9, help="tip height counting as upright")
parser.add_argument("--hold_frac", type=float, default=0.25, help="final fraction that must stay up")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--servo_tau", type=float, default=None,
                    help="override the actuator time constant [s] -- an xi perturbation")
parser.add_argument("--xi", type=str, default=None,
                    help="simulator perturbations: a path to a JSON file, or an inline JSON "
                         "dict. Prefer the FILE form -- PowerShell strips the quotes out of "
                         "an inline JSON argument before python sees it, which fails as a "
                         "decode error rather than as anything obvious.")
parser.add_argument("--servo_order", type=int, default=None, help="1 or 2")
parser.add_argument("--servo_wn", type=float, default=None, help="rad/s, second order")
parser.add_argument("--servo_zeta", type=float, default=0.9)
parser.add_argument("--perturb", action="store_true",
                    help="randomise the initial condition instead of evaluating a near-fixed one")
parser.add_argument("--angle_sigma", type=float, default=0.10, help="rad, half-width on each link angle")
parser.add_argument("--rate_sigma", type=float, default=0.20, help="rad/s, half-width on each link rate")
parser.add_argument("--cart_sigma", type=float, default=0.10, help="m, half-width on cart position")
parser.add_argument("--cart_rate_sigma", type=float, default=0.10, help="m/s, half-width on cart velocity")
parser.add_argument("--obs_noise", type=float, default=0.0,
                    help="std of Gaussian sensor noise added to every observation")
parser.add_argument("--interface", type=str, default="velocity", choices=("velocity", "force"),
                    help="must match the interface the checkpoint was TRAINED with; a policy "
                         "cannot be evaluated through an interface it never spoke")
parser.add_argument("--vulkan", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if sys.platform == "win32" and not args_cli.vulkan:
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/vulkan=false"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

APPLIED: dict = {}   # what actually reached the sim, written to the output JSON
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from interfaces import action_term, apply_interface  # noqa: E402
from dynamics.conventions import rel_to_abs  # noqa: E402


def link_lengths():
    with open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    return [float(p["bodies"][n].get("length", 2.0 * p["bodies"][n]["l_com"]))
            for n in ("link1", "link2", "link3")]


def checkpoints() -> list[str]:
    if args_cli.checkpoint:
        return [args_cli.checkpoint]
    if args_cli.run_list:
        cks = []
        for line in open(args_cli.run_list, encoding="utf-8"):
            run = line.strip()
            if not run:
                continue
            fs = [f for f in os.listdir(run) if f.startswith("model_") and f.endswith(".pt")]
            fs.sort(key=lambda f: int("".join(c for c in f if c.isdigit()) or 0))
            cks.append(os.path.join(run, fs[-1]))
        return cks
    run = args_cli.run
    if run is None:
        root = os.path.join(ROOT, "logs", "rsl_rl", args_cli.experiment)
        run = os.path.join(root, sorted(os.listdir(root))[-1])
    files = [f for f in os.listdir(run) if f.startswith("model_") and f.endswith(".pt")]
    files.sort(key=lambda f: int("".join(c for c in f if c.isdigit()) or 0))
    keep = [f for f in files if int("".join(c for c in f if c.isdigit()) or 0) % args_cli.stride == 0]
    if files and files[-1] not in keep:
        keep.append(files[-1])
    return [os.path.join(run, f) for f in keep]


def main() -> int:
    L = torch.tensor(link_lengths())
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    APPLIED["interface"] = apply_interface(env_cfg, args_cli.interface)

    # A near-fixed initial condition makes N parallel environments N replicas of
    # one trajectory, not N independent trials, so a binomial confidence
    # interval over them is not meaningful. --perturb turns the evaluation into
    # a genuine distribution: independent uniform draws on every initial state,
    # plus optional Gaussian sensor noise on the observation the policy sees.
    if args_cli.perturb:
        ev = env_cfg.events
        a, w = args_cli.angle_sigma, args_cli.rate_sigma
        ev.reset_link1.params["position_range"] = (math.pi - a, math.pi + a)
        ev.reset_link1.params["velocity_range"] = (-w, w)
        ev.reset_links23.params["position_range"] = (-a, a)
        ev.reset_links23.params["velocity_range"] = (-w, w)
        ev.reset_cart.params["position_range"] = (-args_cli.cart_sigma, args_cli.cart_sigma)
        ev.reset_cart.params["velocity_range"] = (-args_cli.cart_rate_sigma, args_cli.cart_rate_sigma)
    if args_cli.servo_tau is not None:
        # cross-check point for the standalone analysis: does Isaac agree with
        # dynamics/closed_loop.py about an actuator the policy never saw?
        action_term(env_cfg).time_constant_s = float(args_cli.servo_tau)
    # ---- xi: one JSON blob so a suite entry maps to exactly one Isaac run ----
    if not args_cli.xi:
        xi = {}
    elif args_cli.xi.strip().startswith("{"):
        xi = json.loads(args_cli.xi)
    else:
        with open(args_cli.xi, encoding="utf-8") as _fh:
            xi = json.load(_fh)
    if xi:
        act = action_term(env_cfg)
        for k, attr in (("order", "order"), ("zeta", "zeta"), ("omega_n", "omega_n"),
                        ("tau", "time_constant_s"), ("delay_s", "delay_s"),
                        ("deadband", "deadband")):
            if k in xi:
                setattr(act, attr, float(xi[k]) if k != "order" else int(xi[k]))
        rob = env_cfg.scene.robot
        if "kv" in xi:
            rob.actuators["cart"].damping = float(xi["kv"])
        if "f_clamp" in xi:
            # ImplicitActuatorCfg refuses effort_limit when effort_limit_sim is
            # already set, and the asset sets the latter.
            rob.actuators["cart"].effort_limit_sim = float(xi["f_clamp"])
        if "joint_damping" in xi:
            rob.actuators["passive"].damping = float(xi["joint_damping"])
        if "joint_friction" in xi:
            rob.actuators["passive"].friction = float(xi["joint_friction"])
        if "gravity" in xi:
            env_cfg.sim.gravity = (0.0, 0.0, -float(xi["gravity"]))
        # NOTE: mass scaling is applied after gym.make, not as an event term.
        # Assigning a new EventTermCfg onto env_cfg.events after the config is
        # instantiated does NOT register it -- the event manager reported only
        # ['reset'] and the perturbation silently did nothing, which reads as
        # "this xi does not matter" rather than as an error.
    if args_cli.servo_order is not None:
        _a = action_term(env_cfg)
        _a.order = int(args_cli.servo_order)
        _a.zeta = float(args_cli.servo_zeta)
        if args_cli.servo_wn is not None:
            _a.omega_n = float(args_cli.servo_wn)
    if args_cli.obs_noise > 0.0:
        from isaaclab.utils.noise import GaussianNoiseCfg
        n = GaussianNoiseCfg(mean=0.0, std=args_cli.obs_noise, operation="add")
        pol = env_cfg.observations.policy
        for name in ("cart", "link_sincos", "link_vel", "last_action"):
            getattr(pol, name).noise = n
        pol.enable_corruption = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    if xi and any(k in xi for k in ("mass_scale", "cart_mass_scale")):
        _robot = env.unwrapped.scene["robot"]
        _view = _robot.root_physx_view
        _names = list(_robot.body_names)
        _m, _I = _view.get_masses().clone(), _view.get_inertias().clone()
        _want = {"cart": float(xi.get("cart_mass_scale", 1.0))}
        for _i, _s in enumerate(xi.get("mass_scale", [1.0, 1.0, 1.0])):
            _want["link%d" % (_i + 1)] = float(_s)
        for _n, _sc in _want.items():
            if _n in _names and _sc != 1.0:
                _b = _names.index(_n)
                _m[:, _b] *= _sc          # inertia scaled by the same factor:
                _I[:, _b] *= _sc          # a density error at fixed geometry
        _idx = torch.arange(_view.count, dtype=torch.int32)
        _view.set_masses(_m, _idx)
        _view.set_inertias(_I, _idx)
        APPLIED["masses"] = {n: round(float(_view.get_masses()[0, i]), 6)
                             for i, n in enumerate(_names)}
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    link_idx = [names.index(j) for j in ("joint1", "joint2", "joint3")]
    dev = env.unwrapped.device
    Ld = L.to(dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

    results = []
    for ck in checkpoints():
        runner.load(ck)
        policy = runner.get_inference_policy(device=dev)

        # Everything below runs under inference_mode. Accumulators must be
        # created inside it too: mixing inference tensors with ordinary ones
        # raises "Inplace update to inference tensor outside InferenceMode".
        with torch.inference_mode():
            env.unwrapped.reset()
            obs = env.get_observations()

            n = env.num_envs
            terminated_early = torch.zeros(n, dtype=torch.bool, device=dev)
            ever_up = torch.zeros(n, dtype=torch.bool, device=dev)
            held = torch.zeros(n, device=dev)
            held_n = 0
            tip_sum = torch.zeros(n, device=dev)

            for k in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                th = torch.cumsum(robot.data.joint_pos[:, link_idx], dim=-1)
                tip = (Ld * torch.cos(th)).sum(dim=-1) / Ld.sum()
                tip_sum += tip
                ever_up |= tip > args_cli.upright
                if k >= hold_from:
                    held += (tip > args_cli.upright).float()
                    held_n += 1
                # a done before the final step is a failure, not a timeout
                if k < steps - 1:
                    terminated_early |= dones.bool()

            hold_frac = held / max(held_n, 1)
            success = (~terminated_early) & ever_up & (hold_frac > 0.95)
            rate = float(success.float().mean())
            early = float(terminated_early.float().mean())
            everup = float(ever_up.float().mean())
            holdm = float(hold_frac.mean())
            tipm = float((tip_sum / steps).mean())

        results.append({
            "checkpoint": os.path.basename(ck),
            "run": os.path.basename(os.path.dirname(ck)),
            "success_rate": rate,
            "early_termination_rate": early,
            "ever_reached_upright": everup,
            "mean_hold_fraction": holdm,
            "mean_tip_height": tipm,
        })
        print("  %-16s success %6.1f%%   early-term %5.1f%%   ever-up %5.1f%%   mean tip %+.3f"
              % (os.path.basename(ck), 100 * rate, 100 * early, 100 * everup, tipm))

    results.sort(key=lambda r: (-r["success_rate"], r["early_termination_rate"]))
    best = results[0]
    print("\nBEST: %s  success %.1f%% over %d episodes from dead hang"
          % (best["checkpoint"], 100 * best["success_rate"], args_cli.num_envs))

    out = args_cli.out or os.path.join(ROOT, "results", "eval_%s.json" % args_cli.experiment)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"task": args_cli.task, "num_envs": args_cli.num_envs,
                   "perturbed": bool(args_cli.perturb),
                   "init_ranges": ({"link_angle_rad": args_cli.angle_sigma,
                                    "link_rate_rad_s": args_cli.rate_sigma,
                                    "cart_pos_m": args_cli.cart_sigma,
                                    "cart_vel_m_s": args_cli.cart_rate_sigma}
                                   if args_cli.perturb else "PLAY default (near-fixed dead hang)"),
                   "obs_noise_std": args_cli.obs_noise,
                   "servo_tau_override": args_cli.servo_tau,
                   "servo_order": args_cli.servo_order, "xi": xi,
                   "applied": APPLIED,
                   "servo_wn": args_cli.servo_wn, "servo_zeta": args_cli.servo_zeta,
                   "upright_threshold": args_cli.upright, "results": results}, fh, indent=2)
    print("[out]", out)
    env.close()
    return 0


def _hard_exit(code: int = 0) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    _hard_exit(code)
