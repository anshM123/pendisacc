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
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure policy success rate.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--experiment", type=str, default="tip_swingup")
parser.add_argument("--run", type=str, default=None, help="run dir; evaluates its checkpoints")
parser.add_argument("--checkpoint", type=str, default=None, help="single checkpoint")
parser.add_argument("--stride", type=int, default=100, help="evaluate every Nth checkpoint")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--upright", type=float, default=0.9, help="tip height counting as upright")
parser.add_argument("--hold_frac", type=float, default=0.25, help="final fraction that must stay up")
parser.add_argument("--out", type=str, default=None)
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401
from dynamics.conventions import rel_to_abs  # noqa: E402


def link_lengths():
    with open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    return [float(p["bodies"][n].get("length", 2.0 * p["bodies"][n]["l_com"]))
            for n in ("link1", "link2", "link3")]


def checkpoints() -> list[str]:
    if args_cli.checkpoint:
        return [args_cli.checkpoint]
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

    env = gym.make(args_cli.task, cfg=env_cfg)
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
