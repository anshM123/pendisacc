"""What states does the stuck policy ACTUALLY visit, and does the capture term
have any support there?

The swing-up plateaus at reward ~67 with capture ~0.0006: it pumps the tip up
through the top and never arrests. Widening the capture Gaussians was supposed
to give it a gradient in, and barely moved the number. Rather than guess at
widths again, measure the (tilt, speed) pairs the policy visits and ask what
the capture term is worth there -- and what widths would make it non-trivial.

  run.cmd tools\\diag_capture.py --checkpoint <file.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
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
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    dev = env.unwrapped.device
    policy = runner.get_inference_policy(device=dev)

    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    idx = [names.index(j) for j in ("joint1", "joint2", "joint3")]
    steps = int(env.unwrapped.max_episode_length)

    with torch.inference_mode():
        env.unwrapped.reset()
        obs = env.get_observations()
        tilts, speeds = [], []
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            th = torch.cumsum(robot.data.joint_pos[:, idx], dim=-1)
            w = torch.cumsum(robot.data.joint_vel[:, idx], dim=-1)
            tilts.append(torch.sum(1.0 - torch.cos(th), dim=-1).cpu().numpy())
            speeds.append(torch.sum(w * w, dim=-1).cpu().numpy())
    tilt = np.concatenate(tilts)      # 0 upright, 6 hanging
    speed = np.concatenate(speeds)    # sum of squared absolute rates

    # Kit owns stdout once it is up, so printed output is swallowed. Write a
    # file, the same reason evaluate.py does.
    out = {
        "checkpoint": os.path.basename(args_cli.checkpoint),
        "steps": int(steps), "num_envs": int(args_cli.num_envs),
        "tilt": {"min": float(tilt.min()), "p1": float(np.percentile(tilt, 1)),
                 "p10": float(np.percentile(tilt, 10)), "median": float(np.median(tilt))},
        "speed": {"min": float(speed.min()), "p1": float(np.percentile(speed, 1)),
                  "p10": float(np.percentile(speed, 10)), "median": float(np.median(speed))},
        "near_upright": {}, "candidate_widths": [],
    }
    for thr in (0.5, 1.0, 2.0):
        m = tilt < thr
        frac = float(m.mean())
        out["near_upright"]["tilt_lt_%.1f" % thr] = {
            "fraction_of_steps": frac,
            "speed_median": float(np.median(speed[m])) if frac else None,
            "speed_p10": float(np.percentile(speed[m], 10)) if frac else None,
            "speed_min": float(speed[m].min()) if frac else None,
        }
    for a, v in ((0.25, 3.0), (1.0, 6.0), (1.5, 12.0), (2.0, 20.0), (3.0, 30.0), (6.0, 60.0)):
        c = np.exp(-tilt / a) * np.exp(-speed / (v * v))
        out["candidate_widths"].append({
            "angle_std": a, "vel_std": v, "mean": float(c.mean()),
            "p99": float(np.percentile(c, 99)), "max": float(c.max())})
    dst = os.path.join(ROOT, "results", "diag_capture.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)
