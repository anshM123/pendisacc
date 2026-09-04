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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

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

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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


if __name__ == "__main__":
    main()
    simulation_app.close()
