"""
Watch a trained policy, or record it to mp4.

  # live GUI window
  python experiments/play.py

  # record video instead of opening a window
  python experiments/play.py --headless --video --enable_cameras

By default it loads the newest checkpoint of the newest run for the task.
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

parser = argparse.ArgumentParser(description="Play a trained triple-pendulum policy.")
parser.add_argument("--task", type=str, default="TIP-Balance-Play-v0")
parser.add_argument("--experiment", type=str, default="tip_balance")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="explicit .pt path")
parser.add_argument("--run", type=str, default=None, help="run directory name; default = newest")
parser.add_argument("--steps", type=int, default=0, help="0 = run until the window is closed")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=600)
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

import gymnasium as gym  # noqa: E402
from importlib import metadata  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401


def newest_checkpoint(experiment: str, run: str | None) -> str:
    root = os.path.join(ROOT, "logs", "rsl_rl", experiment)
    if not os.path.isdir(root):
        raise SystemExit("no logs at %s -- train something first" % root)
    if run is None:
        runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if not runs:
            raise SystemExit("no runs under %s" % root)
        run = runs[-1]
    run_dir = os.path.join(root, run)
    ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt"))
    if not ckpts:
        raise SystemExit("no checkpoints in %s" % run_dir)
    ckpts.sort(key=lambda f: int("".join(c for c in f if c.isdigit()) or 0))
    return os.path.join(run_dir, ckpts[-1])


def main() -> None:
    ckpt = args_cli.checkpoint or newest_checkpoint(args_cli.experiment, args_cli.run)
    print("[play] checkpoint:", ckpt)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    # rsl-rl >= 4 uses a new model-config schema; this migrates the legacy
    # RslRlPpoActorCriticCfg into actor/critic model configs. Without it the
    # runner dies with KeyError: 'class_name'.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(ckpt), "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # NOTE: RslRlVecEnvWrapper.get_observations() returns a TensorDict, not a
    # (obs, extras) tuple. Unpacking it iterates the keys instead.
    obs = env.get_observations()

    step = 0
    episodes = 0
    survived = 0.0
    alive_for = torch.zeros(env.num_envs, device=env.unwrapped.device)
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        alive_for += dt
        done_mask = dones.bool()
        if done_mask.any():
            survived += float(alive_for[done_mask].sum())
            episodes += int(done_mask.sum())
            alive_for[done_mask] = 0.0
        step += 1
        if args_cli.steps and step >= args_cli.steps:
            break
        if args_cli.video and step >= args_cli.video_length + 2:
            break

    still_up = int((alive_for > 0).sum())
    summary = {
        "checkpoint": ckpt,
        "num_envs": int(env.num_envs),
        "steps": step,
        "sim_seconds": step * dt,
        "episodes_ended": episodes,
        "mean_survival_s": (survived / episodes) if episodes else None,
        "episode_cap_s": float(env.unwrapped.max_episode_length_s),
        "still_balancing": still_up,
    }
    print("[play]", summary)
    # Kit takes over stdout after launch, so console output is not a reliable
    # record -- write the numbers to disk as well.
    import json
    out = os.path.join(ROOT, "results", "play_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
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
