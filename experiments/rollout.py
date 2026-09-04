"""
Run the pendulum HEADLESS and record the joint trajectory to disk.

Isaac's renderer is unreliable on this machine (BAR1 exhaustion with many GPU
apps open -> access violation in GUI init and in the hydra engine), so the
viewing path is: record here, animate with tools/animate.py. Physics is
unaffected; only rendering is.

  # passive drop
  run.cmd experiments\\rollout.py --mode passive --seconds 6

  # a trained policy
  run.cmd experiments\\rollout.py --mode policy --seconds 10 ^
      --checkpoint logs\\rsl_rl\\tip_balance\\<run>\\model_150.pt

Writes results/rollout_<mode>.npz with absolute link angles, so it plugs
straight into the analytical model and the animator.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Headless rollout recorder.")
parser.add_argument("--mode", choices=["passive", "policy"], default="passive")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--task", type=str, default="TIP-Balance-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--tilt", type=float, default=0.12, help="passive: initial link1 tilt [rad]")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True          # never open a window; the renderer is the broken part

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
from dynamics.conventions import rel_to_abs  # noqa: E402


def record_passive() -> dict:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from triple_project.assets import CART_JOINT, TRIPLE_PENDULUM_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dt = 1.0 / 500.0
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args_cli.device, gravity=(0.0, 0.0, -9.80665))
    )
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    names = list(robot.joint_names)
    idx = [names.index(CART_JOINT)] + [names.index(j) for j in ("joint1", "joint2", "joint3")]

    jp = robot.data.default_joint_pos.clone()
    jp[:, names.index("joint1")] = args_cli.tilt
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.reset()

    n = int(args_cli.seconds / dt)
    effort = torch.zeros((1, len(names)), device=robot.device)
    qs = []
    for _ in range(n):
        robot.set_joint_effort_target(effort)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        qs.append(robot.data.joint_pos[0, idx].cpu().numpy().copy())
    return {"t": np.arange(1, n + 1) * dt, "q": rel_to_abs(np.asarray(qs)), "dt": dt}


def record_policy() -> dict:
    from importlib import metadata

    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg

    import triple_project.tasks  # noqa: F401

    if not args_cli.checkpoint:
        raise SystemExit("--checkpoint is required for --mode policy")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    idx = [names.index("cart_slide")] + [names.index(j) for j in ("joint1", "joint2", "joint3")]

    dt = env.unwrapped.step_dt
    n = int(args_cli.seconds / dt)
    obs = env.get_observations()
    qs = []
    for _ in range(n):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        qs.append(robot.data.joint_pos[0, idx].cpu().numpy().copy())
    env.close()
    return {"t": np.arange(1, n + 1) * dt, "q": rel_to_abs(np.asarray(qs)), "dt": dt}


def main() -> int:
    data = record_passive() if args_cli.mode == "passive" else record_policy()
    out = args_cli.out or os.path.join(ROOT, "results", "rollout_%s.npz" % args_cli.mode)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, mode=args_cli.mode, **data)
    print("[rollout] wrote", out, "frames:", len(data["t"]))
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
