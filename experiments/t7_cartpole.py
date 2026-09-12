"""T7: does the derived generator hold on a DIFFERENT robot, experimentally?

Section VIII of the paper solves the symmetry group for other robot classes
symbolically. A reviewer is entitled to ask whether any of that survives
contact with a simulator. This tests it on Isaac Lab's own stock
`Isaac-Cartpole-v0` -- a different robot, a different asset, a different
codebase, and a force interface rather than a velocity loop.

For that system the solver returned

    m_pole, m_cart, I_pole, u_force      all scaled by c

so the action scale must move with the masses. The effort limit rides along
for the same reason the force clamp does on the pendulum: a saturation is not
a term in the equations, but a symmetry that demands more force than the
actuator can deliver is not realisable.

Controls are the point. Scaling the masses WITHOUT the action, or the action
WITHOUT the masses, must fail -- otherwise the invariance is trivial.

  run.cmd experiments/t7_cartpole.py --iterations 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 4.0, 16.0, 64.0])
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
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  registers Isaac-Cartpole-v0
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    base_scale = float(env_cfg.actions.joint_effort.scale)

    # Isaac Lab's own tasks register this entry point as a STRING module path,
    # whereas the tasks in this repository register a callable. The registry
    # loader handles both; calling the spec value directly does not.
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    agent_cfg.max_iterations = args_cli.iterations

    log_dir = os.path.join(ROOT, "logs", "rsl_rl", "t7_cartpole")
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    M0, I0 = view.get_masses().clone(), view.get_inertias().clone()
    idx = torch.arange(view.count, dtype=torch.int32)
    jn = list(robot.joint_names)
    slider = jn.index("slider_to_cart")
    E0 = robot.data.joint_effort_limits[0, slider].item() \
        if hasattr(robot.data, "joint_effort_limits") else None

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    print("[t7] training cartpole for %d iterations" % args_cli.iterations)
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dev = env.unwrapped.device
    steps = int(env.unwrapped.max_episode_length)
    act_term = env.unwrapped.action_manager.get_term("joint_effort")

    def apply(c_mass, c_act):
        m, I = M0.clone(), I0.clone()
        m *= c_mass
        I *= c_mass
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        act_term.cfg.scale = base_scale * c_act
        act_term._scale = base_scale * c_act        # the cached value the term uses
        if E0 is not None:
            robot.write_joint_effort_limit_to_sim(
                torch.full((robot.num_instances, 1), E0 * c_act, device=dev),
                joint_ids=[slider])

    def rollout():
        """Success = survived the whole episode without early termination."""
        with torch.inference_mode():
            env.unwrapped.reset()
            obs = env.get_observations()
            early = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            for k in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                if k < steps - 1:
                    early |= dones.bool()
            return float((~early).float().mean())

    rows = []
    print("\nT7: the derived generator on Isaac Lab's stock cartpole")
    print("generator = (m_pole, m_cart, I_pole, u_force) scaled together\n")
    print("  %-34s %8s" % ("condition", "survive"))
    print("  " + "-" * 46)
    for c in args_cli.c_values:
        apply(c, c)
        r = rollout()
        rows.append({"name": "generator_c%g" % c, "c_mass": c, "c_act": c, "success": r})
        print("  %-34s %7.1f%%" % ("GENERATOR  c=%g" % c, 100 * r))
    for c in args_cli.c_values[1:]:
        apply(c, 1.0)
        r = rollout()
        rows.append({"name": "mass_only_c%g" % c, "c_mass": c, "c_act": 1.0, "success": r})
        print("  %-34s %7.1f%%" % ("mass only  c=%g" % c, 100 * r))
    for c in args_cli.c_values[1:]:
        apply(1.0, c)
        r = rollout()
        rows.append({"name": "action_only_c%g" % c, "c_mass": 1.0, "c_act": c, "success": r})
        print("  %-34s %7.1f%%" % ("action only c=%g" % c, 100 * r))

    gen = [r["success"] for r in rows if r["name"].startswith("generator")]
    mo = [r["success"] for r in rows if r["name"].startswith("mass_only")]
    ao = [r["success"] for r in rows if r["name"].startswith("action_only")]
    print("\n  generator, worst over c : %.1f%%" % (100 * min(gen)))
    print("  mass only,  worst       : %.1f%%" % (100 * min(mo)))
    print("  action only, worst      : %.1f%%" % (100 * min(ao)))
    ok = min(gen) >= 0.90 and (min(mo) < 0.5 or min(ao) < 0.5)
    print("\n  generator holds and at least one control fails -> %s"
          % ("SUPPORTED" if ok else "FAIL"))

    out = args_cli.out or os.path.join(ROOT, "results", "T7", "cartpole.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"task": args_cli.task, "iterations": args_cli.iterations,
               "base_action_scale": base_scale, "rows": rows,
               "supported": bool(ok)}, open(out, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % out)
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
    main()
    _hard_exit(0)
