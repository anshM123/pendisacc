"""Finite-time separation growth for parameter-close twins (runs only if the fractal gate locks in).

For pairs of simulators differing by eps in log(m1) (same frozen policy, same
fixed initial condition), record the full closed-loop joint state of both every
control step and store the separation d(t) = |x_a(t) - x_b(t)| (angles wrapped).
Exponential growth of d(t) at a common rate across eps, with curves offset by
log eps, is the signature of sensitive dependence; eps = 0 twins measure how
much separation the simulator's own nondeterminism produces.

  run.cmd experiments/fractal_ftle.py --checkpoint <pt> --out results/fractal/ftle.npz
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TIP-SwingUp-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--pairs", type=int, default=256)
parser.add_argument("--eps", default="0,0.00001,0.0001,0.001")
parser.add_argument("--half", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=31)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if sys.platform == "win32":
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/vulkan=false"
simulation_app = AppLauncher(args_cli).app

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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
IC_LINK1, IC_LINK23 = math.pi + 0.03, 0.01   # same fixed IC as experiments/fractal_map.py (importing it would re-run its CLI)
from interfaces import apply_interface  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(args_cli.seed)
    eps = [float(e) for e in args_cli.eps.split(",")]
    P = args_cli.pairs
    rows = []
    for e in eps:
        base = rng.uniform(-args_cli.half, args_cli.half, (P, 2))
        other = base.copy()
        other[:, 0] += e
        rows += [base, other]
    X = np.concatenate(rows, 0)
    n = len(X)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    apply_interface(env_cfg, "velocity")
    ev = env_cfg.events
    for term, pos in (("reset_link1", IC_LINK1), ("reset_links23", IC_LINK23), ("reset_cart", 0.0)):
        getattr(ev, term).params["position_range"] = (pos, pos)
        getattr(ev, term).params["velocity_range"] = (0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    m, I = view.get_masses().clone(), view.get_inertias().clone()
    for name, col in (("link1", 0), ("link3", 1)):
        b = names.index(name)
        s = torch.tensor(np.exp(X[:, col]), dtype=m.dtype)
        m[:, b] *= s
        I[:, b] *= s.unsqueeze(-1)
    idx = torch.arange(view.count, dtype=torch.int32)
    view.set_masses(m, idx)
    view.set_inertias(I, idx)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    steps = int(env.unwrapped.max_episode_length)
    D = torch.zeros((steps, len(eps), P), device=env.unwrapped.device)
    with torch.inference_mode():
        env.unwrapped.reset()
        obs = env.get_observations()
        for k in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            q = robot.data.joint_pos.clone()
            qd = robot.data.joint_vel
            q[:, 1:] = torch.atan2(torch.sin(q[:, 1:]), torch.cos(q[:, 1:]))
            x = torch.cat([q, 0.1 * qd], -1)            # rates scaled to comparable units
            for i in range(len(eps)):
                a, b = x[2 * i * P: 2 * i * P + P], x[2 * i * P + P: 2 * (i + 1) * P]
                dq = a - b
                dq[:, 1:4] = torch.atan2(torch.sin(dq[:, 1:4]), torch.cos(dq[:, 1:4]))
                D[k, i] = dq.norm(dim=-1)
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(args_cli.out, D=D.cpu().numpy().astype(np.float32), eps=np.array(eps),
                        base=X[0::2 * P][:0], dt=1 / 250)
    print("[ftle] %d twins x %d eps x %d steps -> %s" % (P, len(eps), steps, args_cli.out))
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
