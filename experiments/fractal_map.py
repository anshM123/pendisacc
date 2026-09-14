"""Transfer map of a FROZEN policy over a 2-D simulator-parameter plane (PREREGISTRATION_FRACTAL.md).

Every environment gets its own (x, y) = (log m1 scale, log m3 scale); all start
from the SAME fixed initial condition, so success is a function of parameters
only (up to simulator nondeterminism, which is measured with duplicates).

  run.cmd experiments/fractal_map.py --checkpoint <pt> --mode grid  --n 64 --out results/fractal/grid.json
  run.cmd experiments/fractal_map.py --checkpoint <pt> --mode pairs --pairs 600 --out results/fractal/pairs.json
  zoom: add --cx X --cy Y --half 0.06
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TIP-SwingUp-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", choices=("grid", "pairs"), required=True)
parser.add_argument("--n", type=int, default=64)
parser.add_argument("--dups", type=int, default=128)
parser.add_argument("--pairs", type=int, default=600)
parser.add_argument("--eps", default="0.3,0.1,0.03,0.01,0.003,0.001")
parser.add_argument("--cx", type=float, default=0.0)
parser.add_argument("--cy", type=float, default=0.0)
parser.add_argument("--half", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=20260914)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if sys.platform == "win32":
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from interfaces import apply_interface  # noqa: E402

IC_LINK1 = math.pi + 0.03
IC_LINK23 = 0.01


def link_lengths():
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8"))
    return [float(p["bodies"][n].get("length", 2.0 * p["bodies"][n]["l_com"])) for n in ("link1", "link2", "link3")]


def layout():
    rng = np.random.default_rng(args_cli.seed)
    lo_x, hi_x = args_cli.cx - args_cli.half, args_cli.cx + args_cli.half
    lo_y, hi_y = args_cli.cy - args_cli.half, args_cli.cy + args_cli.half
    meta = {}
    if args_cli.mode == "grid":
        xs, ys = np.linspace(lo_x, hi_x, args_cli.n), np.linspace(lo_y, hi_y, args_cli.n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel()], 1)
        dup_src = rng.choice(len(P), size=args_cli.dups, replace=False)
        P = np.concatenate([P, P[dup_src]], 0)
        meta = {"xs": xs.tolist(), "ys": ys.tolist(), "n": args_cli.n, "dup_src": dup_src.tolist()}
    else:
        eps = [0.0] + [float(e) for e in args_cli.eps.split(",")]
        rows, tags = [], []
        for e in eps:
            base = np.stack([rng.uniform(lo_x, hi_x, args_cli.pairs), rng.uniform(lo_y, hi_y, args_cli.pairs)], 1)
            phi = rng.uniform(0, 2 * np.pi, args_cli.pairs)
            other = base + e * np.stack([np.cos(phi), np.sin(phi)], 1)
            rows += [base, other]
            tags += [e] * args_cli.pairs
        P = np.concatenate(rows, 0)
        meta = {"eps": eps, "pairs": args_cli.pairs,
                "note": "blocks per eps: [base (pairs), other (pairs)] in the order of eps"}
    return P, meta


def main() -> int:
    P, meta = layout()
    num_envs = len(P)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    apply_interface(env_cfg, "velocity")
    ev = env_cfg.events
    ev.reset_link1.params["position_range"] = (IC_LINK1, IC_LINK1)
    ev.reset_link1.params["velocity_range"] = (0.0, 0.0)
    ev.reset_links23.params["position_range"] = (IC_LINK23, IC_LINK23)
    ev.reset_links23.params["velocity_range"] = (0.0, 0.0)
    ev.reset_cart.params["position_range"] = (0.0, 0.0)
    ev.reset_cart.params["velocity_range"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    m, I = view.get_masses().clone(), view.get_inertias().clone()
    sx = torch.tensor(np.exp(P[:, 0]), dtype=m.dtype)
    sy = torch.tensor(np.exp(P[:, 1]), dtype=m.dtype)
    for name, s in (("link1", sx), ("link3", sy)):
        b = names.index(name)
        m[:, b] *= s.to(m.device)
        I[:, b] *= s.to(I.device).unsqueeze(-1)
    idx = torch.arange(view.count, dtype=torch.int32)
    view.set_masses(m, idx)
    view.set_inertias(I, idx)
    got = view.get_masses()
    mass_check = float(torch.max(torch.abs(got[:, names.index("link1")] - m[:, names.index("link1")])))

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    jn = list(robot.joint_names)
    li = [jn.index(j) for j in ("joint1", "joint2", "joint3")]
    dev = env.unwrapped.device
    L = torch.tensor(link_lengths(), device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * 0.75)
    with torch.inference_mode():
        env.unwrapped.reset()
        obs = env.get_observations()
        q0 = robot.data.joint_pos[:, li].clone()
        n = env.num_envs
        early = torch.zeros(n, dtype=torch.bool, device=dev)
        up = torch.zeros(n, dtype=torch.bool, device=dev)
        held = torch.zeros(n, device=dev)
        for k in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            th = torch.cumsum(robot.data.joint_pos[:, li], dim=-1)
            tip = (L * torch.cos(th)).sum(-1) / L.sum()
            up |= tip > 0.9
            if k >= hold_from:
                held += (tip > 0.9).float()
            if k < steps - 1:
                early |= dones.bool()
        hold = held / (steps - hold_from)
        success = (~early) & up & (hold > 0.95)
    ic_spread = float((q0 - q0[0:1]).abs().max())
    out = {"checkpoint": args_cli.checkpoint, "mode": args_cli.mode, "num_envs": num_envs,
           "window": {"cx": args_cli.cx, "cy": args_cli.cy, "half": args_cli.half},
           "ic": {"link1": IC_LINK1, "links23": IC_LINK23, "max_spread_rad": ic_spread},
           "mass_write_max_err": mass_check, "meta": meta, "params": P.tolist(),
           "success": success.cpu().numpy().astype(int).tolist(),
           "hold": hold.cpu().numpy().round(4).tolist(), "early": early.cpu().numpy().astype(int).tolist()}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    json.dump(out, open(args_cli.out, "w", encoding="utf-8"))
    print("[fractal] %s: %d envs, success %.1f%%, IC spread %.2e rad, mass write err %.1e -> %s"
          % (args_cli.mode, num_envs, 100 * float(success.float().mean()), ic_spread, mass_check, args_cli.out))
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
