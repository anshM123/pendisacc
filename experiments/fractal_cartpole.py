"""Control system for the fractal transfer-boundary study: Isaac Lab's stock cartpole.

Trains the stock Isaac-Cartpole-v0 policy (rsl-rl defaults), freezes it, fixes
the initial condition, and maps success (survive the full episode) over
x = log pole-mass scale, y = log cart-mass scale. Grid first; if the window is
all-success or all-failure it is widened once. Then the same uncertainty-exponent
pair design as experiments/fractal_map.py, in the chosen window. Output schema
matches fractal_map.py so tools/fractal_analyze.py reads it.

  run.cmd experiments/fractal_cartpole.py --out_dir results/fractal/cartpole
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Cartpole-v0")
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--n", type=int, default=64)
parser.add_argument("--pairs", type=int, default=292)
parser.add_argument("--eps", default="0.3,0.1,0.03,0.01,0.003,0.001")
parser.add_argument("--half", type=float, default=1.5)
parser.add_argument("--pole_ic", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=20260914)
parser.add_argument("--out_dir", required=True)
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

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_ENVS = 4096


def main() -> int:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N_ENVS)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.iterations
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    view = robot.root_physx_view
    M0, I0 = view.get_masses().clone(), view.get_inertias().clone()
    names = list(robot.body_names)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    log_dir = os.path.join(ROOT, "logs", "rsl_rl", "fractal_cartpole")
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    ck = os.path.join(log_dir, "frozen_model.pt")
    runner.save(ck)
    policy = runner.get_inference_policy(device=base.device)

    # fixed initial condition: every reset term that sets joint positions gets a zero-width range
    em = base.event_manager
    reset_terms = em.active_terms.get("reset", [])
    ic = {}
    for tname in reset_terms:
        cfg = em.get_term_cfg(tname)
        if "position_range" not in cfg.params:
            continue
        jn = " ".join(getattr(cfg.params.get("asset_cfg"), "joint_names", []) or [])
        val = args_cli.pole_ic if "pole" in jn or "pole" in tname else 0.0
        cfg.params["position_range"] = (val, val)
        cfg.params["velocity_range"] = (0.0, 0.0)
        em.set_term_cfg(tname, cfg)
        ic[tname] = val

    steps = int(base.max_episode_length)
    dev = base.device

    def run(P):
        m, I = M0.clone(), I0.clone()
        k = len(P)
        sx = torch.ones(N_ENVS, dtype=m.dtype)
        sy = torch.ones(N_ENVS, dtype=m.dtype)
        sx[:k] = torch.tensor(np.exp(P[:, 0]), dtype=m.dtype)
        sy[:k] = torch.tensor(np.exp(P[:, 1]), dtype=m.dtype)
        for name, s in (("pole", sx), ("cart", sy)):
            b = names.index(name)
            m[:, b] *= s
            I[:, b] *= s.unsqueeze(-1)
        idx = torch.arange(view.count, dtype=torch.int32)
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        with torch.inference_mode():
            base.reset()
            obs = wrapped.get_observations()
            early = torch.zeros(N_ENVS, dtype=torch.bool, device=dev)
            alive = torch.zeros(N_ENVS, device=dev)
            for t in range(steps):
                obs, _, dones, _ = wrapped.step(policy(obs))
                if t < steps - 1:
                    early |= dones.bool()
                alive += (~early).float()
        ok = (~early).cpu().numpy()[:k].astype(int)
        return ok, (alive / steps).cpu().numpy()[:k].round(4)

    half = args_cli.half
    for attempt in range(2):
        xs = ys = np.linspace(-half, half, args_cli.n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([X.ravel(), Y.ravel()], 1)
        ok, hold = run(G)
        frac = float(ok.mean())
        print("[cartpole] grid half %.2f success %.3f" % (half, frac), flush=True)
        if 0.02 < frac < 0.98 or attempt == 1:
            break
        half *= 2.0
    json.dump({"checkpoint": ck, "mode": "grid", "task": args_cli.task, "ic": ic, "params": G.tolist(),
               "meta": {"xs": xs.tolist(), "ys": ys.tolist(), "n": args_cli.n, "dup_src": []},
               "window": {"half": half}, "success": ok.tolist(), "hold": hold.tolist(), "early": (1 - ok).tolist()},
              open(os.path.join(args_cli.out_dir, "grid.json"), "w"))

    rng = np.random.default_rng(args_cli.seed)
    eps = [0.0] + [float(e) for e in args_cli.eps.split(",")]
    rows = []
    for e in eps:
        b0 = rng.uniform(-half, half, (args_cli.pairs, 2))
        phi = rng.uniform(0, 2 * np.pi, args_cli.pairs)
        rows += [b0, b0 + e * np.stack([np.cos(phi), np.sin(phi)], 1)]
    Pp = np.concatenate(rows, 0)
    assert len(Pp) <= N_ENVS
    ok, hold = run(Pp)
    json.dump({"checkpoint": ck, "mode": "pairs", "task": args_cli.task, "ic": ic, "params": Pp.tolist(),
               "meta": {"eps": eps, "pairs": args_cli.pairs}, "window": {"half": half},
               "success": ok.tolist(), "hold": hold.tolist(), "early": (1 - ok).tolist()},
              open(os.path.join(args_cli.out_dir, "pairs.json"), "w"))
    print("[cartpole] pairs done, success %.3f, window half %.2f -> %s" % (float(ok.mean()), half, args_cli.out_dir))
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
