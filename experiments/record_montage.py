"""Record Isaac trajectories for the highlight montage.

One environment per highlight condition, all from the same fixed initial condition as the
transfer maps, 12 s each. Saves joint angles and the cart position every control step, plus
the success verdict computed exactly as in experiments/fractal_map.py.

Conditions are chosen from committed results, not by hand:
  * nominal
  * three mass pairs from the interior of the success region of results/fractal/grid_A.json
  * three from the failure region
  * the twin pair at eps = 1e-3 in results/fractal/pairs.json whose outcomes disagree

  run.cmd experiments/record_montage.py --out results/fractal/montage_clips.npz
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
parser.add_argument("--checkpoint",
                    default="logs/rsl_rl/tip_swingup/2026-09-11_22-27-54_CORR_s1/model_800.pt")
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

IC_LINK1, IC_LINK23 = math.pi + 0.03, 0.01
IC_TABLE = None


def table_reset(env, env_ids):
    robot = env.scene["robot"]
    jn = list(robot.joint_names)
    pos = robot.data.default_joint_pos[env_ids].clone()
    vel = torch.zeros_like(pos)
    T = IC_TABLE.to(pos.device)[env_ids]
    for c, j in enumerate(("joint1", "joint2", "joint3")):
        pos[:, jn.index(j)] += T[:, c]
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def conditions():
    """(label, x, y) in log mass-scale coordinates, chosen from committed results."""
    g = json.load(open(os.path.join(ROOT, "results", "fractal", "grid_A.json"), encoding="utf-8"))
    n = g["meta"]["n"]
    S = np.array(g["success"][: n * n]).reshape(n, n)
    xs, ys = np.array(g["meta"]["xs"]), np.array(g["meta"]["ys"])
    # interior points: all four neighbours agree, so they are not boundary artefacts
    same = np.ones_like(S, dtype=bool)
    same[1:, :] &= S[1:, :] == S[:-1, :]
    same[:-1, :] &= S[:-1, :] == S[1:, :]
    same[:, 1:] &= S[:, 1:] == S[:, :-1]
    same[:, :-1] &= S[:, :-1] == S[:, 1:]
    out = [("nominal", 0.0, 0.0)]
    rng = np.random.default_rng(7)
    for want, tag in ((1, "holds"), (0, "falls")):
        idx = np.argwhere(same & (S == want))
        idx = idx[np.argsort(np.abs(xs[idx[:, 0]]) + np.abs(ys[idx[:, 1]]))]   # nearest first
        pick = idx[rng.choice(min(len(idx), 60), size=3, replace=False)]
        for i, j in pick:
            out.append(("%s: m1 x%.2f, m3 x%.2f" % (tag, np.exp(xs[i]), np.exp(ys[j])), xs[i], ys[j]))
    p = json.load(open(os.path.join(ROOT, "results", "fractal", "pairs.json"), encoding="utf-8"))
    eps, P = p["meta"]["eps"], p["meta"]["pairs"]
    i = eps.index(0.001)
    S2 = np.array(p["success"])
    a0, b0 = 2 * i * P, 2 * i * P + P
    k = int(np.argmax(S2[a0:a0 + P] != S2[b0:b0 + P]))
    par = np.array(p["params"])
    out.append(("twin A", par[a0 + k, 0], par[a0 + k, 1]))
    out.append(("twin B (+0.1% mass)", par[b0 + k, 0], par[b0 + k, 1]))
    return out


def main() -> int:
    global IC_TABLE
    conds = conditions()
    n = len(conds)
    P = np.array([[c[1], c[2]] for c in conds])
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    apply_interface(env_cfg, "velocity")
    IC_TABLE = torch.tensor(np.tile([IC_LINK1, IC_LINK23, IC_LINK23], (n, 1)), dtype=torch.float32)
    ev = env_cfg.events
    for term in ("reset_link1", "reset_links23", "reset_cart"):
        getattr(ev, term).params["position_range"] = (0.0, 0.0)
        getattr(ev, term).params["velocity_range"] = (0.0, 0.0)
    ev.reset_links23.func = table_reset
    ev.reset_links23.params = {}

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    m, I = view.get_masses().clone(), view.get_inertias().clone()
    for col, name in ((0, "link1"), (1, "link3")):
        b = names.index(name)
        s = torch.tensor(np.exp(P[:, col]), dtype=m.dtype)
        m[:, b] *= s
        I[:, b] *= s.unsqueeze(-1)
    idx = torch.arange(view.count, dtype=torch.int32)
    view.set_masses(m, idx)
    view.set_inertias(I, idx)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.join(ROOT, args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    jn = list(robot.joint_names)
    li = [jn.index(j) for j in ("joint1", "joint2", "joint3")]
    ci = jn.index("cart_slide")
    lp = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8"))
    L = np.array([float(lp["bodies"][k].get("length", 2.0 * lp["bodies"][k]["l_com"]))
                  for k in ("link1", "link2", "link3")])
    dev = env.unwrapped.device
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * 0.75)
    TH = np.zeros((steps, n, 3), dtype=np.float32)
    CART = np.zeros((steps, n), dtype=np.float32)
    ACT = np.zeros((steps, n), dtype=np.float32)
    with torch.inference_mode():
        env.unwrapped.reset()
        obs = env.get_observations()
        early = torch.zeros(n, dtype=torch.bool, device=dev)
        up = torch.zeros(n, dtype=torch.bool, device=dev)
        held = torch.zeros(n, device=dev)
        Lt = torch.tensor(L, device=dev, dtype=torch.float32)
        for k in range(steps):
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            th = torch.cumsum(robot.data.joint_pos[:, li], dim=-1)
            TH[k] = th.cpu().numpy()
            CART[k] = robot.data.joint_pos[:, ci].cpu().numpy()
            ACT[k] = a[:, 0].clamp(-1, 1).cpu().numpy()
            tip = (Lt * torch.cos(th)).sum(-1) / Lt.sum()
            up |= tip > 0.9
            if k >= hold_from:
                held += (tip > 0.9).float()
            if k < steps - 1:
                early |= dones.bool()
    ok = ((~early) & up & (held / (steps - hold_from) > 0.95)).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.join(ROOT, args_cli.out)), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, args_cli.out), theta=TH, cart=CART, action=ACT,
                        success=ok, params=P, labels=np.array([c[0] for c in conds]),
                        link_lengths=L, dt=1 / 250.0)
    for c, s in zip(conds, ok):
        print("  %-34s %s" % (c[0], "SUCCESS" if s else "FAIL"))
    print("[montage] %d clips x %d steps -> %s" % (n, steps, args_cli.out))
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
