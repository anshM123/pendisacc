"""Confirm the discovered cart_mass <-> kv null direction in Isaac.

tools/geometry_discover.py found, on the CPU reproduction, that scaling the
cart mass and the inner velocity-loop gain TOGETHER is far more benign than
scaling either alone -- a symmetry between a plant parameter and a controller
gain, holding the loop time constant m_cart/kv fixed.

A first attempt at confirmation used +20% and was uninformative: all four
conditions scored 100.0%, because the policy is simply robust at that size.
A null direction can only be demonstrated where the non-null directions
actually fail, so this sweeps until they do.

  cart_only(c)  cart mass x c, kv unchanged     -- loop time constant x c
  kv_only(c)    kv x c, cart mass unchanged     -- loop time constant / c
  both(c)       both x c                        -- loop time constant UNCHANGED

Prediction: both(c) survives to larger c than either single. If instead all
three degrade together, the direction is an artefact of the reduced-order
model and is reported as unconfirmed.

SANITY CHECK included: kv x 0.25 must visibly hurt. If changing kv does
nothing at all, write_joint_damping_to_sim is not reaching the physics and
every kv number here is meaningless.

  run.cmd tools/geom_confirm.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Confirm the discovered null direction.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--checkpoint", type=str,
                    default="logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--upright", type=float, default=0.9)
parser.add_argument("--hold_frac", type=float, default=0.25)
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
from triple_project.assets import CART_JOINT  # noqa: E402

OUT = os.path.join(ROOT, "results", "geom_confirm.json")
KV0 = 400.0
C_VALUES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def link_lengths():
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                         "triple_pendulum_params.yaml"), encoding="utf-8"))
    b = p["bodies"]
    return [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in ("link1", "link2", "link3")]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    ci = names.index("cart")
    M0 = view.get_masses().clone()
    idx = torch.arange(view.count, dtype=torch.int32)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.join(ROOT, args_cli.checkpoint)
                if not os.path.isabs(args_cli.checkpoint) else args_cli.checkpoint)
    dev = env.unwrapped.device
    policy = runner.get_inference_policy(device=dev)

    jnames = list(robot.joint_names)
    cj = jnames.index(CART_JOINT)
    link_idx = [jnames.index(j) for j in ("joint1", "joint2", "joint3")]
    Ld = torch.tensor(link_lengths(), device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

    def setup(cart_scale, kv):
        m = M0.clone()
        m[:, ci] *= cart_scale
        view.set_masses(m, idx)
        d = torch.full((robot.num_instances, 1), float(kv), device=dev)
        robot.write_joint_damping_to_sim(d, joint_ids=[cj])
        return float(view.get_masses()[0, ci]), float(robot.data.joint_damping[0, cj])

    def rollout():
        with torch.inference_mode():
            env.unwrapped.reset()
            obs = env.get_observations()
            n = env.num_envs
            early = torch.zeros(n, dtype=torch.bool, device=dev)
            ever = torch.zeros(n, dtype=torch.bool, device=dev)
            held = torch.zeros(n, device=dev)
            hn = 0
            for k in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                th = torch.cumsum(robot.data.joint_pos[:, link_idx], dim=-1)
                tip = (Ld * torch.cos(th)).sum(dim=-1) / Ld.sum()
                ever |= tip > args_cli.upright
                if k >= hold_from:
                    held += (tip > args_cli.upright).float()
                    hn += 1
                if k < steps - 1:
                    early |= dones.bool()
            hf = held / max(hn, 1)
            return float(((~early) & ever & (hf > 0.95)).float().mean())

    conds = [("sanity_kv_x0.25", 1.0, 0.25 * KV0)]
    for c in C_VALUES:
        conds.append(("cart_only_x%g" % c, c, KV0))
        conds.append(("kv_only_x%g" % c, 1.0, c * KV0))
        conds.append(("both_x%g" % c, c, c * KV0))

    rows = []
    print("frozen policy, %d episodes per condition\n" % args_cli.num_envs)
    print("  condition             cart [kg]   kv [N s/m]   tau_loop [ms]  success")
    print("  " + "-" * 72)
    for name, cs, kv in conds:
        mc, kvr = setup(cs, kv)
        s = rollout()
        tau_ms = 1000.0 * mc / max(kvr, 1e-9)
        rows.append({"condition": name, "cart_scale": cs, "kv": kv,
                     "cart_mass_kg": mc, "kv_readback": kvr,
                     "tau_loop_ms": tau_ms, "success": s})
        print("  %-20s %8.4f %11.1f %12.3f     %6.1f%%"
              % (name, mc, kvr, tau_ms, 100 * s))

    json.dump({"checkpoint": args_cli.checkpoint, "num_envs": args_cli.num_envs,
               "kv_nominal": KV0, "c_values": list(C_VALUES), "rows": rows},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("")
    print("[out] %s" % OUT)
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
    simulation_app.close()
    _hard_exit(0)
