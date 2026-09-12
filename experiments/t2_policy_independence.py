"""T2: is the symmetry policy-independent?

Pre-registered in PREREGISTRATION_OVERNIGHT.md.

Everything measured in this project is policy-dependent and badly so: 77 points
of transfer spread at identical settings, and in-simulator success ANTI-
correlated with transfer among working policies (rho = -0.570). If the symmetry
is genuinely a property of (plant, interface) then it should be the one thing
that is not -- every competent policy invariant under it, to the same degree.

  run.cmd experiments/t2_policy_independence.py --checkpoints a.pt b.pt ...
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
parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 16.0, 64.0])
parser.add_argument("--upright", type=float, default=0.9)
parser.add_argument("--hold_frac", type=float, default=0.25)
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
from triple_project.assets.triple_pendulum import CART_VELOCITY_GAIN, DRIVE_CLAMP_FORCE  # noqa: E402


def link_lengths():
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                         "triple_pendulum_params.yaml"), encoding="utf-8"))
    b = p["bodies"]
    return [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in ("link1", "link2", "link3")]


def main() -> int:
    labels = args_cli.labels or [os.path.basename(os.path.dirname(c))
                                 for c in args_cli.checkpoints]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    bidx = {n: names.index(n) for n in ("link1", "link2", "link3", "cart") if n in names}
    M0, I0 = view.get_masses().clone(), view.get_inertias().clone()
    idx = torch.arange(view.count, dtype=torch.int32)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    dev = env.unwrapped.device
    jn = list(robot.joint_names)
    cj = jn.index(CART_JOINT)
    li = [jn.index(j) for j in ("joint1", "joint2", "joint3")]
    Ld = torch.tensor(link_lengths(), device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

    def apply(c):
        """The full generator: every inertial parameter, kv and the clamp."""
        m, I = M0.clone(), I0.clone()
        for nm in ("link1", "link2", "link3", "cart"):
            if nm in bidx:
                m[:, bidx[nm]] *= c
                I[:, bidx[nm]] *= c
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        robot.write_joint_damping_to_sim(
            torch.full((robot.num_instances, 1), CART_VELOCITY_GAIN * c, device=dev),
            joint_ids=[cj])
        robot.write_joint_effort_limit_to_sim(
            torch.full((robot.num_instances, 1), DRIVE_CLAMP_FORCE * c, device=dev),
            joint_ids=[cj])

    def rollout(policy):
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
                th = torch.cumsum(robot.data.joint_pos[:, li], dim=-1)
                tip = (Ld * torch.cos(th)).sum(dim=-1) / Ld.sum()
                ever |= tip > args_cli.upright
                if k >= hold_from:
                    held += (tip > args_cli.upright).float()
                    hn += 1
                if k < steps - 1:
                    early |= dones.bool()
            hf = held / max(hn, 1)
            return float(((~early) & ever & (hf > 0.95)).float().mean())

    out = args_cli.out or os.path.join(ROOT, "results", "T2", "policy_independence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    print("T2: is the symmetry policy-independent?\n")
    print("  %-26s %s" % ("policy", "  ".join("c=%-6g" % c for c in args_cli.c_values)))
    print("  " + "-" * 60)
    for ck, lab in zip(args_cli.checkpoints, labels):
        if not os.path.exists(ck):
            print("  [skip] missing %s" % ck)
            continue
        runner.load(ck)
        pol = runner.get_inference_policy(device=dev)
        vals = []
        for c in args_cli.c_values:
            apply(c)
            vals.append(rollout(pol))
        rows.append({"policy": lab, "checkpoint": ck,
                     "c_values": args_cli.c_values, "success": vals})
        print("  %-26s %s" % (lab, "  ".join("%6.1f%%" % (100 * v) for v in vals)))
        json.dump({"rows": rows}, open(out, "w", encoding="utf-8"), indent=1)

    comp = [r for r in rows if r["success"][0] >= 0.90]
    print("\n  %d of %d policies are competent at c=1" % (len(comp), len(rows)))
    if comp:
        worst = min(r["success"][-1] for r in comp)
        drops = [100 * (r["success"][0] - r["success"][-1]) for r in comp]
        print("  under the generator at c=%g: worst competent policy %.1f%%"
              % (args_cli.c_values[-1], 100 * worst))
        print("  P1 (all competent >= 90%%)      -> %s"
              % ("SUPPORTED" if worst >= 0.90 else "FAIL"))
        print("  P2 (spread of drop <= 10 pts)  -> %s  (spread %.1f)"
              % ("SUPPORTED" if (max(drops) - min(drops)) <= 10 else "FAIL",
                 max(drops) - min(drops)))
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
