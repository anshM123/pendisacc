"""T1: is the derived symmetry group the WHOLE null space?

Pre-registered in PREREGISTRATION_OVERNIGHT.md. The solver proves the group of
SCALING symmetries is 3-dimensional. That does not rule out other
behaviourally-null directions which are not scalings. If several random
directions turn out to be null, a larger equivalence structure exists that we
have not characterised -- which would be a bigger finding than the theorem.

Every direction is applied at a magnitude matched to one already known to be
fatal, and every direction gets the force clamp scaled in proportion to its kv
component, so authority is never the reason a direction fails.

  run.cmd experiments/t1_nullspace.py
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
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--n_random", type=int, default=24)
parser.add_argument("--log_scale", type=float, default=3.4657)   # ln(32)
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

# coordinates: log-scaling exponents on these 8 parameters
COORDS = ["m1", "m2", "m3", "I1", "I2", "I3", "m_cart", "kv"]
GROUP = np.ones(8) / np.sqrt(8.0)          # the derived generator, normalised


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
    bidx = {n: names.index(n) for n in ("link1", "link2", "link3", "cart") if n in names}
    M0 = view.get_masses().clone()
    I0 = view.get_inertias().clone()
    idx = torch.arange(view.count, dtype=torch.int32)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    dev = env.unwrapped.device
    policy = runner.get_inference_policy(device=dev)
    jn = list(robot.joint_names)
    cj = jn.index(CART_JOINT)
    li = [jn.index(j) for j in ("joint1", "joint2", "joint3")]
    Ld = torch.tensor(link_lengths(), device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

    def apply(u, s):
        """u: unit direction in log-coordinates; s: log magnitude."""
        f = np.exp(s * np.asarray(u, dtype=float))
        m, I = M0.clone(), I0.clone()
        for j, nm in enumerate(("link1", "link2", "link3")):
            if nm in bidx:
                m[:, bidx[nm]] *= float(f[j])
                I[:, bidx[nm]] *= float(f[3 + j])
        if "cart" in bidx:
            m[:, bidx["cart"]] *= float(f[6])
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        kv = CART_VELOCITY_GAIN * float(f[7])
        robot.write_joint_damping_to_sim(
            torch.full((robot.num_instances, 1), kv, device=dev), joint_ids=[cj])
        # clamp scales with kv so authority is never the reason a direction fails
        robot.write_joint_effort_limit_to_sim(
            torch.full((robot.num_instances, 1), DRIVE_CLAMP_FORCE * float(f[7]), device=dev),
            joint_ids=[cj])
        return f.tolist(), kv

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

    rng = np.random.default_rng(0)
    conds = [("control_c1", np.zeros(8), 0.0), ("GROUP", GROUP, args_cli.log_scale)]
    for i in range(args_cli.n_random):
        u = rng.normal(size=8)
        u /= np.linalg.norm(u)
        conds.append(("rand%02d" % i, u, args_cli.log_scale))

    out = args_cli.out or os.path.join(ROOT, "results", "T1", "nullspace.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    print("T1: is the derived group the whole null space?")
    print("log magnitude %.4f  (= x%.1f along a unit direction)\n"
          % (args_cli.log_scale, np.exp(args_cli.log_scale)))
    print("  %-12s %8s   %s" % ("direction", "success", "cos to group"))
    print("  " + "-" * 58)
    for nm, u, s in conds:
        f, kv = apply(u, s)
        r = rollout()
        cg = float(abs(np.dot(u, GROUP))) if np.linalg.norm(u) > 0 else 1.0
        rows.append({"name": nm, "direction": np.asarray(u).tolist(),
                     "log_magnitude": s, "factors": f, "kv": kv,
                     "cos_to_group": cg, "success": r})
        print("  %-12s %7.1f%%   %.3f" % (nm, 100 * r, cg))
        json.dump({"coords": COORDS, "group": GROUP.tolist(), "rows": rows},
                  open(out, "w", encoding="utf-8"), indent=1)

    grp = [x for x in rows if x["name"] == "GROUP"][0]["success"]
    rnd = [x["success"] for x in rows if x["name"].startswith("rand")]
    n_null = sum(1 for x in rnd if x >= 0.90)
    print("\n  group direction      %.1f%%" % (100 * grp))
    print("  random directions >= 90%%: %d of %d" % (n_null, len(rnd)))
    print("  P1 (group >= 90%%)          -> %s" % ("SUPPORTED" if grp >= 0.90 else "FAIL"))
    print("  P2 (at most 1 random null) -> %s" % ("SUPPORTED" if n_null <= 1 else "FAIL"))
    if n_null >= 4:
        print("  KILL CONDITION: the group is not the whole null space.")
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
