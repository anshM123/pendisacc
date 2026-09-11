"""H7: evaluate every deployed policy under one stress condition.

Pre-registered in PREREGISTRATION_H7.md. One Isaac launch per stress
condition, all 24 policies inside it, because the actuator parameters (tau,
delay) live in the action-term configuration and cannot be changed on a live
environment the way masses can.

  run.cmd experiments/h7_stress.py --xi xi/stress/S1.json --tag S1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H7 stress evaluation.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--xi", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
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
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import triple_project.tasks  # noqa: E402,F401
from interfaces import action_term  # noqa: E402

LOGS = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup")


def policies():
    """The 24 policies already deployed into R*, with their chosen checkpoint."""
    out = []
    for sel, dep, pref in (("claim5_select", "claim5", "C5_"),
                           ("h5_select", "h5", "")):
        for f in sorted(glob.glob(os.path.join(ROOT, "results", sel, "*.json"))):
            tag = os.path.basename(f)[:-5]
            dpath = os.path.join(ROOT, "results", dep, "%s.json" % tag)
            if not os.path.exists(dpath):
                continue
            s = json.load(open(f, encoding="utf-8"))
            best = max(s["results"],
                       key=lambda r: (r["success_rate"], -r["early_termination_rate"]))
            runs = sorted(glob.glob(os.path.join(LOGS, "*_%s" % tag.replace(pref, "", 1)
                                                 if pref else "*_%s" % tag)))
            if not runs:
                runs = sorted(glob.glob(os.path.join(LOGS, "*_%s" % tag)))
            if not runs:
                print("  [skip] no run dir for", tag)
                continue
            ck = os.path.join(runs[-1], best["checkpoint"])
            if not os.path.exists(ck):
                print("  [skip] missing checkpoint", ck)
                continue
            out.append({"policy": tag,
                        "checkpoint": ck,
                        "own_sim": 100.0 * best["success_rate"],
                        "R_star": 100.0 * json.load(open(dpath, encoding="utf-8"))
                        ["results"][0]["success_rate"]})
    return out


def link_lengths():
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                         "triple_pendulum_params.yaml"), encoding="utf-8"))
    b = p["bodies"]
    return [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in ("link1", "link2", "link3")]


def main() -> int:
    xi = json.load(open(os.path.join(ROOT, args_cli.xi)
                        if not os.path.isabs(args_cli.xi) else args_cli.xi,
                        encoding="utf-8"))
    pols = policies()
    print("[h7] tag %s  xi %s" % (args_cli.tag, json.dumps(xi)))
    print("[h7] %d policies" % len(pols))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    act = action_term(env_cfg)
    for k, attr in (("tau", "time_constant_s"), ("delay_s", "delay_s"),
                    ("deadband", "deadband"), ("order", "order"), ("zeta", "zeta")):
        if k in xi:
            setattr(act, attr, int(xi[k]) if k == "order" else float(xi[k]))
    rob = env_cfg.scene.robot
    if "kv" in xi:
        rob.actuators["cart"].damping = float(xi["kv"])
    if "joint_damping" in xi:
        rob.actuators["passive"].damping = float(xi["joint_damping"])
    if "joint_friction" in xi:
        rob.actuators["passive"].friction = float(xi["joint_friction"])
    if "gravity" in xi:
        env_cfg.sim.gravity = (0.0, 0.0, -float(xi["gravity"]))

    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.body_names)
    applied = None
    if any(k in xi for k in ("mass_scale", "cart_mass_scale")):
        m, I = view.get_masses().clone(), view.get_inertias().clone()
        want = {"cart": float(xi.get("cart_mass_scale", 1.0))}
        for i, s in enumerate(xi.get("mass_scale", [1.0, 1.0, 1.0])):
            want["link%d" % (i + 1)] = float(s)
        for n, sc in want.items():
            if n in names and sc != 1.0:
                b = names.index(n)
                m[:, b] *= sc
                I[:, b] *= sc
        idx = torch.arange(view.count, dtype=torch.int32)
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        applied = {n: round(float(view.get_masses()[0, i]), 6) for i, n in enumerate(names)}
        print("[h7] masses in sim:", applied)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    jn = list(robot.joint_names)
    li = [jn.index(j) for j in ("joint1", "joint2", "joint3")]
    dev = env.unwrapped.device
    Ld = torch.tensor(link_lengths(), device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

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

    rows = []
    for p in pols:
        runner.load(p["checkpoint"])
        s = rollout(runner.get_inference_policy(device=dev))
        rows.append(dict(p, stress=100.0 * s))
        print("  %-22s stress %6.1f%%   (own-sim %6.1f%%, R* %6.1f%%)"
              % (p["policy"], 100 * s, p["own_sim"], p["R_star"]))

    out = os.path.join(ROOT, "results", "h7", "%s.json" % args_cli.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"tag": args_cli.tag, "xi": xi, "num_envs": args_cli.num_envs,
               "masses_in_sim": applied, "rows": rows},
              open(out, "w", encoding="utf-8"), indent=1)
    print("")
    print("[out] %s" % out)
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
