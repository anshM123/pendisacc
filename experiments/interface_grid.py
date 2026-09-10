"""The 2-D transfer landscape: P_success(c, delta) under one action interface.

Run 1's headline object. Two isolated success percentages are weak evidence;
a landscape is not. The grid is

    m = m0 * [ c*(1+delta),  c,  c ]

so the two axes separate the two things that must not be confused:

    c      common scale  -- moves ALONG the exact equivalence direction
    delta  asymmetry     -- moves TRANSVERSE to it

delta = 0 is the pure equivalence direction; c = 1 is a pure transverse error.
Every other cell is a mixture. Under a velocity interface with authority to
spare the prediction is a long flat ridge along c and a sharp collapse in
delta. Under a force interface that anisotropy should shrink or rotate.

AUTHORITY IS THE CONFOUND, so it is measured rather than assumed. Every cell
records the realised cart force and velocity and the fraction of steps spent
against the drive's force clamp and speed limit. A cell where the velocity arm
sits on its clamp is not evidence about interfaces -- it is evidence about
saturation, and it is labelled so it can be excluded.

  run.cmd experiments/interface_grid.py --interface force --runs <dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="2-D interface transfer landscape.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--interface", type=str, default="velocity", choices=("velocity", "force"))
parser.add_argument("--runs", type=str, nargs="+", required=True)
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
parser.add_argument("--upright", type=float, default=0.9)
parser.add_argument("--hold_frac", type=float, default=0.25)
parser.add_argument("--stride", type=int, default=200)
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import triple_project.tasks  # noqa: E402,F401
from interfaces import apply_interface  # noqa: E402
from triple_project.assets import CART_JOINT, MAX_CART_SPEED  # noqa: E402
from triple_project.assets.triple_pendulum import DRIVE_CLAMP_FORCE  # noqa: E402


def link_lengths():
    path = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
    with open(path, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    b = p["bodies"]
    return [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in ("link1", "link2", "link3")]


def checkpoint_list(run):
    files = [f for f in os.listdir(run) if f.startswith("model_") and f.endswith(".pt")]

    def num(f):
        return int("".join(ch for ch in f if ch.isdigit()) or 0)

    files.sort(key=num)
    keep = [f for f in files if num(f) % args_cli.stride == 0]
    if files and files[-1] not in keep:
        keep.append(files[-1])
    return [os.path.join(run, f) for f in keep]


def main() -> int:
    Llen = link_lengths()
    labels = args_cli.labels or [os.path.basename(r) for r in args_cli.runs]
    assert len(labels) == len(args_cli.runs), "one label per run"

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    iface = apply_interface(env_cfg, args_cli.interface)
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    body_names = list(robot.body_names)
    M0 = view.get_masses().clone()
    I0 = view.get_inertias().clone()
    idx = torch.arange(view.count, dtype=torch.int32)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    jnames = list(robot.joint_names)
    link_idx = [jnames.index(j) for j in ("joint1", "joint2", "joint3")]
    cart_idx = jnames.index(CART_JOINT)
    dev = env.unwrapped.device
    Ld = torch.tensor(Llen, device=dev)
    steps = int(env.unwrapped.max_episode_length)
    hold_from = int(steps * (1.0 - args_cli.hold_frac))

    def set_masses(scale):
        m, I = M0.clone(), I0.clone()
        for i, s in enumerate(scale):
            name = "link%d" % (i + 1)
            if name in body_names and s != 1.0:
                b = body_names.index(name)
                m[:, b] *= s
                I[:, b] *= s
        view.set_masses(m, idx)
        view.set_inertias(I, idx)

    def rollout(policy):
        """Returns success, early-termination, and AUTHORITY telemetry."""
        with torch.inference_mode():
            env.unwrapped.reset()
            obs = env.get_observations()
            n = env.num_envs
            early = torch.zeros(n, dtype=torch.bool, device=dev)
            ever = torch.zeros(n, dtype=torch.bool, device=dev)
            held = torch.zeros(n, device=dev)
            held_n = 0
            f_abs_sum = f_abs_max = v_abs_sum = v_abs_max = 0.0
            f_sat = v_sat = 0.0
            nsteps = 0
            for k in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                th = torch.cumsum(robot.data.joint_pos[:, link_idx], dim=-1)
                tip = (Ld * torch.cos(th)).sum(dim=-1) / Ld.sum()
                ever |= tip > args_cli.upright
                if k >= hold_from:
                    held += (tip > args_cli.upright).float()
                    held_n += 1
                if k < steps - 1:
                    early |= dones.bool()
                # ---- authority telemetry, the confound this experiment must
                # not be silently measuring instead of the interface ----
                f = robot.data.applied_torque[:, cart_idx].abs()
                v = robot.data.joint_vel[:, cart_idx].abs()
                f_abs_sum += float(f.mean())
                v_abs_sum += float(v.mean())
                f_abs_max = max(f_abs_max, float(f.max()))
                v_abs_max = max(v_abs_max, float(v.max()))
                f_sat += float((f >= 0.99 * DRIVE_CLAMP_FORCE).float().mean())
                v_sat += float((v >= 0.99 * MAX_CART_SPEED).float().mean())
                nsteps += 1
            hf = held / max(held_n, 1)
            ok = (~early) & ever & (hf > 0.95)
            return {"success": float(ok.float().mean()),
                    "early_term": float(early.float().mean()),
                    "mean_abs_F_N": f_abs_sum / nsteps,
                    "max_abs_F_N": f_abs_max,
                    "force_sat_frac": f_sat / nsteps,
                    "mean_abs_v_m_s": v_abs_sum / nsteps,
                    "max_abs_v_m_s": v_abs_max,
                    "speed_sat_frac": v_sat / nsteps}

    print("[grid] interface:", args_cli.interface, json.dumps(iface))
    print("[grid] c      :", args_cli.c_values)
    print("[grid] delta  :", args_cli.deltas)
    print("[grid] %d policies x %d cells"
          % (len(args_cli.runs), len(args_cli.c_values) * len(args_cli.deltas)))

    rows, selection = [], []
    for label, run in zip(labels, args_cli.runs):
        set_masses([1.0, 1.0, 1.0])
        scored = []
        for ck in checkpoint_list(run):
            runner.load(ck)
            r = rollout(runner.get_inference_policy(device=dev))
            scored.append((r["success"], r["early_term"], ck))
            print("  [%s] select %-16s nominal %6.1f%%"
                  % (label, os.path.basename(ck), 100 * r["success"]))
        scored.sort(key=lambda t: (-t[0], t[1]))
        best_r, _, best_ck = scored[0]
        print("  [%s] CHOSEN %s  nominal %.1f%%" % (label, os.path.basename(best_ck), 100 * best_r))
        selection.append({"policy": label, "run": run,
                          "checkpoint": os.path.basename(best_ck),
                          "nominal_success": best_r})
        runner.load(best_ck)
        policy = runner.get_inference_policy(device=dev)

        for c in args_cli.c_values:
            for d in args_cli.deltas:
                scale = [c * (1.0 + d), c, c]
                set_masses(scale)
                r = rollout(policy)
                r.update({"policy": label, "interface": args_cli.interface,
                          "c": c, "delta": d,
                          "scale": [round(float(x), 5) for x in scale],
                          "checkpoint": os.path.basename(best_ck)})
                rows.append(r)
                print("  [%s] c=%-5.2f d=%-5.2f  success %6.1f%%   |F| %6.1f N "
                      "(sat %4.0f%%)   |v| %4.2f m/s (sat %4.0f%%)"
                      % (label, c, d, 100 * r["success"], r["mean_abs_F_N"],
                         100 * r["force_sat_frac"], r["mean_abs_v_m_s"],
                         100 * r["speed_sat_frac"]))

    out = args_cli.out or os.path.join(ROOT, "results", "h4_grid_%s.json" % args_cli.interface)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"interface": args_cli.interface, "interface_detail": iface,
                   "num_envs": args_cli.num_envs, "task": args_cli.task,
                   "c_values": args_cli.c_values, "deltas": args_cli.deltas,
                   "parameterisation": "m = m0 * [c*(1+delta), c, c]",
                   "drive_clamp_N": DRIVE_CLAMP_FORCE,
                   "max_cart_speed_m_s": MAX_CART_SPEED,
                   "selection": selection, "rows": rows}, fh, indent=1)
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
