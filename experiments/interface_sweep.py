"""H4: sweep mass-error DIRECTION at matched magnitude, under one interface.

Pre-registered in PREREGISTRATION_H4.md (commit 9aa34a0). This is the
measurement instrument, not the analysis -- scoring lives in h4_score.py so the
numbers cannot be reshaped once they exist.

Why one process for every (policy, condition) pair: mass perturbations are
applied through the PhysX view AFTER gym.make, not as an event term, so they
can be re-applied to a live env. One Isaac launch costs ~4 minutes and a
rollout costs well under a minute, so sweeping in-process turns a 5-hour job
into a 40-minute one. The env is never rebuilt, which also removes
rebuild-to-rebuild variation from the comparison.

MATCHED PARAMETER DISTANCE is the whole design. For uniform scaling by c the
displacement in link-mass space is dm = (c-1) * m0, of norm (c-1)*||m0||. Each
transverse condition puts that SAME norm entirely into one link:

    scale_i = 1 + (c - 1) * ||m0|| / m0_i

so every condition at a given c is exactly as far from nominal as every other,
and any difference between them is due to direction alone. Without this the
comparison would merely be re-discovering that bigger errors are worse.

  run.cmd experiments/interface_sweep.py --interface force --runs <dir> <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H4 direction-at-matched-magnitude sweep.")
parser.add_argument("--task", type=str, default="TIP-SwingUp-Play-v0")
parser.add_argument("--interface", type=str, default="velocity", choices=("velocity", "force"))
parser.add_argument("--runs", type=str, nargs="+", required=True,
                    help="run directories; the best checkpoint of each is selected by "
                         "measured success at c=1 in its OWN interface, which is what a "
                         "practitioner without access to reality would do")
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 1.5, 2.5, 4.0, 8.0])
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


def link_masses_and_lengths():
    path = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
    with open(path, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    b = p["bodies"]
    names = ("link1", "link2", "link3")
    m = [float(b[n]["mass"]) for n in names]
    L = [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in names]
    return m, L


def conditions(m0):
    """Uniform and per-link transverse errors at MATCHED ||dm|| for each c."""
    norm = float(np.linalg.norm(m0))
    out = []
    for c in args_cli.c_values:
        d = (c - 1.0) * norm
        out.append({"c": c, "direction": "uniform", "scale": [c, c, c], "dm": d})
        if c == 1.0:
            continue
        for i in range(3):
            s = [1.0, 1.0, 1.0]
            s[i] = 1.0 + d / m0[i]
            out.append({"c": c, "direction": "transverse_link%d" % (i + 1),
                        "scale": s, "dm": d})
    return out


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
    m0, Llen = link_masses_and_lengths()
    conds = conditions(m0)
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
    # pristine copies: every condition is applied to THESE, never cumulatively
    M0 = view.get_masses().clone()
    I0 = view.get_inertias().clone()
    idx = torch.arange(view.count, dtype=torch.int32)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    jnames = list(robot.joint_names)
    link_idx = [jnames.index(j) for j in ("joint1", "joint2", "joint3")]
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
                m[:, b] *= s      # inertia scaled by the same factor:
                I[:, b] *= s      # a density error at fixed geometry
        view.set_masses(m, idx)
        view.set_inertias(I, idx)
        return {n: round(float(view.get_masses()[0, i]), 6)
                for i, n in enumerate(body_names)}

    def rollout(policy):
        with torch.inference_mode():
            env.unwrapped.reset()
            obs = env.get_observations()
            n = env.num_envs
            early = torch.zeros(n, dtype=torch.bool, device=dev)
            ever = torch.zeros(n, dtype=torch.bool, device=dev)
            held = torch.zeros(n, device=dev)
            held_n = 0
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
            hf = held / max(held_n, 1)
            ok = (~early) & ever & (hf > 0.95)
            return float(ok.float().mean()), float(early.float().mean())

    print("[sweep] interface  :", args_cli.interface, json.dumps(iface))
    print("[sweep] link masses:", [round(x, 5) for x in m0],
          "  ||m0|| = %.5f kg" % float(np.linalg.norm(m0)))
    print("[sweep] %d policies x %d conditions" % (len(args_cli.runs), len(conds)))

    rows = []
    selection = []
    for label, run in zip(labels, args_cli.runs):
        # Checkpoint selection uses NOMINAL success in the policy's own
        # interface. No information from any perturbed condition is used, so
        # selection cannot leak the answer into the sweep.
        set_masses([1.0, 1.0, 1.0])
        scored = []
        for ck in checkpoint_list(run):
            runner.load(ck)
            r, e = rollout(runner.get_inference_policy(device=dev))
            scored.append((r, e, ck))
            print("  [%s] select %-16s c=1 success %6.1f%%"
                  % (label, os.path.basename(ck), 100 * r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        best_r, _, best_ck = scored[0]
        print("  [%s] CHOSEN %s  (c=1 success %.1f%%)"
              % (label, os.path.basename(best_ck), 100 * best_r))
        selection.append({"policy": label, "run": run,
                          "checkpoint": os.path.basename(best_ck),
                          "nominal_success": best_r,
                          "considered": [{"checkpoint": os.path.basename(c),
                                          "nominal_success": s} for s, _, c in scored]})
        runner.load(best_ck)
        policy = runner.get_inference_policy(device=dev)

        for cd in conds:
            applied = set_masses(cd["scale"])
            r, e = rollout(policy)
            rows.append({"policy": label, "run": run,
                         "checkpoint": os.path.basename(best_ck),
                         "interface": args_cli.interface, "c": cd["c"],
                         "direction": cd["direction"],
                         "scale": [round(float(x), 5) for x in cd["scale"]],
                         "dm_kg": cd["dm"], "success": r, "early_term": e,
                         "masses_in_sim": applied})
            print("  [%s] c=%-5.2f %-18s ||dm||=%.4f  success %6.1f%%"
                  % (label, cd["c"], cd["direction"], cd["dm"], 100 * r))

    out = args_cli.out or os.path.join(ROOT, "results", "h4_sweep_%s.json" % args_cli.interface)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"interface": args_cli.interface, "interface_detail": iface,
                   "num_envs": args_cli.num_envs, "task": args_cli.task,
                   "link_masses_nominal": m0, "c_values": args_cli.c_values,
                   "upright_threshold": args_cli.upright,
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
