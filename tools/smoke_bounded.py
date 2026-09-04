"""
Bounded Isaac Lab smoke test: build an env, step it a fixed number of times,
report throughput, exit.

The stock Isaac Lab tutorial/demo scripts loop on `while simulation_app.is_running()`
and never terminate headless, which is useless for automated checking.
"""

from __future__ import annotations

import argparse
import time

import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
# not provided by AppLauncher; the stock Isaac Lab scripts declare it themselves
parser.add_argument("--disable_fabric", action="store_true",
                    help="disable fabric and use USD I/O instead")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    print("[smoke] building %s with %d envs" % (args.task, args.num_envs))
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                        use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=cfg)
    print("[smoke] obs space:", env.observation_space)
    print("[smoke] act space:", env.action_space)

    env.reset()
    print("[smoke] reset OK")

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(args.steps):
            act = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(act)
    dt = time.perf_counter() - t0

    env.close()
    total = args.steps * args.num_envs
    rate = total / dt
    msg = ("[smoke] %d steps x %d envs = %d env-steps in %.2f s -> %.0f env-steps/s"
           % (args.steps, args.num_envs, total, dt, rate))
    print(msg)
    print("[smoke] PASS")

    # Kit takes over stdout after launch, so the console output above is not a
    # reliable record. Write the result to disk instead.
    import json
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "smoke_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "status": "PASS",
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": str(args.device),
            "fabric": not args.disable_fabric,
            "seconds": dt,
            "env_steps_per_s": rate,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }, fh, indent=2)
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
