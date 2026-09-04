"""
Capture raw Isaac trajectories + joint metadata so the D1/D2/D3 failures can be
diagnosed offline, without paying an Isaac launch per hypothesis.

Writes results/diag_dynamics.npz.
"""

from __future__ import annotations

import argparse
import json
import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=1.0 / 1000.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
USD = os.path.join(ROOT, "assets", "triple_pendulum", "usd", "triple_pendulum.usd")
OUT = os.path.join(ROOT, "results", "diag_dynamics.npz")

REVOLUTE = ["joint1", "joint2", "joint3"]
PRISMATIC = "cart_slide"


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device=args.device, gravity=(0.0, 0.0, -9.80665))
    )
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "cart": ImplicitActuatorCfg(joint_names_expr=[PRISMATIC],
                                        effort_limit_sim=200.0, stiffness=0.0, damping=0.0),
            "passive": ImplicitActuatorCfg(joint_names_expr=REVOLUTE,
                                           effort_limit_sim=0.0, stiffness=0.0, damping=0.0),
        },
    )
    robot = Articulation(cfg)
    sim.reset()

    names = list(robot.joint_names)
    idx = [names.index(PRISMATIC)] + [names.index(j) for j in REVOLUTE]
    zero = torch.zeros((1, len(names)), device=robot.device)

    meta = {
        "joint_names": names,
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().numpy().tolist(),
        "default_joint_vel": robot.data.default_joint_vel[0].cpu().numpy().tolist(),
    }
    for attr in ("joint_pos_limits", "soft_joint_pos_limits", "joint_vel_limits",
                 "soft_joint_vel_limits", "joint_effort_limits"):
        try:
            meta[attr] = getattr(robot.data, attr)[0].cpu().numpy().tolist()
        except Exception as exc:
            meta[attr] = "unavailable: %s" % exc

    def roll(q0, seconds, tag):
        jp = robot.data.default_joint_pos.clone()
        jv = torch.zeros_like(jp)
        for k, i in enumerate(idx):
            jp[0, i] = float(q0[k])
        # reset FIRST, then write, so nothing restores defaults over our write
        robot.reset()
        robot.write_joint_state_to_sim(jp, jv)
        # what did the sim actually accept?
        robot.update(0.0)
        accepted = robot.data.joint_pos[0, idx].cpu().numpy().copy()
        print("[%s] requested %s -> accepted %s" % (tag, np.round(q0, 5), np.round(accepted, 5)))

        n = int(seconds / args.dt)
        qs, vs = [], []
        for _ in range(n):
            robot.set_joint_effort_target(zero)
            robot.write_data_to_sim()
            sim.step()
            robot.update(args.dt)
            qs.append(robot.data.joint_pos[0, idx].cpu().numpy().copy())
            vs.append(robot.data.joint_vel[0, idx].cpu().numpy().copy())
        return accepted, np.arange(1, n + 1) * args.dt, np.asarray(qs), np.asarray(vs)

    data = {}
    for tag, q0, secs in [
        ("upright", np.array([0.0, 0.02, 0.0, 0.0]), 0.05),
        ("hanging", np.array([0.0, np.pi + 0.15, np.pi - 0.10, np.pi + 0.05]), 3.0),
        ("tiny", np.array([0.0, 1e-4, 0.0, 0.0]), 0.30),
    ]:
        acc, t, q, v = roll(q0, secs, tag)
        data[tag + "_q0_req"] = q0
        data[tag + "_q0_acc"] = acc
        data[tag + "_t"] = t
        data[tag + "_q"] = q
        data[tag + "_v"] = v

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, meta=json.dumps(meta), **data)
    print("[out]", OUT)
    print(json.dumps(meta, indent=2))
    return 0


def _hard_exit(code: int = 0) -> None:
    """Terminate immediately after Kit shutdown.

    Isaac Sim frequently hangs inside simulation_app.close() on Windows and
    leaves a python.exe spinning at 100% CPU forever. Three such orphans from
    one night's runs were burning ~57,000 CPU-seconds each and starving a
    training job. Results are already on disk by this point, so flush and use
    os._exit to skip the wedged interpreter teardown.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    _hard_exit(code)
