"""Does a xi setting actually reach the simulation?

Five of six xi knobs measured as having exactly no effect on success, which is
indistinguishable from "this perturbation does not matter" and would have gone
into the correlation as a real data point. This reads the values back OUT of the
running simulation and compares them against what was asked for, so a knob that
silently fails is caught as a mismatch rather than as a null result.

  run.cmd tools/diag_xi.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TIP-SwingUp-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if sys.platform == "win32":
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/vulkan=false"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "source"))
import triple_project.tasks  # noqa: E402,F401

LINES = []


def report(label, asked, got):
    ok = abs(float(asked) - float(got)) < 1e-6 * max(1.0, abs(float(asked)))
    LINES.append("  %-24s asked %-12.6g got %-12.6g %s"
                 % (label, asked, got, "OK" if ok else "<<< DROPPED"))


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=4)

    # ask for a distinctive value on every knob
    cfg.scene.robot.actuators["cart"].damping = 150.0
    cfg.scene.robot.actuators["passive"].damping = 0.004
    cfg.scene.robot.actuators["passive"].friction = 0.004
    cfg.actions.cart_velocity.deadband = 0.05

    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    ci, li = names.index("cart_slide"), names.index("joint1")

    LINES.append("joint properties read back from the running articulation:")
    report("cart damping (kv)", 150.0, float(robot.data.joint_damping[0, ci]))
    report("passive damping", 0.004, float(robot.data.joint_damping[0, li]))
    try:
        report("passive friction", 0.004, float(robot.data.joint_friction_coeff[0, li]))
    except AttributeError:
        try:
            report("passive friction", 0.004, float(robot.data.joint_friction[0, li]))
        except AttributeError:
            LINES.append("  %-24s no joint_friction field on robot.data" % "passive friction")

    term = env.unwrapped.action_manager.get_term("cart_velocity")
    LINES.append("")
    LINES.append("action term:")
    report("deadband", 0.05, float(getattr(term.cfg, "deadband", -1)))

    LINES.append("")
    LINES.append("body masses (physx view), before any scaling event:")
    m = robot.root_physx_view.get_masses()
    for b, n in enumerate(robot.body_names):
        LINES.append("  %-24s %.6f kg" % (n, float(m[0, b])))

    LINES.append("")
    LINES.append("event manager terms: %s" % list(env.unwrapped.event_manager.active_terms))

    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    out = os.path.join(ROOT, "results", "diag_xi.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    simulation_app.close()
    sys.stdout.flush()
    os._exit(code)
