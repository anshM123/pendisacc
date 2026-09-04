"""
Watch the pendulum move in a GUI window. No trained policy required.

  # passive: release from near-upright and watch it fall (this is the real plant)
  python experiments/watch.py --mode passive

  # driven: a hand-written sinusoidal cart force, to see the coupling
  python experiments/watch.py --mode sine

  # random cart forces, i.e. what PPO sees before it learns anything
  python experiments/watch.py --mode random

Leave off --headless (the default here) and an Isaac Sim window opens. Drag with
the mouse to orbit. Press the play/pause controls in the toolbar to freeze a pose.
"""

from __future__ import annotations

import argparse
import math
import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise the triple pendulum.")
parser.add_argument("--mode", choices=["passive", "sine", "random"], default="passive")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tilt", type=float, default=0.12, help="initial link1 tilt [rad]")
parser.add_argument("--force", type=float, default=8.0, help="cart force amplitude [N]")
parser.add_argument("--freq", type=float, default=1.2, help="drive frequency [Hz]")
parser.add_argument("--vulkan", action="store_true",
                    help="use Vulkan instead of D3D12 (broken on this machine)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Isaac Lab's experience file sets `vulkan = true` under [settings.app], forcing
# Vulkan on Windows where Kit otherwise defaults to D3D12. On this machine
# (RTX 5070 Ti Laptop / Blackwell, driver 596.13) the Vulkan path dies with an
# access violation inside omni.kit.viewport.window -- both for the GUI and for
# headless video capture. D3D12 works. Verified: with Vulkan the run aborts at
# viewport startup; with this flag it reaches "app ready" and renders.
#
# Override with --vulkan if a future driver fixes it.
if sys.platform == "win32" and not getattr(args_cli, "vulkan", False):
    extra = "--/app/vulkan=false"
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " " + extra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source"))
from triple_project.assets import CART_JOINT, TRIPLE_PENDULUM_CFG  # noqa: E402


@configclass
class WatchSceneCfg(InteractiveSceneCfg):
    """Scene entities must be declared as dataclass FIELDS.

    Assigning `.robot` onto an InteractiveSceneCfg instance after construction
    does not register it -- InteractiveScene walks the dataclass fields, so the
    robot would silently never be spawned.
    """

    robot: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 500.0, device=args_cli.device,
                                gravity=(0.0, 0.0, -9.80665))
    )
    sim.set_camera_view(eye=(2.2, 2.2, 1.4), target=(0.0, 0.0, 0.4))

    light_cfg = sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.95), intensity=900.0)
    light_cfg.func("/World/DomeLight", light_cfg)

    scene = InteractiveScene(WatchSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5))
    sim.reset()

    robot: Articulation = scene["robot"]
    names = list(robot.joint_names)
    cart_i = names.index(CART_JOINT)
    print("[watch] joints:", names)
    print("[watch] mode  :", args_cli.mode)

    def reset() -> None:
        jp = robot.data.default_joint_pos.clone()
        jv = torch.zeros_like(jp)
        # relative-angle offsets; only link1 is tilted, so the chain starts
        # straight but leaning
        jp[:, names.index("joint1")] = args_cli.tilt
        robot.write_joint_state_to_sim(jp, jv)
        robot.reset()

    reset()
    dt = sim.get_physics_dt()
    t = 0.0
    step = 0
    effort = torch.zeros((scene.num_envs, len(names)), device=robot.device)

    while simulation_app.is_running():
        if args_cli.mode == "sine":
            effort[:, cart_i] = args_cli.force * math.sin(2.0 * math.pi * args_cli.freq * t)
        elif args_cli.mode == "random":
            if step % 25 == 0:
                effort[:, cart_i] = args_cli.force * (2.0 * torch.rand(scene.num_envs, device=robot.device) - 1.0)
        else:
            effort[:, cart_i] = 0.0

        robot.set_joint_effort_target(effort)
        scene.write_data_to_sim()
        sim.step()
        t += dt
        step += 1
        scene.update(dt)

        # re-release every 6 s so it keeps repeating without babysitting
        if t > 6.0:
            t = 0.0
            reset()


if __name__ == "__main__":
    main()
    simulation_app.close()
