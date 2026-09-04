"""
Build step: convert the generated URDF to USD. Nothing else.

This MUST run in its own process. Doing the conversion and then creating a
SimulationContext in the same process crashes Isaac Sim with an access violation:
the URDF importer pulls in omni.kit.tool.asset_importer, and SimulationContext's
stage re-init renders while those extensions are being torn down. Keeping the
conversion as a separate build step avoids the whole class of problem.
"""

from __future__ import annotations

import argparse
import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "assets", "triple_pendulum", "urdf", "triple_pendulum.urdf")
USD_DIR = os.path.join(ROOT, "assets", "triple_pendulum", "usd")


def main() -> int:
    cfg = UrdfConverterCfg(
        asset_path=URDF,
        usd_dir=USD_DIR,
        usd_file_name="triple_pendulum.usd",
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        force_usd_conversion=True,
        # target_type="none" -> PhysX drive stiffness and damping are set to 0,
        # which is what makes the three pendulum joints genuinely passive.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    conv = UrdfConverter(cfg)
    print("[usd] wrote", conv.usd_path)
    print("[usd] exists:", os.path.exists(conv.usd_path))
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
