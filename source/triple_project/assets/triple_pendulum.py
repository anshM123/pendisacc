"""Articulation config for the CAD-derived triple pendulum.

Masses, inertias and lengths all come from the generated USD, which comes from
configs/robot/triple_pendulum_params.yaml, which comes from CAD. Nothing here
restates a physical number.

The three revolute joints are PASSIVE. They are given an actuator group only so
Isaac Lab has something to attach to; stiffness and damping are zero and the
effort limit is zero, so no torque is ever applied to them. Joint friction is
applied separately from the parameter vector xi, never baked in here.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
USD_PATH = os.path.join(_ROOT, "assets", "triple_pendulum", "usd", "triple_pendulum.usd")

CART_JOINT = "cart_slide"
LINK_JOINTS = ["joint1", "joint2", "joint3"]

# ---------------------------------------------------------------------------
# Drivetrain, FROZEN: StepperOnline A6-RS400H2A1-M17 (400 W) driving the cart
# through a GT2 2 mm / 40 T pulley, r = 0.0127324 m.
#
# COMMAND MODE: direct torque. Teensy -> Waveshare RS485 -> A6-RS in torque mode
# (C00.00 = 2, internal digital reference C03.41, writable during operation).
# The CNC4PC C34SOA6-RS step/dir path is bypassed. So the RL action maps to
# motor torque -> cart force, which is exactly the B*F_cart term of the
# analytical model:  a_t -> tau_cmd -> F_cart = tau / r.
#
# FORCE. F = tau / r:
#   rated 1.27 N.m -> 99.7 N        peak 4.45 N.m -> 349.5 N
# The policy is scaled to the RATED force, not the peak. Peak torque is a
# short-duration rating; swing-up pumping is a sustained repeated demand, so
# training against 350 N would hand the policy authority the drive cannot hold.
# TODO: duty-cycle/thermal model permitting brief excursions toward peak.
MAX_CART_FORCE = 100.0        # action scale, ~rated
DRIVE_CLAMP_FORCE = 349.5     # drive current limit; should not normally bind

# omega * r: 4.0 m/s at rated 3000 rpm, 8.0 m/s at peak 6000 rpm
MAX_CART_SPEED = 4.0

TRIPLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={CART_JOINT: 0.0, "joint1": 0.0, "joint2": 0.0, "joint3": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # pure torque source: no position or velocity loop in the drive, so no
        # stiffness and no damping. The only limit is the current clamp.
        "cart": ImplicitActuatorCfg(
            joint_names_expr=[CART_JOINT],
            effort_limit_sim=DRIVE_CLAMP_FORCE,
            velocity_limit_sim=MAX_CART_SPEED,
            stiffness=0.0,
            damping=0.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=LINK_JOINTS,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
