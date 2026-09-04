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
# COMMAND PATH: Teensy -> STEP/DIR -> CNC4PC C34SOA6-RS -> A6-RS POSITION mode.
# Pulse frequency advances the commanded position, so the natural real-time
# command is a signed cart VELOCITY. A direct torque action is NOT executable
# on this path -- see configs/robot/hardware.yaml for why STEP/DIR beat direct
# RS485 torque control (Modbus framing is too slow for a per-millisecond loop).
#
#   a_t -> v_cmd -> pulse rate + DIR -> position reference -> A6 inner loop
#       -> motor torque (saturated) -> GT2 -> cart force
#
# The inner loop is modelled first-order as a stiff velocity servo under a force
# clamp:   F = clamp( D * (v_cmd - v_cart), +-F_clamp )
#
# SPEED. omega * r: 4.0 m/s at rated 3000 rpm, 8.0 m/s at peak 6000 rpm. The
# action is scaled to the rated figure. The 4 MHz pulse ceiling is not the
# binding limit -- motor speed is.
MAX_CART_SPEED = 4.0

# FORCE. F = tau / r: rated 1.27 N.m -> 99.7 N, peak 4.45 N.m -> 349.5 N.
# The clamp is the drive's current limit; it should rarely bind because the
# velocity loop, not the policy, decides the force.
DRIVE_CLAMP_FORCE = 349.5
RATED_CART_FORCE = 99.7

# Inner velocity-loop gain [N.s/m]. ~2 ms loop time constant against the
# 0.79 kg effective translating mass (cart + carriages + reflected rotor).
# ASSUMED: replace once the A6's position-loop bandwidth is measured.
CART_VELOCITY_GAIN = 400.0

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
        # velocity servo standing in for the A6 position loop: no stiffness,
        # damping = velocity-loop gain, force bounded by the drive's clamp
        "cart": ImplicitActuatorCfg(
            joint_names_expr=[CART_JOINT],
            effort_limit_sim=DRIVE_CLAMP_FORCE,
            velocity_limit_sim=MAX_CART_SPEED,
            stiffness=0.0,
            damping=CART_VELOCITY_GAIN,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=LINK_JOINTS,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
