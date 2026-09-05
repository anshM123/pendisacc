"""
Balance task: hold all three links upright, starting near upright.

This is deliberately the *easy* task. Its purpose is to prove PPO can control the
plant at all and to establish the training protocol that later gets frozen. Once
it trains reliably, the reward and hyperparameters are frozen and only simulator
physics is varied -- that is the whole experimental design.

Timing. lambda_max = 16.3 rad/s means a 61 ms divergence time constant, so the
control rate has to resolve it: sim dt = 2 ms with decimation 2 gives 250 Hz
control, about 15 control steps per time constant. Do not raise dt for speed
without re-checking that number.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import triple_project.mdp as mdp
from triple_project.actuators import LaggedJointVelocityActionCfg
from triple_project.assets import CART_JOINT, LINK_JOINTS, MAX_CART_SPEED, TRIPLE_PENDULUM_CFG


@configclass
class TriplePendulumSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.95), intensity=800.0),
    )


@configclass
class ActionsCfg:
    """The cart force is the only actuation, exactly as on the real machine."""

    # Signed cart VELOCITY. The drive runs STEP/DIR into the A6's position
    # loop, so pulse rate -- i.e. commanded velocity -- is what the hardware
    # actually accepts. A force action would not be executable.
    # Servo response is NOT instantaneous. time_constant_s is a deliberately
    # pessimistic 100 ms -- about 1.5x the pendulum's own 64 ms divergence time
    # constant -- pending measurement of the real drive. See hardware.yaml.
    cart_velocity = LaggedJointVelocityActionCfg(
        asset_name="robot", joint_names=[CART_JOINT], scale=MAX_CART_SPEED,
        use_default_offset=False,
        time_constant_s=0.100,
        delay_s=0.0,
        # hard bound in PHYSICAL units: the drive cannot be commanded past its
        # rated speed no matter what the policy emits
        clip={CART_JOINT: (-MAX_CART_SPEED, MAX_CART_SPEED)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """12-D observation.

        Nothing simulator-only appears here: no friction coefficients, no
        simulator id, no mass perturbation, no latency. The real robot cannot
        measure those, so a policy allowed to see them would not transfer and
        the whole comparison would be void.
        """

        cart = ObsTerm(func=mdp.cart_state)               # 2
        link_sincos = ObsTerm(func=mdp.link_angles_sincos)  # 6
        link_vel = ObsTerm(func=mdp.link_vels)             # 3
        last_action = ObsTerm(func=mdp.last_action)        # 1

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[CART_JOINT]),
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.10, 0.10),
        },
    )
    reset_links = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LINK_JOINTS)),
            # relative-angle offsets; small, so the chain starts near upright
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    # primary task
    upright = RewTerm(func=mdp.uprightness, weight=1.0)
    tip = RewTerm(func=mdp.tip_height, weight=2.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # scaled by step_dt like every other term, so -10 was worth -0.04;
    # see the note in the swing-up config
    terminating = RewTerm(func=mdp.is_terminated, weight=-250.0)
    # shaping -- keep the cart near centre and the motion calm
    cart_pos = RewTerm(func=mdp.cart_pos_l2, weight=-0.5)
    cart_vel = RewTerm(func=mdp.cart_vel_l2, weight=-0.01)
    link_vel = RewTerm(func=mdp.link_vel_l2, weight=-0.005)
    effort = RewTerm(func=mdp.action_l2, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_bounds = DoneTerm(func=mdp.cart_out_of_bounds, params={"bound": 0.55})
    fallen = DoneTerm(func=mdp.any_link_fallen, params={"max_tilt": 0.7})


@configclass
class TriplePendulumBalanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: TriplePendulumSceneCfg = TriplePendulumSceneCfg(
        num_envs=4096, env_spacing=2.5, clone_in_fabric=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (2.5, 2.5, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.dt = 1.0 / 500.0          # 2 ms -> 250 Hz control with decimation 2
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.80665)


@configclass
class TriplePendulumBalanceEnvCfg_PLAY(TriplePendulumBalanceEnvCfg):
    """Small, well-spaced version for watching the motion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
