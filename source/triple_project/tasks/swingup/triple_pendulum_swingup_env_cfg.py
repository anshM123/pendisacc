"""
Swing-up task: bring all three links from hanging to upright, and hold them.

This is a much harder problem than balance. The plant is underactuated by 3 (one
cart force, four DOFs), the target is an unstable equilibrium with lambda_max =
16.3 rad/s, and reaching it requires pumping energy in over several swings while
staying inside a 1.28 m rail.

Three design choices carry most of the weight:

1. NO "link fallen" termination. Swing-up necessarily passes through large
   angles, so the balance task's termination would kill every useful episode.
   Only the time limit and the rail limit end an episode.

2. Starts are always BELOW HORIZONTAL. The first attempt sampled joint1 over
   the full circle, reasoning that this gives a curriculum for free. It did the
   opposite: the policy harvested almost all its reward from episodes that
   already started near the top, never needed to pump, and the rising reward
   curve measured better catching rather than swing-up. Measured outcome: max
   tip height -0.334, i.e. it never even reached horizontal, and 0.0% of time
   near upright.
   Restricting joint1 to pi +- 1.5 rad removes the easy episodes entirely, so
   pumping is the only way to earn the height and capture rewards.

3. A separate "capture" bonus that requires upright AND slow. A pure height
   reward is maximised by whipping through the top at speed, which is the
   opposite of what we want.

The link-velocity penalty from the balance task is deliberately absent: swinging
fast is required here, and penalising it directly suppresses the pumping motion.
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
from triple_project.assets import CART_JOINT, LINK_JOINTS, MAX_CART_SPEED, TRIPLE_PENDULUM_CFG


@configclass
class SwingUpSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.95), intensity=800.0),
    )


@configclass
class ActionsCfg:
    # Signed cart VELOCITY. The drive runs STEP/DIR into the A6's position
    # loop, so pulse rate -- i.e. commanded velocity -- is what the hardware
    # actually accepts. A force action would not be executable.
    cart_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[CART_JOINT], scale=MAX_CART_SPEED,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cart = ObsTerm(func=mdp.cart_state)
        link_sincos = ObsTerm(func=mdp.link_angles_sincos)
        link_vel = ObsTerm(func=mdp.link_vels)
        last_action = ObsTerm(func=mdp.last_action)

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
            "position_range": (-0.20, 0.20),
            "velocity_range": (-0.10, 0.10),
        },
    )
    # joints 2 and 3 stay near-straight so the arm is not tied in a knot at t=0
    reset_link1 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            # pi +- 1.5 rad -> always starts below horizontal, so there are no
            # free near-upright episodes to harvest
            "position_range": (math.pi - 1.5, math.pi + 1.5),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_links23 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3"]),
            "position_range": (-0.4, 0.4),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    # primary: get the tip up. -1 hanging, +1 fully upright.
    tip = RewTerm(func=mdp.tip_height, weight=4.0)
    # secondary: all links up, not just the tip (rules out folded configurations
    # that happen to put the tip high)
    upright = RewTerm(func=mdp.uprightness, weight=0.6)
    # the catch: upright AND slow
    capture = RewTerm(func=mdp.upright_capture, weight=8.0)
    # stay on the rail.
    #
    # NOTE on scale: Isaac Lab's RewardManager multiplies EVERY term by step_dt
    # (reward_manager.py: value = func * weight * dt). With dt = 4 ms a
    # one-shot termination weight of -5 is worth -0.02, i.e. nothing next to a
    # capture reward that accumulates to ~1.2 per episode. Measured effect:
    # 83% of episodes ended by running off the rail, because doing so was free.
    # A one-shot penalty therefore needs weight ~ desired_cost / dt.
    # Only a whisper of centring. There is ~1.2 m of usable rail and pumping a
    # triple pendulum REQUIRES using it; a strong centring penalty pays the
    # policy to stand still, which is the opposite of swing-up.
    cart_pos = RewTerm(func=mdp.cart_pos_l2, weight=-0.10)
    terminating = RewTerm(func=mdp.is_terminated, weight=-250.0)   # -> -1.0 actual
    # keep the control smooth, but weakly -- pumping needs large forces
    effort = RewTerm(func=mdp.action_l2, weight=-0.004)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ~1.2 m of usable travel -> +-0.60 m, with a little margin to the hard stop
    cart_bounds = DoneTerm(func=mdp.cart_out_of_bounds, params={"bound": 0.60})
    # NOTE: deliberately no `fallen` term -- see module docstring.


@configclass
class TriplePendulumSwingUpEnvCfg(ManagerBasedRLEnvCfg):
    scene: SwingUpSceneCfg = SwingUpSceneCfg(num_envs=4096, env_spacing=2.5, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 12.0     # long enough to pump up and then hold
        self.viewer.eye = (2.5, 2.5, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.sim.dt = 1.0 / 500.0        # 250 Hz control, as in the balance task
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.80665)


@configclass
class TriplePendulumSwingUpEnvCfg_PLAY(TriplePendulumSwingUpEnvCfg):
    """Small version, and always starting from HANGING so the swing-up is visible."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.2
        # framed for watching: slightly off-axis, high enough to see the tip at
        # full extension (~0.87 m above the rail) and wide enough for the cart's
        # +-0.45 m travel
        # close enough that the machine fills the frame; the tip reaches ~0.87 m
        # above the rail and the cart travels about +-0.45 m
        self.viewer.eye = (1.05, -1.55, 0.80)
        self.viewer.lookat = (0.0, 0.0, 0.30)
        # start hanging: joint1 = pi in relative coordinates puts every link down
        self.events.reset_link1.params["position_range"] = (math.pi - 0.05, math.pi + 0.05)
        self.events.reset_link1.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_links23.params["position_range"] = (-0.02, 0.02)
        self.events.reset_links23.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_cart.params["position_range"] = (0.0, 0.0)
        self.events.reset_cart.params["velocity_range"] = (0.0, 0.0)
