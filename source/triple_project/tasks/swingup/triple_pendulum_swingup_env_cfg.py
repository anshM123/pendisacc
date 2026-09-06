"""
Swing-up task: bring all three links from hanging to upright, and hold them.

This is a much harder problem than balance. The plant is underactuated by 3 (one
cart force, four DOFs), the target is an unstable equilibrium with lambda_max =
16.3 rad/s, and reaching it requires pumping energy in over several swings while
staying inside a 1.28 m rail.

Four design choices carry most of the weight:

0. EVERY per-step task reward is non-negative, so that surviving an episode is
   never worse than ending it. With signed rewards the policy learned to crash
   on purpose; see the RewardsCfg docstring for the arithmetic and the measured
   failure. This is the single change that made training seed-independent.

1. NO "link fallen" termination. Swing-up necessarily passes through large
   angles, so the balance task's termination would kill every useful episode.
   Only the time limit and the rail limit end an episode.

2. Starts are UNIFORM OVER THE CIRCLE, a reverse curriculum. Restricting them
   to below horizontal was tried and made the catch undiscoverable: every
   failing seed logged capture reward of essentially zero for 2000 iterations.
   The earlier objection to near-upright starts -- that they inflate the reward
   curve -- was really an objection to using reward as the metric, and
   evaluate.py now scores only from dead hang, so it no longer applies.

3. A separate "capture" bonus that requires upright AND slow. A pure height
   reward is maximised by whipping through the top at speed, which is the
   opposite of what we want.

A link-velocity penalty IS present, unlike the first version of this task. It
was left out on the reasoning that swinging fast is required, but with link
energy unbounded the policy converged on spinning the chain at ~62 rad/s per
link instead of ever stopping. The penalty is quadratic and small enough to be
free during pumping; see the note on the term.
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
        cart = ObsTerm(func=mdp.cart_state)
        link_sincos = ObsTerm(func=mdp.link_angles_sincos)
        link_vel = ObsTerm(func=mdp.link_vels)
        # Clamped to +-1. Under the old wrapper clipping the policy observed a
        # clipped action; clamping here keeps that distribution identical now
        # that the raw action reaches the manager, so removing clip_actions
        # changes the COST on std and nothing else.
        last_action = ObsTerm(func=mdp.last_action_clipped)

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
            # Uniform over the full circle, so ~17% of episodes start within
            # 30 degrees of upright and the CATCH is learned directly instead
            # of stumbled into. Every failing run logs capture reward of
            # essentially zero for its whole life; the catch is a narrow target
            # that dead-hang exploration reaches only by luck, which is what
            # made success seed-dependent.
            #
            # This was previously restricted to pi +- 1.5 for a good reason
            # that no longer applies: near-upright starts let the policy
            # harvest easy episodes and inflated the reward curve while the tip
            # never rose above -0.334. That was a MEASUREMENT failure -- reward
            # was the metric. evaluate.py now always scores from dead hang, so
            # an easier training distribution can no longer flatter the
            # reported number. If the policy learns only to balance and not to
            # swing up, the dead-hang success rate says so immediately.
            "position_range": (-math.pi, math.pi),
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
    """Every task term is NON-NEGATIVE. That is the load-bearing property.

    Isaac Lab scales each term by step_dt, so with the old signed terms a
    hanging episode accumulated
        (4.0 * -1 + 0.6 * -3) * 0.004 * 3000 = -69.6
    while driving off the rail cost a one-shot -250 * 0.004 = -1.0. Ending the
    episode was ~70x cheaper than living through it, so the optimal policy for a
    net that had not yet found pumping was to crash immediately -- and PPO found
    exactly that. Seed 2 spent 550 consecutive iterations ending 100% of
    episodes on the rail at a mean length of 63 of 3000 steps. Seed 1 escaped
    only because it was still exploring widely when it stumbled into a pump.
    Which seed worked was a coin flip, which is what "not reliable" meant.

    Weights are chosen so the SPAN of each term is unchanged (tip was 4.0 over
    [-1,1], a span of 8, so it is now 8.0 over [0,1]); only the offset moves,
    and an affine shift leaves the shaping gradient identical.
    """

    # primary: get the tip up. 0 hanging, +1 fully upright.
    tip = RewTerm(func=mdp.tip_height_pos, weight=8.0)
    # secondary: all links up, not just the tip (rules out folded configurations
    # that happen to put the tip high)
    upright = RewTerm(func=mdp.uprightness_pos, weight=3.6)
    # the catch: upright AND slow
    capture = RewTerm(func=mdp.upright_capture, weight=8.0)

    # ---- costs, all small enough that no survivable episode accumulates more
    # ---- negative reward than the one-shot termination penalty below
    # Centring, weak enough not to suppress pumping. At -0.10 the policy reached
    # upright in 100% of episodes yet HELD it in ~3%: it got up wherever it
    # happened to be and then drifted into the rail.
    cart_pos = RewTerm(func=mdp.cart_pos_l2, weight=-0.50)
    # The quadratic above is far too flat to act as a wall (-0.125/s at x=0.5).
    # This one is exactly zero over the +-0.50 m the swing-up actually uses and
    # rises steeply through the last 10 cm.
    rail_wall = RewTerm(func=mdp.cart_rail_margin, weight=-0.5,
                        params={"bound": 0.60, "margin": 0.10})
    # Explicitly reward staying slow while up, so holding beats re-swinging.
    cart_vel = RewTerm(func=mdp.cart_vel_l2, weight=-0.02)
    # Bound the link energy. This was deliberately ABSENT, on the reasoning
    # that penalising link speed suppresses the pumping swing-up needs. With
    # nothing bounding it, the policy found a rotational limit cycle instead:
    # measured on a stuck checkpoint, median sum(omega^2) = 11660, i.e. ~62
    # rad/s per link, a 46 m/s tip. It passes THROUGH upright 15.3% of the
    # time and never stops -- mean tip height 0, worth 8.0*0.49 = 3.9, and it
    # requires no precision at all. No capture width can compete with that,
    # because exp(-sum(omega^2)/vel_std^2) is annihilated at 11660 for any
    # sane vel_std; widening the ANGLE axis (0.25 -> 1.0) moved the capture
    # reward from 0.0000 to 0.0006 and changed nothing.
    #
    # Quadratic, so it separates the two regimes by itself:
    #   pumping   sum(omega^2) ~ 600    -> episodic cost  2.9   (vs ~235 earned)
    #   spinning  sum(omega^2) ~ 11660  -> episodic cost 56     (vs 48 earned)
    # a ~175x ratio, so pumping stays essentially free and spinning stops
    # paying for itself.
    #
    # This is also a hardware requirement, not only a shaping one: a 46 m/s tip
    # would destroy PETG links and their bearings.
    link_vel = RewTerm(func=mdp.link_vel_l2, weight=-0.0004)
    # -> -10.0 actual after the dt scaling. Sized to dominate the worst
    # accumulation of the shaping costs above (~-3.6 over a full episode) by a
    # clear margin, while staying small next to a successful episode's ~+235,
    # so it does not distort the value function.
    terminating = RewTerm(func=mdp.is_terminated, weight=-2500.0)
    # This term now sees the RAW policy output (no wrapper clipping), so it is
    # what bounds the action std. Sized to be negligible at a sane std and
    # punitive at a runaway one -- episodic cost is weight * E[a^2] * 12:
    #   std 0.5 -> -0.30 (0.1% of a successful episode's ~235)
    #   std 1.0 -> -0.60
    #   std 5.0 -> -15    (6%)
    #   std 47  -> -1325
    # so exploration is free and saturation is not.
    effort = RewTerm(func=mdp.action_l2, weight=-0.05)
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
