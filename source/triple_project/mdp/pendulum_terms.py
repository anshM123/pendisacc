"""
Observation, reward and termination terms for the triple pendulum.

EVERY angle here is converted out of Isaac's relative-joint convention into
ABSOLUTE angles from the upward vertical before use. See dynamics/conventions.py
for why that distinction is load-bearing. In particular the policy must see
absolute angles: "link 3 is upright" is a statement about its absolute
orientation, and a policy fed relative angles would have to infer the sum itself.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

CART_JOINT = "cart_slide"
LINK_JOINTS = ("joint1", "joint2", "joint3")

_CACHE: dict[int, tuple[int, list[int]]] = {}


def _indices(asset: Articulation) -> tuple[int, list[int]]:
    """(cart joint index, [link joint indices in chain order]) -- cached per asset."""
    key = id(asset)
    if key not in _CACHE:
        names = list(asset.joint_names)
        _CACHE[key] = (names.index(CART_JOINT), [names.index(n) for n in LINK_JOINTS])
    return _CACHE[key]


def abs_link_angles(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute link angles from vertical, shape (N, 3). 0 == upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, link = _indices(asset)
    return torch.cumsum(asset.data.joint_pos[:, link], dim=-1)


def abs_link_vels(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute link angular velocities, shape (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, link = _indices(asset)
    return torch.cumsum(asset.data.joint_vel[:, link], dim=-1)


# ---------------------------------------------------------------- observations


def last_action_clipped(
    env: ManagerBasedRLEnv, clip: float = 1.0, action_name: str | None = None
) -> torch.Tensor:
    """The previous action, clamped to +-`clip`.

    The swing-up task deliberately runs without RslRlVecEnvWrapper's action
    clipping, so that mdp.action_l2 can see -- and charge for -- the raw policy
    output; see the note in the swing-up PPO config. That would otherwise also
    change what the policy OBSERVES, since last_action reads the same
    action_manager.action. Clamping here keeps the observation distribution
    exactly as it was under wrapper clipping, so only the cost changes.
    """
    action = env.action_manager.action if action_name is None else env.action_manager.get_term(action_name).raw_actions
    return torch.clamp(action, -clip, clip)



def link_angles_sincos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sin/cos of the three absolute link angles, shape (N, 6).

    sin/cos rather than the raw angle so the policy never sees the +-pi wrap
    discontinuity, which matters a lot once swing-up is enabled.
    """
    th = abs_link_angles(env, asset_cfg)
    return torch.cat([torch.sin(th), torch.cos(th)], dim=-1)


def link_vels(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return abs_link_vels(env, asset_cfg)


def cart_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cart position and velocity, shape (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cart, _ = _indices(asset)
    return torch.stack([asset.data.joint_pos[:, cart], asset.data.joint_vel[:, cart]], dim=-1)


# -------------------------------------------------------------------- rewards


def uprightness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of cos(theta_i) over the three links, in [-3, 3]; 3 == fully upright.

    Smooth, bounded, and has no preferred direction of tilt, so it does not bias
    the policy toward a particular recovery strategy.
    """
    return torch.cos(abs_link_angles(env, asset_cfg)).sum(dim=-1)


def tip_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Height of the distal tip above the first pivot, normalised to [-1, 1].

    Uses the actual link lengths, so it stays correct if the CAD changes.
    """
    th = abs_link_angles(env, asset_cfg)
    lengths = _link_lengths(env, asset_cfg)
    return (lengths * torch.cos(th)).sum(dim=-1) / lengths.sum()


_LENGTH_CACHE: dict[int, torch.Tensor] = {}


def _link_lengths(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    key = id(env)
    if key not in _LENGTH_CACHE:
        import os

        import yaml

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(root, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8") as fh:
            p = yaml.safe_load(fh)
        vals = []
        for name in ("link1", "link2", "link3"):
            e = p["bodies"][name]
            vals.append(float(e.get("length", 2.0 * e["l_com"])))
        _LENGTH_CACHE[key] = torch.tensor(vals, device=env.device, dtype=torch.float32)
    return _LENGTH_CACHE[key]


def tip_height_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """`tip_height` shifted into [0, 1]: 0 hanging, 1 upright.

    This shift is NOT cosmetic. Isaac Lab multiplies every reward by step_dt, so
    with the signed version a hanging episode paid
        4.0 * (-1) * 0.004 * 3000 steps = -48
    while ending the episode by driving into the rail cost only -250*0.004 = -1.
    Terminating was therefore ~50x cheaper than surviving, and PPO found that.
    Measured: seed 2 ended 100% of episodes on the rail for 550 consecutive
    iterations, mean episode length 63 of 3000 steps, reward pinned at -2.0.
    Seed 1 escaped only because its exploration noise was still wide (std 0.17)
    when it stumbled into pumping -- i.e. reliability was down to the seed.

    An affine shift leaves the gradient toward upright untouched; it changes
    only what a TERMINATED episode is worth relative to one that survives, which
    is exactly the quantity that was wrong. Every per-step task reward is now
    non-negative, so timing out always beats crashing.
    """
    return 0.5 * (1.0 + tip_height(env, asset_cfg))


def uprightness_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """`uprightness` shifted into [0, 1]. Same reasoning as `tip_height_pos`."""
    return (3.0 + uprightness(env, asset_cfg)) / 6.0


def cart_rail_margin(
    env: ManagerBasedRLEnv,
    bound: float = 0.60,
    margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft wall: 0 until `margin` from the rail limit, rising to 1 at `bound`.

    The quadratic centring term is far too flat to act as a wall -- at x = 0.5 m
    it is worth -0.125/s, which a policy mid-swing simply pays. This term is
    zero over the whole region the swing-up actually uses and only bites in the
    last 10 cm, so it discourages the excursion without suppressing pumping.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cart, _ = _indices(asset)
    over = (torch.abs(asset.data.joint_pos[:, cart]) - (bound - margin)) / margin
    return torch.square(torch.clamp(over, 0.0, 1.0))


def upright_capture(
    env: ManagerBasedRLEnv,
    angle_std: float = 1.0,
    vel_std: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus for being upright AND slow -- the "catch" at the top of a swing-up.

    Product of two Gaussians so it only pays out when both conditions hold. A
    pure height reward is maximised by whipping through the top at speed, which
    is exactly the wrong behaviour; this term is what makes the policy stop.

    WIDTHS. These were 0.25 and 3.0, which made the term unreachable by
    anything but luck. Evaluated on the state the policy is actually in while
    swinging through the top:

        each link at   |   old (0.25, 3.0)   |   now (1.0, 6.0)
        20 deg, 2 rad/s|        0.128        |       0.598
        30 deg, 3 rad/s|        0.010        |       0.316
        45 deg, 4 rad/s|        0.00014      |       0.109

    At 0.00014 there is no gradient telling the policy to slow down until it is
    already within ~20 degrees of upright -- a region it can only enter by
    accident. Measured consequence on two seeds of an otherwise identical
    config: seed 1 stumbled into it near iteration 400 and reached 100% success;
    seed 2 never found it in 2000 iterations, logged capture reward of exactly
    0.0000 throughout, and settled into a limit cycle -- swinging the tip up
    through the top and back down forever, hold fraction pinned at 14.0% and
    mean tip height -0.02. Which seed worked was luck, again.

    Widening changes only the BASIN, not the optimum: the product is still
    maximised at zero tilt and zero rate, so what the policy is asked to do is
    unchanged; there is now a continuous path in from where it starts.
    """
    th = abs_link_angles(env, asset_cfg)
    w = abs_link_vels(env, asset_cfg)
    tilt = torch.sum(1.0 - torch.cos(th), dim=-1)          # 0 upright, 6 hanging
    speed = torch.sum(torch.square(w), dim=-1)
    return torch.exp(-tilt / angle_std) * torch.exp(-speed / (vel_std ** 2))


def link_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(abs_link_vels(env, asset_cfg)), dim=-1)


def cart_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cart, _ = _indices(asset)
    return torch.square(asset.data.joint_pos[:, cart])


def cart_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cart, _ = _indices(asset)
    return torch.square(asset.data.joint_vel[:, cart])


# --------------------------------------------------------------- terminations


def any_link_fallen(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when any link exceeds `max_tilt` radians from vertical.

    Balance-task only. The swing-up task must NOT use this, since swing-up
    necessarily passes through large angles.
    """
    th = abs_link_angles(env, asset_cfg)
    return (torch.abs(torch.atan2(torch.sin(th), torch.cos(th))) > max_tilt).any(dim=-1)


def cart_out_of_bounds(
    env: ManagerBasedRLEnv,
    bound: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cart, _ = _indices(asset)
    return torch.abs(asset.data.joint_pos[:, cart]) > bound
