"""Swap the ACTION INTERFACE without touching anything else.

The claim under test is that equivalence classes in physical-parameter space
belong to the pair (plant, interface) and not to the plant alone. Testing it
requires changing exactly one thing: the physical variable the policy commands.
Everything else -- plant, servo lag, transport delay, force clamp, speed limit,
observation, reward, PPO hyperparameters, episode length -- must be identical,
or a difference in transfer can be blamed on the difference in setup instead.

  velocity   cart velocity command through a stiff inner loop.
             F = clamp(kv * (v_cmd - v_cart), +-F_clamp). This is the frozen
             baseline and what the STEP/DIR hardware actually accepts.

  force      cart force command straight through. The inner loop is REMOVED
             (kv = 0), so the policy's output is the whole cart force.

The force arm keeps the same clamp (349.5 N peak) and the same speed limit,
and its action is scaled to the clamp -- so it can command at least as much
force as the velocity loop could ever ask for. Handicapping its authority
would confound interface with authority, which is precisely the confound that
sank the c=8 training arm in PREREGISTRATION_H3. It is not repeated here.

What is NOT identical, and cannot be:

  * the units of the action. Both are normalised to +-1 before the reward sees
    them, so the action_l2 penalty is applied to comparable numbers, but a
    velocity command of 1.0 and a force command of 1.0 are different physical
    requests. There is no way to remove this and still change the interface;
    it is inherent to the question.
  * the closed-loop bandwidth of the cart itself. The velocity loop imposes a
    2 ms first-order response on the cart; the force interface has none. That
    IS the interface difference, not a confound to be removed.
"""

from __future__ import annotations

INTERFACES = ("velocity", "force")


def action_term(env_cfg):
    """The single action term, whichever interface is installed."""
    acts = env_cfg.actions
    for name in ("cart_velocity", "cart_effort"):
        term = getattr(acts, name, None)
        if term is not None:
            return term
    raise AttributeError("no cart action term on this env cfg")


def apply_interface(env_cfg, name: str) -> dict:
    """Install the named interface. Returns a record of what was changed."""
    if name not in INTERFACES:
        raise ValueError("unknown interface %r; expected one of %s" % (name, INTERFACES))
    if name == "velocity":
        return {"interface": "velocity", "changed": False}

    from triple_project.actuators import LaggedJointEffortActionCfg
    from triple_project.assets import CART_JOINT, MAX_CART_SPEED
    from triple_project.assets.triple_pendulum import DRIVE_CLAMP_FORCE

    old = action_term(env_cfg)
    env_cfg.actions.cart_effort = LaggedJointEffortActionCfg(
        asset_name="robot",
        joint_names=[CART_JOINT],
        # peak drive force: at least the authority the velocity loop had
        scale=DRIVE_CLAMP_FORCE,
        # identical servo model to the velocity arm
        time_constant_s=float(old.time_constant_s),
        delay_s=float(old.delay_s),
        deadband=float(old.deadband),
        order=int(old.order),
        zeta=float(old.zeta),
        omega_n=old.omega_n,
        clip={CART_JOINT: (-DRIVE_CLAMP_FORCE, DRIVE_CLAMP_FORCE)},
    )
    del env_cfg.actions.cart_velocity

    # remove the inner velocity loop: the policy's force is now the ONLY force
    cart = env_cfg.scene.robot.actuators["cart"]
    cart.damping = 0.0
    cart.stiffness = 0.0
    cart.effort_limit_sim = DRIVE_CLAMP_FORCE
    cart.velocity_limit_sim = MAX_CART_SPEED    # the motor still cannot overspeed

    return {"interface": "force", "changed": True,
            "scale_N": float(DRIVE_CLAMP_FORCE),
            "clamp_N": float(DRIVE_CLAMP_FORCE),
            "speed_limit_m_s": float(MAX_CART_SPEED),
            "kv_removed": True,
            "time_constant_s": float(old.time_constant_s),
            "delay_s": float(old.delay_s)}
