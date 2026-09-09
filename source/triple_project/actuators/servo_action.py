"""
A cart-velocity action that does NOT respond instantly.

The A6 drive's inner position loop has finite bandwidth and the STEP/DIR path
carries transport delay. Modelling it as instantaneous flatters the policy: it
gets to change cart velocity arbitrarily fast, which is exactly the sort of
hidden actuator advantage that does not survive contact with hardware.

Two distinct effects, deliberately separated because they degrade control very
differently:

  * transport DELAY  -- a pure dead time before a command has any effect.
    Ruinous for an unstable plant: nothing the controller does is visible for
    `delay_s`, so it is flying blind over that window.
  * loop LAG         -- a first-order filter on the reference. The drive starts
    responding immediately but reaches the commanded velocity only after
    ~`time_constant_s`.

        v_ref[k] = v_ref[k-1] + (dt / (dt + tau)) * (v_cmd[k - D] - v_ref[k-1])

The filter acts on the COMMAND, not on the force. The inner loop still delivers
whatever force it can (up to the drive's clamp) to track the filtered reference,
which is what a real servo does -- its current loop saturates at peak torque, it
does not go weak. Lowering the velocity gain instead would wrongly starve the
cart of force as well as slowing it.

Scale for judging the numbers: the upright equilibrium has lambda_max = 15.5
rad/s, i.e. a 64 ms divergence time constant. Actuator lag comparable to that
should be expected to make the task hard or impossible, and if so that is a
hardware requirement, not a tuning problem.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class LaggedJointVelocityAction(JointVelocityAction):
    """Joint velocity action with transport delay and first-order loop lag."""

    cfg: "LaggedJointVelocityActionCfg"

    def __init__(self, cfg: "LaggedJointVelocityActionCfg", env) -> None:
        super().__init__(cfg, env)
        dt = env.step_dt
        self._dt = dt
        # first-order smoothing coefficient; tau = 0 reproduces the instant case
        tau = float(cfg.time_constant_s)
        self._alpha = 1.0 if tau <= 0.0 else dt / (dt + tau)
        # Second-order option, for asking whether it is the actuator's ORDER
        # that matters rather than its speed. omega_n is given explicitly so the
        # comparison can be bandwidth- or rise-time-matched against the
        # first-order case instead of naively setting omega_n = 1/tau, which
        # yields only 75% of the first-order -3 dB bandwidth and is therefore
        # simply a slower actuator wearing a different label.
        self._order = int(getattr(cfg, "order", 1))
        self._zeta = float(getattr(cfg, "zeta", 0.9))
        wn = getattr(cfg, "omega_n", None)
        self._wn = float(wn) if wn else (1.0 / tau if tau > 0 else 0.0)
        # transport delay, rounded to whole control steps
        self._delay_steps = int(round(float(cfg.delay_s) / dt))
        n = self.num_envs
        d = self.action_dim
        self._filtered = torch.zeros((n, d), device=self.device)
        self._filtered_dot = torch.zeros((n, d), device=self.device)
        if self._delay_steps > 0:
            self._queue = torch.zeros((self._delay_steps + 1, n, d), device=self.device)
            self._head = 0

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._filtered.zero_()
            self._filtered_dot.zero_()
            if self._delay_steps > 0:
                self._queue.zero_()
        else:
            self._filtered[env_ids] = 0.0
            self._filtered_dot[env_ids] = 0.0
            if self._delay_steps > 0:
                self._queue[:, env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        cmd = self._processed_actions
        if self.cfg.deadband > 0.0:
            # commands below the deadband produce no motion at all
            cmd = torch.sign(cmd) * torch.clamp(cmd.abs() - self.cfg.deadband, min=0.0)
        if self._delay_steps > 0:
            # push the newest command, pop the one from `delay_steps` ago
            self._queue[self._head] = cmd
            self._head = (self._head + 1) % self._queue.shape[0]
            cmd = self._queue[self._head]
        if self._order == 2:
            acc = self._wn ** 2 * (cmd - self._filtered) - 2.0 * self._zeta * self._wn * self._filtered_dot
            self._filtered_dot = self._filtered_dot + self._dt * acc
            self._filtered = self._filtered + self._dt * self._filtered_dot
        else:
            self._filtered += self._alpha * (cmd - self._filtered)
        self._processed_actions = self._filtered

    @property
    def servo_info(self) -> dict:
        return {
            "time_constant_s": float(self.cfg.time_constant_s),
            "delay_s": float(self.cfg.delay_s),
            "delay_steps": int(self._delay_steps),
            "alpha": float(self._alpha),
            "control_dt_s": float(self._dt),
        }


@configclass
class LaggedJointVelocityActionCfg(actions_cfg.JointVelocityActionCfg):
    """Cart velocity command with a realistic (or deliberately pessimistic) servo."""

    class_type: type[ActionTerm] = LaggedJointVelocityAction

    time_constant_s: float = MISSING
    """First-order lag of the drive's inner loop reaching the commanded velocity."""

    delay_s: float = 0.0
    """Pure transport dead time: Teensy -> STEP/DIR -> drive acting on it."""

    deadband: float = 0.0
    """Commanded velocity below this magnitude produces no motion [m/s]."""

    order: int = 1
    """1 = first-order lag, 2 = second-order. A MODEL-FORM change, not a
    parameter one: no amount of randomising time_constant_s covers it."""

    zeta: float = 0.9
    """Damping ratio, second order only."""

    omega_n: float | None = None
    """Natural frequency [rad/s], second order only. Set explicitly to match
    bandwidth or rise time against a first-order reference; defaults to 1/tau,
    which does NOT match either."""
