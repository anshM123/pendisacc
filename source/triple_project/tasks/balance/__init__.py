"""Registers the triple pendulum balance task."""

import gymnasium as gym

from . import agents
from .triple_pendulum_env_cfg import (
    TriplePendulumBalanceEnvCfg,
    TriplePendulumBalanceEnvCfg_PLAY,
)

gym.register(
    id="TIP-Balance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriplePendulumBalanceEnvCfg,
        "rsl_rl_cfg_entry_point": agents.TriplePendulumBalancePPORunnerCfg,
    },
)

gym.register(
    id="TIP-Balance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriplePendulumBalanceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.TriplePendulumBalancePPORunnerCfg,
    },
)
