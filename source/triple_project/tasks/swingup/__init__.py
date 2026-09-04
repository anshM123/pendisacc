"""Registers the triple pendulum swing-up task."""

import gymnasium as gym

from . import agents
from .triple_pendulum_swingup_env_cfg import (
    TriplePendulumSwingUpEnvCfg,
    TriplePendulumSwingUpEnvCfg_PLAY,
)

gym.register(
    id="TIP-SwingUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriplePendulumSwingUpEnvCfg,
        "rsl_rl_cfg_entry_point": agents.TriplePendulumSwingUpPPORunnerCfg,
    },
)

gym.register(
    id="TIP-SwingUp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriplePendulumSwingUpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.TriplePendulumSwingUpPPORunnerCfg,
    },
)
