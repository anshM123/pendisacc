"""PPO configuration for the balance task.

Once balance trains reliably this gets FROZEN. Every scientific comparison in the
project varies simulator physics only; if the learning setup drifts between runs,
differences in transfer cannot be attributed to the simulator.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TriplePendulumBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 1200
    save_interval = 50
    # see the note in the swing-up config: bounds mdp.action_l2
    clip_actions = 1.0
    experiment_name = "tip_balance"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # 1.0 meant ~+-40 N of exploration noise on 0.5 kg of moving mass, which
        # kicked the plant far outside the balance basin and destabilised training
        init_noise_std=0.4,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # wider than cartpole's [32, 32]: 3 coupled unstable modes is a much
        # harder control problem than 1
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,   # lowered: a growing std was what ran away
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # CRITICAL: control runs at 250 Hz, so gamma sets the horizon in SECONDS.
        # gamma=0.99 -> 1/(1-g) = 100 steps = 0.4 s, far too myopic for an 8 s
        # balance task that also has to keep the cart centred. 0.997 -> ~1.3 s.
        gamma=0.997,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
