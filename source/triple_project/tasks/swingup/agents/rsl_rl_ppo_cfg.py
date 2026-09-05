"""PPO config for swing-up.

Differences from balance, and why:
  * gamma 0.999 -> ~4 s horizon at 250 Hz. Swing-up needs the policy to value a
    payoff several swings away; the balance value of 0.997 (1.3 s) is too short
    to connect "pump now" with "upright later".
  * more exploration noise than balance -- it has to discover pumping at all.
  * longer rollouts and more iterations; this is a much harder problem.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TriplePendulumSwingUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 3000
    save_interval = 100
    # NO wrapper-level action clipping, deliberately.
    #
    # RslRlVecEnvWrapper clips before env.step (vecenv_wrapper.py:153), and
    # ActionManager then stores that already-clipped value as
    # action_manager.action -- which is exactly what mdp.action_l2 reads. So
    # with clip_actions set, the effort penalty is blind to any std above the
    # clip while the entropy bonus keeps paying for more. std then has
    # unbounded benefit and zero cost, and there is no restoring force at large
    # std: whichever way the balance tips, it runs away. Measured on identical
    # configs differing only in entropy_coef: 0.003 collapsed to std 0.00,
    # 0.006 exploded to std 47.2 (at which point every sampled action saturates
    # and the policy is training as a bang-bang controller). A knife edge, not
    # a tuning range.
    #
    # Physics stays safe without it: the action term carries a clip in PHYSICAL
    # units (+-MAX_CART_SPEED), applied to _processed_actions inside
    # JointAction.process_actions, so the commanded velocity is bounded no
    # matter what the policy emits. Removing the wrapper clip only restores the
    # quadratic cost on std, which gives a stable fixed point.
    clip_actions = None
    experiment_name = "tip_swingup"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Raised from 0.003. With clip_actions the old runaway (std 0.8 -> 2.52)
        # cannot recur, and 0.003 was collapsing too early instead: seed 1 hit
        # std 0.00 by iteration 1500 and its reward then drifted 8.06 -> 6.97,
        # with no exploration left to recover.
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
