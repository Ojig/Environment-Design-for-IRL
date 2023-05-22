from ray.rllib.policy.policy import PolicySpec

from config.base_config import EnvAlgoConfig


class IRLConfig(EnvAlgoConfig):
    BASE = dict(
        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.99
        ),
        n_expert_rollouts=10,
        perf_top_percent=1.,
        n_samples_saved=1000,
        disc_batchsize=2048,
        gpu=-1,
        n_discriminator_train_step=16,
        n_remember=20,
        n_rl_iter=1,
        entropy_coeff=0.0097,
        n_reward_funcs=10
    )

    MazeED__AIRL = dict(
        n_expert_rollouts=50,
        n_reward_funcs=3,

        entropy_coeff=0.005,
        n_remember=5,
        n_samples_saved=200 * 42,
        n_rl_iter=1,
        disc_batchsize=128, #256
        n_discriminator_train_step=8,

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.995
        ),

    )

    MazeED__MULTI_AIRL = dict(
        n_expert_rollouts=50,
        n_envs=5,
        n_reward_funcs=3,
        entropy_coeff=5e-3,
        n_remember=12,
        n_samples_saved=200 * 42,
        n_rl_iter=1,
        disc_batchsize=512, # 64
        n_discriminator_train_step=64, #8

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.995
        ),
    )

    HopperED__AIRL = dict(
        n_expert_rollouts=50,

        perf_top_percent=1.,
        n_samples_saved=200*42,
        disc_batchsize=1024*2,
        n_discriminator_train_step=16,
        n_remember=300,
        n_rl_iter=1,
        entropy_coeff=0.006,
        n_reward_funcs=10,

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.995
        ),

    )

    HopperED__MULTI_AIRL = dict(
        n_expert_rollouts=50,
        n_envs=5,
        perf_top_percent=1.,
        n_samples_saved=200*42,
        disc_batchsize=1024,
        n_discriminator_train_step=8,
        n_remember=500,
        n_rl_iter=1,
        entropy_coeff=0.007,
        n_reward_funcs=10,

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.995
        ),
    )

    CheetahED__AIRL = dict(
        n_expert_rollouts=50,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=16,
        n_remember=5,
        n_rl_iter=1,
        entropy_coeff=0.007,
        n_reward_funcs=10
    )

    CheetahED__MULTI_AIRL = dict(
        n_expert_rollouts=50,
        n_envs=5,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=8,
        n_remember=2,
        n_rl_iter=1,
        entropy_coeff=0.01,
        n_reward_funcs=10,
    )

    AntED__AIRL = dict(
        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.9956469934074685
        ),
        n_expert_rollouts=50,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=16,
        n_remember=5,
        n_rl_iter=1,
        entropy_coeff=0.001,
        n_reward_funcs=10
    )

    AntED__MULTI_AIRL = dict(
        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.9956469934074685
        ),
        n_expert_rollouts=75,
        n_envs=5,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=20,
        n_remember=7,
        n_rl_iter=1,
        entropy_coeff=1e-6, # too high ?
        n_reward_funcs=10,
    )

    WalkerED__AIRL = dict(
        n_expert_rollouts=50,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=16,
        n_remember=5,
        n_rl_iter=1,
        entropy_coeff=0.02,
        n_reward_funcs=10,
        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.99
        ),
    )

    WalkerED__MULTI_AIRL = dict(
        n_envs=5,
        n_expert_rollouts=50,
        perf_top_percent=1.,
        n_samples_saved=8400,
        disc_batchsize=2048,
        n_discriminator_train_step=16,
        n_remember=5,
        n_rl_iter=1,
        entropy_coeff=0.02,
        n_reward_funcs=10,
    )

    SwimmerED__AIRL = dict(
        n_expert_rollouts=50,
        n_reward_funcs=3,

        entropy_coeff=0.006,
        n_remember=2,
        n_samples_saved=200 * 42,
        n_rl_iter=1,
        disc_batchsize=2048,
        n_discriminator_train_step=16,

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.999
        ),
    )

    SwimmerED__MULTI_AIRL = dict(
        n_envs=5,

        n_expert_rollouts=50,
        n_reward_funcs=3,

        entropy_coeff=0.006,
        n_remember=2,
        n_samples_saved=200 * 42,
        n_rl_iter=1,
        disc_batchsize=2048,
        n_discriminator_train_step=16,

        discriminator_conf=dict(
            lr=1e-3,
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.999
        ),
    )

    def postprocess(
            self
    ):

        if "n_envs" in self:
            self["multiagent"] = dict(
                policies={str(k): PolicySpec() for k in range(self["n_envs"])},
                policy_mapping_fn=lambda p_id, ep, worker, **kwargs: str(p_id)
            )



if __name__ == '__main__':
    print(IRLConfig("HopperED", "MULTI_AIRL"))
