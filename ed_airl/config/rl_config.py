from copy import deepcopy

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import MeanStdFilter

from config.base_config import EnvAlgoConfig
from utils.distribution import ScaledStd
from utils.norm import RewardNormalizerCallBack


ModelCatalog.register_custom_action_dist(
    "scaled_gaussian",
    ScaledStd,
)


class RLConfig(EnvAlgoConfig):
    BASE = dict(
        disable_env_checking=True,
        framework="tf2",
        eager_tracing=True,
        render_env=False,
        explore=True,
        model=dict(
            fcnet_hiddens=[64, 64]
        )
        # memory_per_worker= 800 * 1024 * 1024,
        # object_store_memor_y_per_worker = 128 * 1024 * 1024,
    )

    MazeED__PPO = dict(
        gamma=0.995,
        lambda_=1.,
        kl_coeff=1.0,
        lr=0.0002,
        train_batch_size=160000//2,
        batch_mode="complete_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=20,
        sgd_minibatch_size=32768//2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.,
        clip_param=0.1,
        grad_clip=0.5,
        num_envs_per_worker=16,
    )

    MazeED__PPO__AIRL = dict(
        gamma=0.995,  # 0.995
        lambda_=0.99,
        kl_coeff=1.0,
        lr=0.0002,
        train_batch_size=41*1000,
        batch_mode="complete_episodes",
        num_sgd_iter=20,
        sgd_minibatch_size=3500,
        vf_loss_coeff=0.05,
        entropy_coeff=0.,
        clip_param=0.1,
        grad_clip=0.8,
        num_envs_per_worker=1,
        #observation_filter=MeanStdFilter
    )

    # Works with defaults
    MazeED__PPO__MULTI_AIRL = dict()

    MazeED__PPO_estimated = MazeED__PPO__AIRL.copy()
    MazeED__PPO_estimated["lr"] = 0.0002
    MazeED__PPO_estimated["gamma"] = 0.995
    MazeED__PPO_estimated["entropy_coeff"] = 1e-5

    HopperED__PPO = dict(
        gamma=0.995,
        kl_coeff=1.0,
        lr=0.0001,
        train_batch_size=160000,
        batch_mode="complete_episodes",
        num_sgd_iter=20,
        sgd_minibatch_size=32768,
        num_envs_per_worker=2,
    )

    HopperED__PPO__AIRL = dict(
        gamma=0.99, # 0.995
        kl_coeff=0.,
        lr=0.00058,
        train_batch_size=8192,
        lambda_=0.99,
        batch_mode="truncate_episodes",
        num_sgd_iter=20,
        clip_param=0.1,
        sgd_minibatch_size=512,
        grad_clip=0.9,
        vf_loss_coeff=0.203,
        entropy_coeff=0.011,
        num_envs_per_worker=1,
    )

    HopperED__PPO_estimated = HopperED__PPO__AIRL.copy()
    HopperED__PPO_estimated["entropy_coeff"] = 0.002

    HopperED__PPO__MULTI_AIRL = dict(
        gamma=0.99,
        kl_coeff=0.,
        lr=0.00058,
        train_batch_size=8192,
        lambda_=0.99,
        batch_mode="complete_episodes",
        num_sgd_iter=20,
        clip_param=0.1,
        sgd_minibatch_size=512,
        grad_clip=0.9,
        vf_loss_coeff=0.203,
        entropy_coeff=0.011,
        num_envs_per_worker=1,
    )

    CheetahED__PPO = dict(
        gamma=0.99,
        lambda_=0.96,
        kl_coeff=1.0,
        lr=0.0003,
        train_batch_size=65536*31//42,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=4096*31//42,
        vf_loss_coeff=0.5,
        entropy_coeff=0.001,
        clip_param=0.2,
        grad_clip=0.5,
        num_envs_per_worker=32,
        observation_filter=MeanStdFilter
    )

    CheetahED__PPO__AIRL = dict(
        gamma=0.99,
        lambda_=0.96,
        kl_coeff=.0,
        lr=0.0005,
        train_batch_size=int(42 * 1024),  # 10000 for mujoco
        batch_mode="truncate_episodes",
        num_sgd_iter=20,
        sgd_minibatch_size=4096,# * 31 // 42,
        vf_loss_coeff=0.1,
        entropy_coeff=0.01,
        clip_param=0.1,
        grad_clip=0.8,
        num_envs_per_worker=16,
        observation_filter=MeanStdFilter
    )

    CheetahED__PPO_estimated = dict(
        gamma=0.99,
        lambda_=0.96,
        kl_coeff=1.0,
        lr=0.0005,
        train_batch_size=65536,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=4096,
        vf_loss_coeff=0.5,
        entropy_coeff=0.004,
        clip_param=0.1,
        grad_clip=0.8,
        num_envs_per_worker=32,
        observation_filter=MeanStdFilter
    )

    CheetahED__PPO__MULTI_AIRL = dict(
        gamma=0.99,
        lambda_=0.96,
        kl_coeff=.0,
        lr=0.0005,
        train_batch_size=42 * 1024,  # 10000 for mujoco
        batch_mode="truncate_episodes",
        num_sgd_iter=20,
        sgd_minibatch_size=4096,
        vf_loss_coeff=0.05,
        entropy_coeff=0.01,
        clip_param=0.1,
        grad_clip=0.8,
        num_envs_per_worker=16,
        observation_filter=MeanStdFilter
    )

    AntED__PPO = dict(
        gamma=0.9956469934074685,
        kl_coeff=1.0,
        lr=0.00023767897095611476,
        train_batch_size=8192,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=4096,
        vf_loss_coeff=0.46,
        entropy_coeff=0.0004, # 0007724114264418187
        clip_param=0.3,
        grad_clip=0.8613922338228183,
        num_envs_per_worker=4,
        observation_filter=MeanStdFilter,
        lambda_=0.8,
    )

    AntED__PPO_estimated = dict(
        gamma=0.9960370296387727,
        kl_coeff=1.0,
        lr=0.00017952780317577008,
        train_batch_size=65536//2,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=2048//2,
        vf_loss_coeff=0.09085304817922157,
        entropy_coeff=0.0004,  # 0007724114264418187
        clip_param=0.15424753769372318,
        grad_clip=0.8,
        num_envs_per_worker=4,
        observation_filter=MeanStdFilter,
        lambda_=0.9025241639549184,
    )

    AntED__PPO__AIRL = dict(
        gamma=0.9960370296387727,
        kl_coeff=1.0,
        lr=0.00017952780317577008,
        train_batch_size=65536//4,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=2048//4,
        vf_loss_coeff=0.09085304817922157,
        entropy_coeff=0.0004,  # 0007724114264418187
        clip_param=0.15424753769372318,
        grad_clip=0.8,
        num_envs_per_worker=4,
        observation_filter=MeanStdFilter,
        lambda_=0.9025241639549184,
    )

    AntED__PPO__MULTI_AIRL = dict(
        gamma=0.9960370296387727,
        kl_coeff=1.0,
        lr=0.00017952780317577008,
        train_batch_size=65536//4,
        batch_mode="truncate_episodes",
        rollout_fragment_length=200,
        num_sgd_iter=32,
        sgd_minibatch_size=2048//4,
        vf_loss_coeff=0.09085304817922157,
        entropy_coeff=0.0004,
        clip_param=0.15424753769372318,
        grad_clip=0.8,
        num_envs_per_worker=4,
        observation_filter=MeanStdFilter,
        lambda_=0.9025241639549184,
    )

    WalkerED__PPO = dict(
        lr=4.e-4,
        vf_loss_coeff=0.5,
        gamma=0.99,
        lambda_=0.99,
        entropy_coeff=0.,
        num_sgd_iter=20,
        kl_coeff=.0,
        num_envs_per_worker=1,
        sgd_minibatch_size= 32768//16,
        train_batch_size= 320000//16,
        batch_mode= "complete_episodes",
        observation_filter=MeanStdFilter,
        clip_param=0.1,
        grad_clip=0.9,
        model=dict(
            fcnet_hiddens=[64, 64],
            free_log_std=True,
        )
    )

    WalkerED__PPO_estimated = {
        "batch_mode": "complete_episodes",
        "clip_param": 0.2507924437264407,
        "entropy_coeff": 0.01949398870974174,
        "gamma": 0.99,
        "grad_clip": 0.9,
        "kl_coeff": 0.0,
        "lambda": 0.8,
        "lr": 0.00010705107925263512,
        "num_envs_per_worker": 8,
        "num_sgd_iter": 20,
        "rollout_fragment_length": 200,
        "sgd_minibatch_size": 1024,
        "train_batch_size": 16384,
        "vf_loss_coeff": 0.9526268988092673,
        "observation_filter": MeanStdFilter,
    }

    WalkerED__PPO__AIRL = dict(
        gamma=0.98,
        kl_coeff=.0,
        lr=3.05e-5,
        train_batch_size=8192*4, #16384,
        rollout_fragment_length=200,
        num_sgd_iter=20,
        sgd_minibatch_size=512*4, # 1024,
        vf_loss_coeff=0.617,
        entropy_coeff=0.002,
        clip_param=0.4,
        grad_clip=0.6,
        num_envs_per_worker=1,
        batch_mode="complete_episodes",
        observation_filter=MeanStdFilter,
        lambda_=0.92,
        model=dict(
            fcnet_hiddens=[64, 64],
            #free_log_std=True,
        )
    )

    WalkerED__PPO__MULTI_AIRL = dict(
        gamma=0.98,
        kl_coeff=.0,
        lr=3.05e-5,
        train_batch_size=8192,  # 16384,
        rollout_fragment_length=200,
        num_sgd_iter=20,
        sgd_minibatch_size=512,  # 1024,
        vf_loss_coeff=0.617,
        entropy_coeff=0.002,
        clip_param=0.4,
        grad_clip=0.6,
        num_envs_per_worker=1,
        batch_mode="complete_episodes",
        observation_filter=MeanStdFilter,
        lambda_=0.92,
        model=dict(
            fcnet_hiddens=[64, 64],
            # free_log_std=True,
        )
    )

    SwimmerED__PPO = dict(
        gamma=0.999,
        kl_coeff=1.0,
        lr=0.0003,
        clip_param=0.1,
        train_batch_size=160000,
        batch_mode="truncate_episodes",
        num_sgd_iter=20,
        entropy_coeff=0.0001,
        sgd_minibatch_size=32768,
        num_envs_per_worker=16,
    )

    SwimmerED__PPO__AIRL = dict(
        gamma=0.999,
        lambda_=0.95,
        kl_coeff=.0,
        lr=0.00134,
        train_batch_size=4096*5//2,
        rollout_fragment_length=200,
        num_sgd_iter=5,
        sgd_minibatch_size=4096,
        vf_loss_coeff=0.62,
        entropy_coeff=0.006,
        clip_param=0.1,
        grad_clip=2.,
        num_envs_per_worker=1, # 16
        batch_mode="truncate_episodes",

    )

    SwimmerED__PPO__MULTI_AIRL = dict(
        gamma=0.999,
        lambda_=0.95,
        kl_coeff=.0,
        lr=0.00134,
        train_batch_size=4096*5//2,
        rollout_fragment_length=200,
        num_sgd_iter=5,
        sgd_minibatch_size=4096,
        vf_loss_coeff=0.62,
        entropy_coeff=0.006,
        clip_param=0.1,
        grad_clip=2.,
        num_envs_per_worker=1,
        batch_mode="truncate_episodes",
    )

    SwimmerED__PPO_estimated = SwimmerED__PPO__AIRL.copy()
    SwimmerED__PPO_estimated["entropy_coeff"] = 0.002




    def __init__(
            self,
            env,
            algo,
            irl_algo=None,
            *args,
    ):
        super(RLConfig, self).__init__(env, algo)
        if irl_algo is not None:
            self.config_name += "__" + irl_algo

    @property
    def rl_config(self):
        new = deepcopy(self)
        new.config_name = "__".join(new.config_name.split("__")[:2])
        # get rid of irl
        return new

    def postprocess(
            self
    ):
        if "lambda_" in self:
            lambda_ = self.pop("lambda_")
            if "lambda" not in self:
                self["lambda"] = lambda_


if __name__ == '__main__':
    print(RLConfig("HopperED", "MULTI_AIRL"))
