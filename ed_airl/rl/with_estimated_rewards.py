import os

from env_design import envs
import numpy as np
from ray import tune
from ray.rllib import SampleBatch
from ray.rllib.algorithms.registry import ALGORITHMS

from config.builder import ConfigBuilder
from env_design.env_params import EnvParams
from irl import custom_rewards
from irl.custom_rewards import CustomRewardCallBack
from irl.reward_function import RewardEnsemble
from utils.rollout import make_rollouts


def eval_reward_function(
        env_params: EnvParams,
        reward_design_id: int,
        config: ConfigBuilder,
        max_iter: int = 220,
        from_path=None,
        from_object=None,
        name="",
):

    run_config = config.build_base_rl(
        env_params,
        callbacks=CustomRewardCallBack
    )

    algo_cls = ALGORITHMS[config.rl_algo]()[0]
    algo_inst = algo_cls(run_config)
    policy_cls = algo_inst.get_default_policy_class(run_config)
    algo_inst.cleanup()

    if from_object is not None:
        reward_function = from_object
    else:
        reward_function = RewardEnsemble(
            from_ckpt=from_path if from_path is not None else f"checkpoints/estimated_reward_functions/{config.irl_algo}/{config.rl_algo}_{run_config['env']}_{reward_design_id} "
        )

    rew_params = reward_function.get_params()
    for i in range(len(rew_params)):
        rew_params[i]["frozen"] = False

    run_name = f"{config.rl_algo}_{run_config['env']}_{env_params.params_id}_{reward_design_id}"
    ckpt_path = ["checkpoints"]
    if name != "":
        ckpt_path += [name]
    ckpt_path += ["estimated_reward_functions", config.irl_algo, "eval"]


    results = tune.run(
        custom_rewards.wrap_algo_with_custom_rewards(
            algo_cls,
            policy_cls,
            reward_function.config,
            rew_params,
            compute_reward_distance=True
        ),

        name=run_name,
        config=run_config,
        num_samples=1,
        checkpoint_at_end=True,
        stop={"iterations_since_restore": max_iter,
              },
        local_dir=os.sep.join(ckpt_path),
    )

    best_result = results.get_best_trial(metric="episode_reward_mean", mode="max")

    return best_result


def eval_policy_on_estimated(
        env_params: EnvParams,
        policy,
        n_trajs: int,
        reward_function: RewardEnsemble,
):

    rollouts = make_rollouts(
        policy,
        n_trajs,
        env_creator=policy.config["env"],
        perf_top_percent=1.
    )

    return np.mean(np.concatenate([
        reward_function(trajectory[SampleBatch.OBS], normalize=False) for trajectory in rollouts
    ]))




