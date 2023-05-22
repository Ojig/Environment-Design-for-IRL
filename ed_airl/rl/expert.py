import os
import pickle

import numpy as np
import ray.tune as tune
from ray.rllib import SampleBatch
from ray.rllib.algorithms.registry import ALGORITHMS

import utils.model_loading
from config.builder import ConfigBuilder
from env_design import envs
from env_design.env_params import EnvParams
from utils import next_obs_wrapper
from utils.model_loading import get_ckpt_path
from utils.rollout import make_rollouts, visualize


def get_expert_for(
        env_params: EnvParams,
        config: ConfigBuilder,
):
    run_config = config.build(
        env_params=env_params
    )

    run_name = f"{config.rl_algo}_{run_config['env']}_{env_params.params_id}"

    if not os.path.exists(os.sep.join(["checkpoints/experts", run_name])):

        print(run_config)

        algo_cls = ALGORITHMS[config.rl_algo]()[0]
        if config.rl_algo in ["APPO"]:
            algo_cls = next_obs_wrapper.wrap_algo(algo_cls)

        env2criteria = {
            "CheetahED": 8_000_000,
            "AntED": 6_000_000,
            "HopperED": 17_000_000,
            "MazeED": 3_000_000,
            "SwimmerED": 18_000_000,
        }
        stopping_criteria = {"agent_timesteps_total": env2criteria[run_config["env"]]}

        tune.run(
            algo_cls,
            name=run_name,
            metric="training_iteration",
            mode="min",
            config=run_config,
            num_samples=1,
            checkpoint_at_end=True,
            stop=stopping_criteria,
            local_dir='checkpoints/experts',
        )


def load_expert_policy(
        used_algo,
        env_params,
        config: dict,
        full_path=None
):
    if full_path is None:
        path = [f"checkpoints/experts", f"{used_algo}_{config['env']}_{env_params.params_id}"]
    else:
        path = full_path
        # look at all runs

    p = utils.model_loading.load_policy(
        path,
        used_algo,
        config,
    )
    return p


def generate_rollouts(
        env_params,
        config: ConfigBuilder,
        n_rollouts,
        filtering=True,
        perf_top_percent=0.25,
        render=False,
        full_path=None
):
    rl_config = config.build_base_rl(
        env_params
    )

    if full_path is None:

        expert_policy = load_expert_policy(
            config.rl_algo,
            env_params,
            rl_config,
            full_path=full_path
        )

        if render:
            visualize(
                expert_policy,
                n_rollouts
            )
        else:

            return make_rollouts(
                expert_policy,
                n_rollouts,
                env_creator=expert_policy.config["env"],
                filtering=filtering,
                perf_top_percent=perf_top_percent
            )

    else:
        _, all_runs, _ = next(os.walk(full_path))

        for run in all_runs:
            p = get_ckpt_path([full_path, run])
            print(p)
            policy = load_expert_policy(
                config.rl_algo,
                env_params,
                rl_config,
                full_path=p
            )
            visualize(policy, n_rollouts)


def save_rollouts(
        env_params,
        config: ConfigBuilder,
        n_rollouts,
        filtering=True,
        perf_top_percent=0.25,
):
    conf = config.build(
        env_params
    )
    rollouts = generate_rollouts(
        env_params,
        config,
        n_rollouts,
        filtering,
        perf_top_percent,
    )

    path = f"checkpoints/experts/rollouts/{conf['env']}_{env_params.params_id}"

    with open(path + '.pkl',
              'wb+') as f:
        pickle.dump(rollouts, f)


def load_expert_rollouts(
        env,
        params_id,
        n_rollouts
):
    path = f"checkpoints/experts/rollouts/{env}_{params_id}.pkl"

    with open(path, 'rb') as f:
        rollouts = pickle.load(f)

    return rollouts[:n_rollouts]


if __name__ == '__main__':
    rollouts = load_expert_rollouts("CheetahED", 0, 100)

    print([
        np.sum(r[SampleBatch.REWARDS]) for r in rollouts
    ])
