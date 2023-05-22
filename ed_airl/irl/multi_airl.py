import os
import pickle
from pprint import pprint
from typing import Type, List

import numpy as np
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, MultiAgentBatch
from ray.rllib.utils import try_import_tf
from ray.tune import register_env
from ray.tune import loguniform, uniform, choice

from config.builder import ConfigBuilder
from env_design.multi_env_wrapper import make_multi_config_env
from irl.airl import make_AIRL
from irl.discriminator import Discriminator
from rl.expert import generate_rollouts, load_expert_rollouts
from utils import next_obs_wrapper
from utils.queuing import RolloutQueue, ConcatenatedRolloutQueue
from utils.rollout import sample_batch_indexation_helper, ma_synchronous_parallel_sample

_, tf, _ = try_import_tf()


def make_multi_AIRL(rl_algo: Type[Algorithm]):
    AIRL = make_AIRL(rl_algo)

    class multi_AIRL(AIRL):

        def setup(self, config):
            if config["gpu"] >= 0:
                gpus = tf.config.list_physical_devices("GPU")
                print(gpus)

                if gpus:
                    gpu0 = gpus[config["gpu"]]  # Only use GPU 0 when multiple GPUs
                    tf.config.experimental.set_memory_growth(gpu0,
                                                             True)  # Set the usage of GPU memory according to needs

            self.expert_rollouts = config.pop("expert_rollouts")
            super(AIRL, self).setup(config)
            self.discriminator = Discriminator(
                state_shape=self.env_creator(config["env_config"]).observation_space.shape,
                n_envs=len(config["multiagent"]["policies"]),
                **config["discriminator_conf"]
            )
            self.rollout_queue = ConcatenatedRolloutQueue(config["n_remember"])
            self.last_reward_func_params = [None for _ in range(config["n_reward_funcs"])]
            self.train_cntr = 0

            self.labels = np.zeros((config["disc_batchsize"]) * 2, dtype=np.float32)
            self.labels[-config["disc_batchsize"]:] = 1.
            self.losses = np.ones(len(config["multiagent"]["policies"]), dtype=np.float32)

        def postprocess_samples(self, samples):
            # keep multi agent batches
            return samples  # concat_samples(list(samples.policy_batches.values()))

        def parallel_samples(self):
            return ma_synchronous_parallel_sample(
                worker_set=self.workers,
                policies=self.config["multiagent"]["policies"],
                max_steps_per_policy=self.config["n_samples_saved"],
                concat=True)

        def build_discriminator_batch(
                self,
                latest_rollouts,
                rollout_queue,
                expert_rollouts,
        ):
            # Probably can be done in the queue itself already
            policy_samples = rollout_queue.concatenated()
            expert_samples = self.postprocess_samples(expert_rollouts)

            batches = []

            actual_batch_size = self.config["disc_batchsize"]  # // len(self.config["multiagent"]["policies"]

            for i in self.config["multiagent"]["policies"]:
                latest_idxs = np.random.choice(
                    latest_rollouts.policy_batches[i].count,
                    size=(actual_batch_size // 2,),
                    replace=False
                )

                policy_idxs = np.random.choice(
                    policy_samples.policy_batches[i].count,
                    size=(actual_batch_size // 2,),
                    replace=False
                )
                expert_idxs = np.random.choice(
                    expert_samples.policy_batches[i].count,
                    size=(actual_batch_size,),
                    replace=False
                )

                batches.append(concat_samples(
                    [sample_batch_indexation_helper(latest_rollouts.policy_batches[i], latest_idxs),
                     sample_batch_indexation_helper(policy_samples.policy_batches[i], policy_idxs),
                     sample_batch_indexation_helper(expert_samples.policy_batches[i], expert_idxs)]
                ))

            return batches

        def compute_logp(self, samples, agent_idx):
            p = self.get_policy(agent_idx)
            dist, _ = p.model(samples.policy_batches[agent_idx])
            action_dist = p.dist_class(dist, p.model)

            x = action_dist.logp(samples.policy_batches[agent_idx][SampleBatch.ACTIONS])
            return np.array(x)

        def recompute_expert_rollout_probs(self):
            # More efficient if we run many discriminator train steps with the same policy
            for i in self.config["multiagent"]["policies"]:
                self.expert_rollouts.policy_batches[i][SampleBatch.ACTION_LOGP] = self.compute_logp(
                    self.expert_rollouts, i
                )

        def save_checkpoint(self, checkpoint_dir: str) -> str:
            file_name = checkpoint_dir.split(os.sep)[-3]
            path = os.getcwd().split("/checkpoints/")[0]

            for i in range(len(self.last_reward_func_params)):
                self.last_reward_func_params[i]["frozen"] = True

            print(self.last_reward_func_params)

            full_path = f"{path}/checkpoints/estimated_reward_functions/MULTI_AIRL/{file_name}" if self.config[
                                                                                                       "reward_ckpt_path"] is None \
                else "../best_guess_reward_function"

            suffix = ""
            ext = ".pkl"
            i = 0
            while os.path.exists(full_path + suffix + ext):
                i += 1
                suffix = f"_v{i}"

            with open(full_path + suffix + ext,
                      'wb+') as f:
                pickle.dump(self.last_reward_func_params, f)

            path = rl_algo.save_checkpoint(self, checkpoint_dir)
            return path

    return multi_AIRL


def run(
        all_env_params,
        config: ConfigBuilder,
        max_iter=500,
        deterministic_choice: List = None,
        from_saved_rollouts=True,

):
    irl_config = config.build_base_irl()

    assert deterministic_choice is None or irl_config["n_envs"] == len(
        deterministic_choice), "deterministic_choice must be of same length than n_envs"

    randomly_selected_design_ids = deterministic_choice if deterministic_choice is not None else \
        np.concatenate(
            [np.array([0]), np.random.choice(len(all_env_params) - 1, irl_config["n_envs"] - 1, replace=False) + 1])
    randomly_selected_designs = [p for p in np.array(all_env_params)[randomly_selected_design_ids]]

    print()
    print("Selected designs:")
    pprint(randomly_selected_designs)
    print()

    multi_env = make_multi_config_env(
        irl_config["env"],
        randomly_selected_designs
    )
    multi_env_name = f"random_ed_{irl_config['env']}"
    register_env(multi_env_name, multi_env)

    run_name = f"{config.rl_algo}_{multi_env_name}"

    n_rollouts = irl_config["n_expert_rollouts"] // irl_config["n_envs"]

    rollout_dict = {}
    if from_saved_rollouts:
        for expert_id, (env_params_id, env_params) in enumerate(zip(randomly_selected_design_ids,
                                                                    randomly_selected_designs)):
            rollout_dict[str(expert_id)] = concat_samples(load_expert_rollouts(
                irl_config["env"],
                env_params_id,
                n_rollouts
            ))
    else:
        for expert_id, (env_params_id, env_params) in enumerate(zip(randomly_selected_design_ids,
                                                                    randomly_selected_designs)):
            rollouts = generate_rollouts(
                env_params,
                config,
                n_rollouts=n_rollouts,
                filtering=True,
                perf_top_percent=irl_config["perf_top_percent"]
            )
            rollout_dict[str(expert_id)] = concat_samples(rollouts)

    expert_rollouts = MultiAgentBatch(rollout_dict, env_steps=sum(
        [r.count for r in rollout_dict.values()]
    ))

    run_config = config.build()

    run_config["expert_rollouts"] = expert_rollouts

    run_config["env"] = multi_env_name
    run_config["env_config"] = dict(num_agents=run_config.pop("n_envs"), selected_designs=randomly_selected_design_ids)

    pprint(run_config)

    algo_cls = ALGORITHMS[config.rl_algo]()[0]

    if config.rl_algo in ["APPO"]:
        algo_cls = next_obs_wrapper.wrap_algo(algo_cls)



    tune.run(
        make_multi_AIRL(algo_cls),
        name=run_name,
        config=run_config,
        num_samples=1,
        checkpoint_at_end=True,
        stop={"training_iteration": max_iter,
              },
        local_dir='checkpoints/irl/rllib_airl/'
    )
