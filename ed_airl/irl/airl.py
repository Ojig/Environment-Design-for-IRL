import os
import pickle
from time import time
from typing import Type

import numpy as np
import ray
from ray import tune
from ray.rllib import Policy
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict

from config.builder import ConfigBuilder
from env_design.env_params import EnvParams
from irl.custom_rewards import wrap_rewards, CustomRewardCallBack
from irl.discriminator import Discriminator
from rl.expert import generate_rollouts, load_expert_rollouts
from utils import next_obs_wrapper
from utils.python import partialclass
from utils.queuing import RolloutQueue, ConcatenatedRolloutQueue
from utils.rollout import sample_batch_indexation_helper

_, tf, _ = try_import_tf()


class AIRLConfig(AlgorithmConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class)

        self.expert_rollouts = []
        self.n_expert_rollouts = 50,
        self.perf_top_percent = 0.25,
        self.discriminator_conf = {
                                      'lr': 1e-3,
                                      'layer_dims': [32, 32],
                                      'default_activation': 'relu',
                                      'gamma': 0.99,
                                  },

        self.n_rollout_queued = 5
        self.n_samples_saved = 2048
        self.disc_batchsize = 128
        self.gpu = 0
        self.n_discriminator_train_step = 10,
        self.n_remember = 20,
        self.n_rl_iter = 1,

        self.n_reward_funcs = 5,

        self.reward_ckpt_path = None

    @override(AlgorithmConfig)
    def training(
            self,
            *,
            n_samples_saved=None,
            discriminator_conf=None,
            n_expert_rollouts=None,
            perf_top_percent=None,
            n_rollout_queued=None,
            disc_batchsize=None,
            n_discriminator_train_step=None,
            n_remember=None,
            n_rl_iter=None,
            n_reward_funcs=None,
            reward_ckpt_path=None,
            **kwargs,
    ) -> "AIRLConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)


        if n_reward_funcs is not None:
            self.n_reward_funcs = n_reward_funcs
        if n_samples_saved is not None:
            self.n_samples_saved = n_samples_saved
        if discriminator_conf is not None:
            self.discriminator_conf.update(**discriminator_conf)
        if n_expert_rollouts is not None:
            self.n_expert_rollouts = n_expert_rollouts
        if perf_top_percent is not None:
            self.perf_top_percent = perf_top_percent
        if n_rollout_queued is not None:
            self.n_rollout_queued = n_rollout_queued
        if disc_batchsize is not None:
            self.disc_batchsize = disc_batchsize
        if n_remember is not None:
            self.n_remember = n_remember
        if n_discriminator_train_step is not None:
            self.n_discriminator_train_step = n_discriminator_train_step
        if n_rl_iter is not None:
            self.n_rl_iter = n_rl_iter
        if reward_ckpt_path is not None:
            self.reward_ckpt_path = reward_ckpt_path

        return self


def make_AIRL(rl_algo: Type[Algorithm]):

    class AIRL(rl_algo):

        # pass expert policy trajectories OK ?
        # Override default policy class OK
        # Add metric callback (Not necessary)

        # => subclass from multi_agent_env ?
        # How to map policy i to env i ?

        # Add required config params OK
        @classmethod
        def get_default_config(cls) -> AlgorithmConfigDict:
            default_conf = AIRLConfig(rl_algo).to_dict()
            default_conf.update(rl_algo.get_default_config())

            # class Allcallbacks(CustomRewardCallBack, TraceMallocCallback):
            #     pass
            #
            # default_conf["callbacks"] = Allcallbacks
            default_conf["callbacks"] = CustomRewardCallBack

            return default_conf

        def setup(self, config):
            if config["gpu"] >= 0:
                gpus = tf.config.list_physical_devices("GPU")
                print(gpus)

                if gpus:
                    gpu0 = gpus[config["gpu"]]  # Only use GPU 0 when existing multiple GPUs
                    tf.config.experimental.set_memory_growth(gpu0, True)  # Set the usage of GPU memory according to needs

            self.expert_rollouts = config.pop("expert_rollouts")
            super(AIRL, self).setup(config)
            self.discriminator = Discriminator(
                state_shape=self.env_creator(config["env_config"]).observation_space.shape,
                **config["discriminator_conf"]
            )
            self.rollout_queue = ConcatenatedRolloutQueue(config["n_remember"])
            self.last_reward_func_params = [None for _ in range(config["n_reward_funcs"])]
            self.train_cntr = 0

            self.labels = np.zeros(config["disc_batchsize"]*2, dtype=np.float32)
            self.labels[-config["disc_batchsize"]:] = 1.
            self.losses = np.ones(1)



        def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
            custom_reward_cls = wrap_rewards(
                super(AIRL, self).get_default_policy_class(config), shaped=False,
            )
            return partialclass(custom_reward_cls,
                                rew_config=config["discriminator_conf"],
                                rew_weights=None)

        def build_discriminator_batch(
                self,
                latest_rollouts,
                rollout_queue,
                expert_rollouts,
        ):
            # Probably can be done in the queue itself already
            policy_samples = rollout_queue.concatenated()
            latset_idxs = np.random.choice(
                latest_rollouts.count,
                size=(self.config["disc_batchsize"] // 2,),
                replace=False
            )
            policy_idxs = np.random.choice(
                policy_samples.count,
                size=(self.config["disc_batchsize"]//2,),
                replace=False
            )
            expert_idxs = np.random.choice(
                self.postprocess_samples(expert_rollouts).count,
                size=(self.config["disc_batchsize"],),
                replace=False
            )

            # _, _, recomputed = self.get_policy().compute_actions(
            #     expert_batch[SampleBatch.OBS]
            # )
            # expert_batch[SampleBatch.ACTION_LOGP] = recomputed[SampleBatch.ACTION_LOGP]

            batch = concat_samples(
                [sample_batch_indexation_helper(latest_rollouts, latset_idxs),
                    sample_batch_indexation_helper(policy_samples, policy_idxs),
                 sample_batch_indexation_helper(expert_rollouts, expert_idxs)]
            )

            return [batch]

        def postprocess_samples(self, samples):
            return samples


        def compute_logp(self, samples, idx="default_policy"):
            p = self.get_policy(idx)
            dist, _ = p.model(samples)
            action_dist = p.dist_class(dist, p.model)

            x = action_dist.logp(samples[SampleBatch.ACTIONS])
            return np.array(x)

        def recompute_expert_rollout_probs(self):
            # More efficient if we run many discriminator train steps with the same policy
            self.expert_rollouts[SampleBatch.ACTION_LOGP] = self.compute_logp(
                self.expert_rollouts
            )

        def parallel_samples(self):
            return synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["n_samples_saved"], concat=True
            )

        @ExperimentalAPI
        def training_step(self) -> ResultDict:
            # Override training step,
            # Save some trajs from our policy

            if self.train_cntr == 0:
                rew_params = [self.discriminator.get_params()]
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: p.reward_function.update_params(rew_params)
                )
                for worker in self.workers.remote_workers():
                    worker.foreach_policy_to_train.remote(
                        lambda p, pid: p.reward_function.update_params(rew_params)
                    )

            if self.train_cntr % self.config["n_rl_iter"] == 0:
                t0 = time()
                if self.rollout_queue.length() >= 0:
                    rollouts = self.parallel_samples()
                    # if dealing with multiple envs at the same time:
                    rollouts = self.postprocess_samples(rollouts)
                    self.rollout_queue.put(rollouts)

                #   to train discriminator for each policy train step

                t_expert_compute_start = time()

                self.recompute_expert_rollout_probs()
                t1 = time()

                t3 = 0
                t4 = 0
                for disc_iter in range(self.config["n_discriminator_train_step"]):
                    t2 = time()
                    batches = self.build_discriminator_batch(
                        rollouts,
                        self.rollout_queue,
                        self.expert_rollouts,
                    )
                    ttmp = time()
                    t3 += ttmp - t2
                    loss, self.losses = self.discriminator.train(
                        input_dicts=[
                            dict(
                                states=batch[SampleBatch.OBS],
                                next_states=batch[SampleBatch.NEXT_OBS],
                                logprobs=batch[SampleBatch.ACTION_LOGP],
                            )
                        for batch in batches],
                        labels=self.labels,
                        gpu=self.config['gpu'],
                        log=False
                    )
                    t4 += time() - ttmp
                #   allow policy training for multiple steps without discriminator (when rl_training_step > 0)
                #   stock rollouts for discriminator
                rew_params = self.discriminator.get_params()

                params = self.retrieve_reward_params()
                norm_params = self.retrieve_reward_norm(params)

                def get_params_for_pid(pid):
                    return [dict(
                        **rew_params,
                        norm_params=norm_params[pid],
                    )]
                self.last_reward_func_params.append(self.discriminator.get_params())
                self.last_reward_func_params.pop(0)
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: p.reward_function.update_params(get_params_for_pid(pid))
                )
                for worker in self.workers.remote_workers():
                    worker.foreach_policy_to_train.remote(
                        lambda p, pid: p.reward_function.update_params(get_params_for_pid(pid))
                    )

            results = super(AIRL, self).training_step()

            if self.train_cntr % self.config["n_rl_iter"] == 0:
                policies = list(self.config["multiagent"]["policies"].keys())
                results[policies[0]]["custom_metrics"] = dict(
                    discriminator_loss_mean=loss.numpy(),
                    discriminator_train_ms=t4*1000,
                    discriminator_collect_ms=(t_expert_compute_start-t0)*1000,
                    discriminator_recompute_expert=(t1-t_expert_compute_start)*1000,
                    discriminator_batching_ms=t3*1000,
                    discriminator_rollout_queue_size=self.rollout_queue.size(),
                    discriminator_rollout_queue_size_bytes=self.rollout_queue.size_bytes(),
                    discriminator_rollout_queue_length=self.rollout_queue.length(),
                    #GAN_reward_running_mean=self.discriminator.normalizer.running_mean,
                    #GAN_reward_running_var=self.discriminator.normalizer.running_var,
                )
                for i, sub_loss in enumerate(self.losses):
                    results[policies[0]]["custom_metrics"][f"discriminator_loss_{i}"] = sub_loss.numpy()
            self.train_cntr += 1

            return results


        #def kill_big_workers(self):
        #   for worker in self.workers.remote_workers():
        #        worker.stop()

        def retrieve_reward_params(self):
            all_params = {}
            for pid in self.config["multiagent"]["policies"]:
                params = self.workers.local_worker().for_policy(
                    lambda p: p.reward_function.get_norms(),
                    policy_id=pid
                )
                for worker in self.workers.remote_workers():
                    params.extend(ray.get(worker.for_policy.remote(
                        lambda p: p.reward_function.get_norms(),
                        policy_id=pid
                    )))
                all_params[pid]= params

            return all_params

        def retrieve_reward_norm(self, all_params):

            norms = {}
            for pid, params in all_params.items():
                m = 0.
                v = 0.
                for p in params:
                    mi, vi = p
                    m += mi
                    v += vi

                norms[pid] = (m / len(params), v/len(params))

            return norms

        def save_checkpoint(self, checkpoint_dir: str) -> str:
            file_name = checkpoint_dir.split(os.sep)[-3]
            path = os.getcwd().split("/checkpoints/")[0]

            for i in range(len(self.last_reward_func_params)):
                self.last_reward_func_params[i]["frozen"] = True

            print(self.last_reward_func_params)

            full_path = f"{path}/checkpoints/estimated_reward_functions/AIRL/{file_name}" if self.config["reward_ckpt_path"] is None \
                else "../single_env_reward_function"
            suffix = ""
            ext = ".pkl"
            i = 0
            while os.path.exists(full_path + suffix + ext):
                i += 1
                suffix = f"_v{i}"

            with open(full_path+suffix+ext,
                  'wb+') as f:
                pickle.dump(self.last_reward_func_params, f)

            path = super(AIRL, self).save_checkpoint(checkpoint_dir)
            return path

    return AIRL



def run(
        env_params: EnvParams,
        config: ConfigBuilder,
        max_iter = 3000,
        from_saved_rollouts=True,
        n_runs=1,

):
    run_config = config.build(
        env_params
    )

    run_name = f"{config.rl_algo}_{run_config['env']}_{env_params.params_id}"

    if from_saved_rollouts:
        expert_rollouts = concat_samples(load_expert_rollouts(
            run_config["env"],
            env_params.params_id,
            n_rollouts=run_config["n_expert_rollouts"]
        ))
    else:
        expert_rollouts = concat_samples(generate_rollouts(
            env_params,
            config,
            n_rollouts=run_config["n_expert_rollouts"],
            filtering=True,
            perf_top_percent=run_config["perf_top_percent"]
        ))



    run_config["expert_rollouts"] = expert_rollouts

    algo_cls = ALGORITHMS[config.rl_algo]()[0]

    if config.rl_algo in ["APPO"]:
        algo_cls = next_obs_wrapper.wrap_algo(algo_cls)

    tune.run(
        make_AIRL(algo_cls),
        name=run_name,
        config=run_config,
        num_samples=1,
        checkpoint_at_end=True,
        stop={"training_iteration": max_iter,
              #"time_since_restore": 60*12,
              #"episode_reward_mean": envs.DESIGNS[run_config["env"]].EXPERT_EPISODE_RETURN,
              },
        local_dir='checkpoints/irl/rllib_airl/'
    )