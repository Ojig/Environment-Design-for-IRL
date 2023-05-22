from typing import List

from env_design import envs
import numpy as np
from fire import Fire
from env_design.multi_env_wrapper import make_multi_config_env
from ray import tune

from env_design.env_params import EnvParams
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, MultiAgentBatch
from ray.tune import register_env

from config import irl_config
from config.builder import ConfigBuilder
from irl.airl import make_AIRL
from irl.multi_airl import make_multi_AIRL
from irl.reward_function import RewardFunction, RewardEnsemble
from rl.expert import load_expert_rollouts
from rl.with_estimated_rewards import eval_reward_function
from utils.model_loading import load_policy
from utils.python import Default
from utils.rollout import make_rollouts

import logging
logging.basicConfig(filename="ed_airl.log", level=logging.DEBUG, format="%(asctime)s %(message)s")


class EDAIRL:

    BASE_ENV = 0

    def __init__(
            self,
            all_env_params: List[EnvParams],
            env,
            rl_algo,
            inst: Default,
            max_iter_airl=220, # 220
            max_iter_multi_airl=350, #600
            max_iter_rl=220, #320
            n_eval_rollouts=100,
            n_iter=5,
            subset_size=5,
            out_path="test",

            **kwargs
    ):
        self.max_iter_airl = max_iter_airl
        self.max_iter_multi_airl = max_iter_multi_airl
        self.max_iter_rl = max_iter_rl
        self.n_eval_rollouts = n_eval_rollouts
        self.n_iter = n_iter
        self.subset_size = subset_size
        self.out_path = out_path

        inst.setup(
            env=env,
            rl_algo=rl_algo,
            irl_algo="AIRL",

            **kwargs

            #n_reward_funcs=1,
            #n_samples_saved=500
        )
        self.single_env_config: ConfigBuilder = inst.config_to_build

        inst.setup(
            env=env,
            rl_algo=rl_algo,
            irl_algo="MULTI_AIRL",

            **kwargs
            #n_reward_funcs=1,
            #n_samples_saved=500
        )
        self.multi_env_config: ConfigBuilder = inst.config_to_build

        self.airl_config = self.single_env_config.build_base_irl()

        self.rollout_dict = {}
        self.reward_functions = []

        self.all_env_params = {i: p for i, p in enumerate(all_env_params)}
        self.selected_envs = []
        self.v_map = {}
        self.algo_cls = ALGORITHMS[self.single_env_config.rl_algo]()[0]


    def __call__(self):

        best_guess = self.base_env_init()

        for n in range(1, self.n_iter):
            # find the best env to learn from
            T_star_id = self.arbitrary_ed(
                best_guess,
                self.reward_functions
            )
            T_star = self.all_env_params.pop(T_star_id)
            self.selected_envs.append(T_star)

            self.rollout_dict[str(n)] = concat_samples(load_expert_rollouts(
                self.airl_config["env"],
                params_id=T_star.params_id,
                n_rollouts=self.airl_config["n_expert_rollouts"] // self.n_iter
            ))

            # Pointless for last iter
            if n < self.n_iter -1:
                # Get R_k the single_env reward for our newly chosen env
                run_config = self.single_env_config.build(env_params=T_star)
                run_config["expert_rollouts"] = self.rollout_dict[str(n)]
                run_config["reward_ckpt_path"] = True
                tune.run(
                    make_AIRL(self.algo_cls),
                    name=self.out_path,
                    config=run_config,
                    num_samples=1,
                    checkpoint_at_end=True,
                    stop={"training_iteration": self.max_iter_airl},
                    local_dir=f"checkpoints/irl/ed_AIRL/iter_{n}/"
                )
                self.reward_functions.append(RewardEnsemble(
                    from_ckpt=self.get_reward_path(n)
                ))

            # Make our new best guess
            run_config = self.multi_env_config.build()
            run_config["n_envs"] = len(self.selected_envs)
            run_config["multiagent"]["policies"] = {str(k): PolicySpec() for k in range(run_config["n_envs"])}
            run_config["expert_rollouts"] = MultiAgentBatch(self.rollout_dict, env_steps=sum(
                [r.count for r in self.rollout_dict.values()]
            ))

            # register multi_env
            multi_env = make_multi_config_env(
                self.airl_config["env"],
                self.selected_envs
            )
            multi_env_name = f"ed_airl_iter_{n}_{self.airl_config['env']}"
            register_env(multi_env_name, multi_env)

            run_config["env"] = multi_env_name
            run_config["env_config"] = dict(num_agents=run_config.pop("n_envs"), selected_designs=self.selected_envs)
            run_config["reward_ckpt_path"] = True
            tune.run(
                make_multi_AIRL(self.algo_cls),
                name=self.out_path,
                config=run_config,
                num_samples=1,
                checkpoint_at_end=True,
                stop={"training_iteration": self.max_iter_multi_airl},
                local_dir=f"checkpoints/irl/ed_AIRL/iter_{n}/"
            )

            best_guess = RewardEnsemble(
                from_ckpt=self.get_reward_path(n, multi=True)
            )

    def base_env_init(self):
        BASE_ENV = self.all_env_params.pop(EDAIRL.BASE_ENV)
        self.selected_envs.append(BASE_ENV)


        self.rollout_dict["0"] = concat_samples(load_expert_rollouts(
            self.airl_config["env"],
            EDAIRL.BASE_ENV,
            n_rollouts=self.airl_config["n_expert_rollouts"] // self.n_iter
        ))

        run_config = self.single_env_config.build(BASE_ENV)
        run_config["expert_rollouts"] = self.rollout_dict["0"]
        run_config["reward_ckpt_path"] = True
        tune.run(
            make_AIRL(self.algo_cls),
            name=self.out_path,
            config=run_config,
            num_samples=1,
            checkpoint_at_end=True,
            stop={"training_iteration": self.max_iter_airl},
            local_dir="checkpoints/irl/ed_AIRL/iter_0/"
        )

        R_0 = RewardEnsemble(
            from_ckpt=self.get_reward_path(0)
        )

        tune.run(
            make_AIRL(self.algo_cls),
            name=self.out_path,
            config=run_config,
            num_samples=1,
            checkpoint_at_end=True,
            stop={"training_iteration": self.max_iter_airl},
            local_dir="checkpoints/irl/ed_AIRL/iter_0bis/"
        )

        best_guess = RewardEnsemble(
            from_ckpt=self.get_reward_path("0bis")
        )
        self.reward_functions.append(R_0)

        return best_guess

    def arbitrary_ed(
            self,
            best_guess: RewardEnsemble,
            single_env_reward_functions: List[RewardEnsemble],
    ):
        if best_guess is None:
            return np.random.choice(list(self.all_env_params.keys()))

        if self.subset_size is None:
            selected_envs = list(self.all_env_params.values())
            env_ids = np.arange(len(selected_envs))
        else:
            env_ids = np.random.choice(
                list(self.all_env_params.keys()), self.subset_size, replace=False
            )

            selected_envs = [self.all_env_params[k] for k in env_ids]

        delta = np.zeros(
            (len(selected_envs), len(single_env_reward_functions)),
            dtype=np.float32
        )

        regrets = np.zeros(
            len(selected_envs),
            dtype=np.float32
        )

        all_vs = [np.array([], dtype=np.float32) for _ in range(len(single_env_reward_functions))]

        for i, (env_id, env_params) in enumerate(zip(env_ids, selected_envs)):
            rl_config = self.single_env_config.build_base_rl(env_params)
            ckpt = eval_reward_function(
                env_params,
                0,
                self.single_env_config,
                max_iter=self.max_iter_rl,
                from_object=best_guess,
            ).checkpoint.to_air_checkpoint()
            pi_bar = load_policy(
                ckpt,
                self.single_env_config.rl_algo,
                rl_config
            )
            pi_bar.config["env_config"]["noisy_timelimit"]=False
            trajectories = make_rollouts(
                pi_bar,
                n_rollouts=self.n_eval_rollouts,
                env_creator=pi_bar.config["env"],
                perf_top_percent=1.
            )

            v_pi_bar = [
                np.array([
                    np.sum(reward_function(trajectory[SampleBatch.OBS], normalize=False)) for trajectory in trajectories
                ]) for reward_function in single_env_reward_functions
            ]

            for rew_id, reward_function in enumerate(single_env_reward_functions):
                if (env_id, rew_id) not in self.v_map:
                    ckpt = eval_reward_function(
                        env_params,
                        0,
                        self.single_env_config,
                        max_iter=self.max_iter_rl,
                        from_object=reward_function,
                    ).checkpoint.to_air_checkpoint()

                    pi_star = load_policy(
                        ckpt,
                        self.single_env_config.rl_algo,
                        rl_config
                    )
                    pi_star.config["env_config"]["noisy_timelimit"] = False
                    trajectories = make_rollouts(
                        pi_star,
                        n_rollouts=self.n_eval_rollouts,
                        env_creator=pi_star.config["env"],
                        perf_top_percent=1.
                    )

                    v_pi_star = np.array([
                            np.sum(reward_function(trajectory[SampleBatch.OBS], normalize=False)) for trajectory in trajectories
                        ])

                    self.v_map[(env_id, rew_id)] = v_pi_star

                all_vs[rew_id] = np.concatenate([all_vs[rew_id], self.v_map[(env_id, rew_id)], v_pi_bar[rew_id]])

                d = np.mean(self.v_map[(env_id, rew_id)]) - np.mean(v_pi_bar[rew_id])
                if len(single_env_reward_functions) == 1:
                    d = abs(d)

                delta[i, rew_id] = d

        regrets[:] = np.sum(
            delta /
            (
                np.array([np.max(all_vs[rew_id]) for rew_id in range(len(single_env_reward_functions))])
                - np.array([np.min(all_vs[rew_id]) for rew_id in range(len(single_env_reward_functions))])
            )[np.newaxis]
            , axis=1)


        logging.debug(f"Sampled envs:{env_ids}")
        logging.debug(f"and their respective regrets:{regrets}")
        logging.debug(f"v_map={delta}")

        return env_ids[np.argmax(regrets)]

    def get_reward_path(self, iter, multi=False):
        return (
            f"checkpoints/irl/ed_AIRL/iter_{iter}/{self.out_path}/best_guess_reward_function" if multi
            else
            f"checkpoints/irl/ed_AIRL/iter_{iter}/{self.out_path}/single_env_reward_function"
        )





if __name__ == '__main__':

    Fire(EDAIRL)