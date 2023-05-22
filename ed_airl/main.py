import pickle
import fire
import numpy as np
from imitation.data.types import Trajectory
from imitation.data import types as imit_types
from ray.rllib import SampleBatch

from env_design import envs
from irl.ed_airl import EDAIRL
from rl import expert
from rl.expert import save_rollouts
from irl import airl, multi_airl
from utils.python import Default

import ray


class Main(Default):

    def generate_designs(
            self,
            env="HopperED",
            n_sample=3,
            params_to_sample_from=None, # By default, sample each tunable parameters
            name="designs",
    ):
        """
        Generates a {name}.json file holding different environment configurations inside the respective env folder in
        env_design, randomly generates configurations according to env_design/{env}.py

        Args:
            env: environment to generate designs for
            n_sample: how many environment configuration to generate
            params_to_sample_from: None, or list of the names of the parameters to randomize
            name: name of the environment set.
        """

        print()
        print(f"Generating design config file for {env=}, {n_sample=}, {params_to_sample_from=}...")
        print()
        env_design = envs.DESIGNS[env]


        env_design.generate_config(
            n_sample,
            params_to_sample_from,
            name,
        )
        print(f"Generated design config file at env_design/{env_design.path}{name}.json .")

    def generate_experts(
            self,
            env='HopperED',
            # cf the implemented algos on
            # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
            rl_algo='PPO',
            env_designs='handpicked',
            which=None,
            **kwargs,
    ):
        """
        Generates an expert policy in checkpoints/experts/ for each environment in the specified environment design file.
        Args:
            env: The environment we want to generate and expert for
            rl_algo: Which policy optimization to use
            env_designs: For which environment set do we want to generate experts ?
            which: an optional list of environment indexes we want to generate an expert for.
                ie.: which=[0] means that we generate an expert for the base environment only.
                Otherwise, processes all environments in the configuration file
            **kwargs: Additional hyperparameters
        """

        self.setup(
            env=env,
            rl_algo=rl_algo,
            **kwargs
        )

        env_design = envs.DESIGNS[env]
        '''
        We generate an expert for each env design found in the config file
        '''
        env_design_params = env_design.load_config(env_designs)
        for design_id, env_params in enumerate(env_design_params):
            if which is None or design_id in which:
                print(f"\nObtaining expert policy for design {env_params}...")
                print(f"{design_id}/{len(env_design_params)} generated.\n")

                if env_params.disabled:
                    pass
                else:
                    expert.get_expert_for(
                        env_params=env_params,
                        config=self.config_to_build,
                    )

    def save_rollouts(
            self,
            rl_algo="PPO",
            env="HopperED",
            env_designs="demo",
            n_rollouts=100,
            perf_top_percent=1,
            deterministic=True,
            which=None,
            **kwargs
    ):
        """
        Saves rollout for an existing expert, in checkpoints/experts/rollouts/
        Args:
            rl_algo: Which policy optimization algorithm was used to train the expert ?
            env_designs: The name of the environment design file, followed by "=",
                followed by the environment index in the file
            n_rollouts: How many rollouts to save in the file ?
            perf_top_percent: Select the top perf_top_percent% rollouts in terms of total return.
            deterministic: Generate the rollouts with a deterministic expert policy
            which: an optional list of environment indexes we want to save rollouts for.
                ie.: which=[0] means that we process the base environment only.
                Otherwise, processes all environments found in the configuration file.
            **kwargs: Additional hyperparameters
        """
        self.setup(
            rl_algo=rl_algo,
            env=env,
            explore=not deterministic,
            **kwargs
        )

        env_design = envs.DESIGNS[env]
        all_params = env_design.load_config(env_designs)

        for i, params in enumerate(all_params):
            if which is None or i in which:
                save_rollouts(
                    params,
                    self.config_to_build,
                    n_rollouts,
                    perf_top_percent=perf_top_percent
                )

    def airl(
            self,
            rl_algo='PPO',
            env='HopperED',
            design='handpicked=0',
            max_iter=220,
            from_saved_rollouts=True,
            n_runs=1,
            **kwargs,
    ):
        """
        Runs state-only AIRL
        Args:
            rl_algo: Policy optimization algorithm to use for the generating policy
            env: For which environment do we want to learn the reward function ?
            design: The name of the environment design file, followed by "=",
                followed by the environment index in the file:
            max_iter: How many discriminator-generator rounds do we run ?
            from_saved_rollouts: If true, use the rollouts saved in checkpoints/experts/rollouts/,
                else generate rollouts with the existing experts in checkpoints/experts
            **kwargs: Additional hyperparameters
        """
        self.setup(
            env=env,
            rl_algo=rl_algo,
            irl_algo="AIRL",
            **kwargs,
        )

        env_design = envs.DESIGNS[env]
        config_name, params_id = design.split(sep='=')
        params = env_design.load_config(config_name)[int(params_id)]
        airl.run(
            env_params=params,
            config=self.config_to_build,
            max_iter=max_iter,
            n_runs=n_runs,
            from_saved_rollouts=from_saved_rollouts
        )

    def ed_airl(
            self,
            env="CheetahED",
            rl_algo="PPO",
            max_iter_airl=220,
            max_iter_multi_airl=350,
            max_iter_rl=220,
            n_eval_rollouts=100,
            n_iter=5,
            subset_size=5,
            out_path="experimenting",
            env_designs="demo",
            seed=None,
            **kwargs
    ):
        """
        Runs ED-AIRL
        Args:
            env: Environment to run ED-AIRL on
            rl_algo: Policy optimization algorithm to use
            max_iter_airl: How many iterations we run single environment AIRL for each point estimates
            max_iter_multi_airl: How many iterations we run multi-environment AIRL for each best guess estimates
            max_iter_rl: How many iterations we run policy optimization when estimating policy values
            n_eval_rollouts: How many trajectories we take to estimate the policy values
            n_iter: How many expert-learner rounds do we do ?
            subset_size: How many environments do we sample from the demo set each time we want to select an environment.
            out_path: Name of the run, generates folders in checkpoints/irl/ed_AIRL/. The last estimate is found at
                checkpoints/irl/ed_AIRL/iter_{n_iter-1}/{out_path}/best_guess_reward_function.pkl
            env_designs: The demo set we want to use for environment selection
            seed: Numpy random seed.
            **kwargs: For convenience, can pass algo hyperparameters.
        """
        if seed is not None:
            np.random.seed(seed)

        env_design = envs.DESIGNS[env]
        all_env_params = env_design.load_config(env_designs)

        EDAIRL(
            all_env_params,
            env,
            rl_algo,
            self,
            max_iter_airl,
            max_iter_multi_airl,
            max_iter_rl,
            n_eval_rollouts,
            n_iter,
            subset_size,
            out_path,

            **kwargs
        )(
            # kwargs
        )

    def multi_airl(
            self,
            env='CheetahED',
            rl_algo='PPO',
            max_iter=150,
            env_designs='handpicked',
            environment_indexes=None,
            from_saved_rollouts=True,
            seed=None,
            **kwargs
    ):
        """
        Runs AIRL-ME
        Args:
            env: Environment to run ED-AIRL on
            rl_algo: Policy optimization algorithm to use
            max_iter: Maximum amount of learning iterations
            env_designs: The demo set we want to use for environment selection
            environment_indexes: List of the indexes of the environments we want to learn from, if None, randomly
                selects environments from the set.
            from_saved_rollouts: If true, use the rollouts saved in checkpoints/experts/rollouts/,
                else generate rollouts with the existing experts in checkpoints/experts
            **kwargs: For convenience, can pass algo hyperparameters.
        """

        if seed is not None:
            np.random.seed(seed)

        self.setup(
            env=env,
            rl_algo=rl_algo,
            irl_algo="MULTI_AIRL",

            **kwargs
        )

        env_design = envs.DESIGNS[env]
        all_params = env_design.load_config(env_designs)

        multi_airl.run(
            all_params,
            self.config_to_build,
            max_iter=max_iter,
            deterministic_choice=environment_indexes,
            from_saved_rollouts=from_saved_rollouts
        )

    def convert_rollouts_to_npz(
            self,
            rollout_path,
            amount=40
    ):
        """
        Convenience for using our expert trajectories with the HumanCompatibleAI imitation library
        """

        with open(rollout_path, 'rb') as f:
            rollouts = pickle.load(f)

        trajectories = [
            Trajectory(
                obs=np.concatenate([rollout[SampleBatch.OBS], rollout[SampleBatch.NEXT_OBS][-1:]]),
                acts=rollout[SampleBatch.ACTIONS],
                infos=None,
                terminal=True
            ) for rollout in rollouts[:amount]
        ]

        trajectories_path = rollout_path[:-3] + "npz"

        imit_types.save(trajectories_path, trajectories)




if __name__ == '__main__':

    exit(fire.Fire(Main))
