import functools
from pathlib import Path

import fire
import numpy as np
import stable_baselines3
from imitation.algorithms.bc import reconstruct_policy
from imitation.policies.serialize import load_policy
from imitation.util import util
from imitation.data import rollout

from numpy.random import Generator, PCG64
import gym
import os

rng = np.random.Generator(PCG64())

from env_design import envs
from env_design.envs import (
    custom_swimmer,
    custom_ant,
    custom_hopper,
    custom_maze,
    custom_cheetah,
)

"""
Deployment of the GAIL and BC implementations from https://github.com/HumanCompatibleAI/imitation for our experiments.
"""



# cf. https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/mujoco.py
def _include_position_in_observation(cls):
    cls.__init__ = functools.partialmethod(
        cls.__init__,
        exclude_current_positions_from_observation=False,
    )
    return cls

def _no_early_termination(cls):
    cls.__init__ = functools.partialmethod(cls.__init__, terminate_when_unhealthy=False)
    return cls

@_include_position_in_observation
class CheetahWithPosition(custom_cheetah.CheetahED):
    """with position observation and early termination."""


# @_no_early_termination # We want fair comparison
@_include_position_in_observation
class HopperWithPosition(custom_hopper.HopperED):
    """with position observation and early termination."""


from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
# @_include_position_in_observation # already included
class SwimmerWithPosition(custom_swimmer.SwimmerED):
    """with position observation and early termination."""

    def step(self, action):
        return SwimmerEnv.step(self, action)


@_include_position_in_observation
class AntWithPosition(custom_ant.AntED):
    """with position observation and early termination."""



class MazeWithPosition(custom_maze.MazeED):
    """with position observation and early termination."""

envs_dict = {
    "MazeED": MazeWithPosition,
    "HopperED": HopperWithPosition,
    "AntED": AntWithPosition,
    "SwimmerED": SwimmerWithPosition,
    "CheetahED": CheetahWithPosition,
}


def run(
        env="MazeED",
        ckpt_dir="",
        ckpt_type="th"
):
    """

    Args:
        env: Which environment to run the experiment on
        ckpt_dir: Where is the imitation checkpoint located ?
        ckpt_type: Is it a model.zip checkpoint or a .th checkpoint ?
    """
    _, runs, _ = next(os.walk(ckpt_dir))
    print(envs_dict[env].__name__)

    env_cls = envs_dict[env]

    gym.envs.register(
         id="custom",
         entry_point='stable_baselines.run_gail_experiments:%s' % (env_cls.__name__),
         max_episode_steps=1000,
    )

    venv = util.make_vec_env("custom", rng=rng, n_envs=1)

    means= {
        "demo": {},
        "test": {}
    }

    env_design = envs.DESIGNS[env]

    for r in runs:
        print(r)
        r = ckpt_dir + "/" + r
        if ckpt_type == "th":
            r += "/final.th"
            local_policy = reconstruct_policy(r)
        else:
            r += "/checkpoints/final/gen_policy/model.zip"
            local_policy = load_policy("ppo", venv, path=r)

        for run_type in means:
            env_design_params = env_design.load_config(run_type)
            for design_id, env_params in enumerate(env_design_params):

                print(env_params.get())

                def gym_plug_custom_params(cls):
                    cls.__init__ = functools.partialmethod(
                        cls.__init__,
                        **env_params.get()
                    )
                    return cls

                @gym_plug_custom_params
                class WithPositionED(env_cls):
                    def __init__(self, *args, **k):
                        super().__init__(*args, **k)

                env_id = "customED%s%d" % (run_type, design_id)
                gym.envs.register(
                    id=env_id,
                    entry_point=WithPositionED,
                    max_episode_steps=250 if env == "MazeED" else 1000,
                )

                post_wrappers = [] # [video_wrapper_factory(Path("videos/" + env_id))]
                venv = util.make_vec_env(env_id, rng=rng, n_envs=16, post_wrappers=post_wrappers)
                sample_until = rollout.make_sample_until(None, 50)
                trajectories = rollout.generate_trajectories(local_policy, venv, sample_until, rng)

                if design_id not in means[run_type]:
                    means[run_type][design_id] = []

                means[run_type][design_id].append(rollout.rollout_stats(trajectories)["return_mean"])
                print("\n---\n")
                print(means)


if __name__ == '__main__':
    fire.Fire(run)

