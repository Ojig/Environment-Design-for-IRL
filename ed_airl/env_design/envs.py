import os

import imageio
from gym.wrappers import TimeLimit, OrderEnforcing
from mujoco_py import MujocoException
from ray.tune import registry
import numpy as np

import gym
from env_design.hopper import hopperED
from env_design.halfcheetah import cheetahED
from env_design.swimmer import swimmerED
from env_design.ant import antED
from env_design.walker import walkerED
from env_design.maze import mazeED

from env_design.halfcheetah import customizable as custom_cheetah
from env_design.swimmer import customizable as custom_swimmer
from env_design.hopper import customizable as custom_hopper
from env_design.ant import customizable as custom_ant
from env_design.walker import customizable as custom_walker
from env_design.maze import customizable as custom_maze

class MujocoExceptionWrapper(gym.Wrapper):
    # We got big values when simulating with customized envs.
    def __init__(self, env):
        super().__init__(env)
        self._last_state = np.zeros(self.observation_space.shape)

    def step(self, action):
        try:
            observation, reward, done, info = self.env.step(action)
            self._last_state = observation
        except MujocoException as e:
            observation = self._last_state
            reward = 0.
            done = True
            info = {}

        return observation, reward, done, info


class GIFRecorder(gym.Wrapper):
    def __init__(self, env, path):
        super().__init__(env)
        self._last_state = None
        self.ep_obs = []
        self.ep_id = 0
        self.dur = 0.01
        self.width = 250
        self.height = 200
        self.path = path
        self.only_one = True

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if len(self.ep_obs) < 500:
            obs = self.env.render(mode='rgb_array', width=self.width, height=self.height)
            assert obs.shape == (self.height, self.width, 3), obs.shape  # height first!
            self.ep_obs.append(obs)

        if done and self.only_one:
            self.only_one = False
            name = self.path if isinstance(self.path, str) is not None else 'visual'
            suffix = ""
            ext = ".gif"
            i = 1
            while os.path.exists(name+suffix + ext):
                i += 1
                suffix = f"_{i}"
            path = name+suffix + ext

            from utils.python import mkdir_p
            directory = os.path.dirname(path)
            if directory: mkdir_p(os.path.dirname(path))

            with imageio.get_writer(path, mode='I', duration=self.dur, fps=24) as writer:
                for obs_np in self.ep_obs:
                    writer.append_data(obs_np)

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.ep_obs = []
        self.ep_id += 1
        return self.env.reset(**kwargs)



class NoisyTimeLimit(gym.Wrapper):
    """
    Just a tool to desync workers when the episode length is fixed.
    """
    def __init__(self, env, max_episode_steps=None, noise_size=1):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._max_episode_steps_base = max_episode_steps
        self.noise_size = noise_size
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._max_episode_steps = self._max_episode_steps_base \
                                  + np.random.randint(-self.noise_size, self.noise_size+1) * int(self.noisy_timelimit)
        return self.env.reset(**kwargs)



DESIGNS = {
    hopperED.HopperDesign.env: hopperED.HopperDesign(),
    cheetahED.CheetahDesign.env: cheetahED.CheetahDesign(),
    swimmerED.SwimmerDesign.env: swimmerED.SwimmerDesign(),
    antED.AntDesign.env: antED.AntDesign(),
    walkerED.WalkerDesign.env: walkerED.WalkerDesign(),
    mazeED.MazeDesign.env: mazeED.MazeDesign()

}


REGISTERED = {
    hopperED.HopperDesign.env: hopperED.HopperDesign.env,
    cheetahED.CheetahDesign.env: cheetahED.CheetahDesign.env,
    swimmerED.SwimmerDesign.env: swimmerED.SwimmerDesign.env,
    antED.AntDesign.env: antED.AntDesign.env,
    walkerED.WalkerDesign.env: walkerED.WalkerDesign.env,
    mazeED.MazeDesign.env: mazeED.MazeDesign.env
}


ENV_MAKERS = {
    hopperED.HopperDesign.env: custom_hopper.HopperED,
    cheetahED.CheetahDesign.env: custom_cheetah.CheetahED,
    swimmerED.SwimmerDesign.env: custom_swimmer.SwimmerED,
    antED.AntDesign.env: custom_ant.AntED,
    walkerED.WalkerDesign.env: custom_walker.WalkerED,
    mazeED.MazeDesign.env: custom_maze.MazeED
}


def register_ed_env(
        name,
        timelimit=1000,
        timelimit_noise=200,
        save_gifs=False,
):
    if save_gifs:
        def maker(conf):
            return GIFRecorder(
                TimeLimit(
                    MujocoExceptionWrapper(
                        OrderEnforcing(
                            ENV_MAKERS[name](**conf)
                    )), max_episode_steps=timelimit), save_gifs # <- path to save gifs
            )
    else:
        def maker(conf):
            return NoisyTimeLimit(
                MujocoExceptionWrapper(
                    OrderEnforcing(
                        ENV_MAKERS[name](**conf)
                )), max_episode_steps=timelimit, noise_size=timelimit_noise
            )

    registry.register_env(name, maker)

    # stable baselines

    # gym.envs.register(
    #     id=name,
    #     entry_point='gym.envs.classic_control:LqrEnv',
    #     max_episode_steps=150,
    #     kwargs={'size': 1, 'init_state': 10., 'state_bound': np.inf},
    # )


if __name__ == '__main__':
    x = NoisyTimeLimit(
        MujocoExceptionWrapper(
            OrderEnforcing(
                custom_swimmer.SwimmerED()
            ),
        ),
        max_episode_steps=1000,
        noise_size=200)
    print(x.noisy_timelimit)