import gym

from maze.customizable import MazeED
gym.envs.register(
     id='MazeED',
     entry_point='maze.customizable:MazeED',
     max_episode_steps=250,
)