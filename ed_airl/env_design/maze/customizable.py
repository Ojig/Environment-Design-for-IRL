#from env_design import mujoco_utils
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np


class MazeED(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
            self, 
            sparse_reward=True,
            no_reward=False,
            
            length=0.6,
            wall_1=False,
            wall_1_x1=0,
            wall_1_y1=0,
            wall_1_x2=0,
            wall_1_y2=0,

            wall_2=False,
            wall_2_x1=0,
            wall_2_y1=0,
            wall_2_x2=0,
            wall_2_y2=0,

            wall_3=False,
            wall_3_x1=0,
            wall_3_y1=0,
            wall_3_x2=0,
            wall_3_y2=0,

            wall_4=False,
            wall_4_x1=0,
            wall_4_y1=0,
            wall_4_x2=0,
            wall_4_y2=0,

            disabled: bool = False,
            noisy_timelimit=True,
            **kwargs,
    ):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.length = length

        self.episode_length = 0
        self.done_cntr = 0
        self.finish_frames = 50

        self.discovered = [
            0,
            0,
            0
        ]

        from maze.point_mass_maze import point_mass_maze
        model = point_mass_maze(
            self.length,
            True,

            wall_1,
            wall_1_x1,
            wall_1_y1,
            wall_1_x2,
            wall_1_y2,
            wall_2,
            wall_2_x1,
            wall_2_y1,
            wall_2_x2,
            wall_2_y2,
            wall_3,
            wall_3_x1,
            wall_3_y1,
            wall_3_x2,
            wall_3_y2,
            wall_4,
            wall_4_x1,
            wall_4_y1,
            wall_4_x2,
            wall_4_y2,
        )

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)
        self.goal_pos = np.copy(self.model.body_pos)

        if disabled:
            self.model.actuator_ctrlrange[:] = 0.
            self.model.actuator_gear[:] = 0.

        self.noisy_timelimit = noisy_timelimit

    def step(self, a):
        vec_dists = [self.get_body_com("particle") - self.get_body_com("target_%d" % target)
                     for target in range(len(self.discovered))]

        reward_ctrl = - np.square(a).sum()

        # wait more steps before ending ep
        # if (all(self.discovered)):
        #     self.done_cntr += 1

        reward = 0.
        for i, dist in enumerate(vec_dists):
            reward_dist = np.linalg.norm(dist)
            if (reward_dist > 1e-8  # initial step has everything at 0,0,0
                    and reward_dist <= 0.075):
                self.discovered[i] = 1
                self.model.body_pos[2+i, :] = -1
                reward = 1.
                break

        reward += 0.001 * reward_ctrl #1/self.max_episode_length

        position_before = self.sim.data.qpos[:2]
        self.do_simulation(a, self.frame_skip)
        position_after = self.sim.data.qpos[:2]
        velocity = (position_after - position_before) / self.dt
        ob = self._get_obs(velocity)

        self.episode_length += 1
        return ob, reward, self.done_cntr>self.finish_frames, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        self.done_cntr = 0
        self.discovered = [
            0,
            0,
            0
                           ]
        self.model.body_pos[:] = self.goal_pos
        return self._get_obs([0,0])

    def _get_obs(self, velocity):

        return np.concatenate([
            self.get_body_com("particle")[:2],
            #velocity, # seems to make reward functions hard to optimize
            #self.state_vector(),
            self.discovered
            #self.get_body_com("target"),
        ])


if __name__ == '__main__':
    x = MazeED()
    print(dir(x.model))
    print(x.model.actuator_ctrlrange)
    print(x.model.geom_size)
    print(x.model.body_mass)