from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
import os
import pathlib
from env_design import mujoco_utils


class SwimmerED(SwimmerEnv):
    def __init__(
        self,
        *args,

        gravity: float = -9.81,

        mass_02: float = None,  # 34.6
        mass_03: float = None,  # 34.6
        mass_04: float = None,  # 34.6

        thickness_02_y: float = None,  # 0.5
        thickness_03_y: float = None,  # 0.15
        thickness_04_y: float = None,  # 0.145

        # actuator_01: float = None,
        # actuator_02: float = None,

        # joint_axis_01: int = None,
        # joint_axis_02: int = None,
        # joint_axis_03: int = None,
        #
        # joint_range_01: float = None,
        # joint_range_02: float = None,

            disabled: bool = False,
        noisy_timelimit = True,

        **kwargs,
    ):

        super().__init__(*args, **kwargs,
                         exclude_current_positions_from_observation=False,
                         )

        # Detects collision with the ground
        # self.model.opt.collision = 2

        # self.model.geom_friction[:, 0] = 0.1
        #
        if gravity is not None:
            self.model.opt.gravity[2] = gravity

        if mass_02 is not None:
            self.model.body_mass[1] = mass_02
        if mass_03 is not None:
            self.model.body_mass[2] = mass_03
        if mass_04 is not None:
            self.model.body_mass[3] = mass_04


        if thickness_02_y is not None:
            self.model.geom_size[1, 1] = thickness_02_y
        if thickness_03_y is not None:
            self.model.geom_size[2, 1] = thickness_03_y
        if thickness_04_y is not None:
            self.model.geom_size[3, 1] = thickness_04_y

        # Impossible env
        if disabled:
            mujoco_utils.set_as_impossible(self.model)
        self.noisy_timelimit = noisy_timelimit

        # if actuator_01 is not None:
        #     self.model.actuator_gear[:] *= actuator_01
        #
        # if actuator_02 is not None:
        #     self.model.actuator_gear[:] *= actuator_02
        #
        #
        # if joint_range_01 is not None:
        #     self.model.jnt_range[-2, :] *= joint_range_01
        # if joint_range_02 is not None:
        #     self.model.jnt_range[-1, :] *= joint_range_02
        #
        # if joint_axis_01 is not None:
        #     self.model.jnt_axis[3, :] = 0.
        #     self.model.jnt_axis[3, joint_axis_01] = 1.
        # if joint_axis_02 is not None:
        #     self.model.jnt_axis[4, :] = 0.
        #     self.model.jnt_axis[4, joint_axis_02] = 1.
        # if joint_axis_03 is not None:
        #     self.model.jnt_axis[5, :] = 0.
        #     self.model.jnt_axis[5, joint_axis_03] = 1.

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # x,y first 2 idxs
        # Observation auto normalization does not work well with our estimated rewards.
        # 10, 4
        observation[0] /= 10.
        observation[1] /= 4.
        return observation, reward, done, info


if __name__ == '__main__':

    x = SwimmerED()
