import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.spaces import Box

from env_design import mujoco_utils


class AntED(AntEnv):
    def __init__(
        self,
        *args,

        gravity: float = -9.81,

        mass_02: float = None,  # 0.327
        mass_03: float = None,  # 0.037
        mass_04: float = None,  # 0.037
        mass_05: float = None,  # 0.065
        mass_06: float = None,  # 0.037
        mass_07: float = None,  # 0.037
        mass_08: float = None,  # 0.065
        mass_09: float = None,  # 0.037
        mass_10: float = None,  # 0.037
        mass_11: float = None,  # 0.065
        mass_12: float = None,  # 0.037
        mass_13: float = None,  # 0.037
        mass_14: float = None,  # 0.065
            
        thickness_head: float = None,  # 0.25 # head
        thickness_03_y: float = None,  # 0.14
        thickness_04_y: float = None,  # 0.14
        thickness_05_y: float = None,  # 0.28
        thickness_06_y: float = None,  # 0.14
        thickness_07_y: float = None,  # 0.14
        thickness_08_y: float = None,  # 0.28
        thickness_09_y: float = None,  # 0.14
        thickness_10_y: float = None,  # 0.14
        thickness_11_y: float = None,  # 0.28
        thickness_12_y: float = None,  # 0.14
        thickness_13_y: float = None,  # 0.14
        thickness_14_y: float = None,  # 0.28

        disable_leg1: bool = False,
        disable_leg2: bool = False,
        disable_leg3: bool = False,
        disable_leg4: bool = False,


        disabled: bool = False,

        include_velocity=False,
        noisy_timelimit=True,
        **kwargs,
    ):

        self.disable_leg1 = disable_leg1
        self.disable_leg2 = disable_leg2
        self.disable_leg3 = disable_leg3
        self.disable_leg4 = disable_leg4
        self.noisy_timelimit = noisy_timelimit



        super().__init__(*args,
                         #exclude_current_positions_from_observation=False,
                         **kwargs)

        # self.ctrl_cost_weight = 0.

        if gravity is not None:
            self.model.opt.gravity[2] = gravity

        if mass_02 is not None:
            self.model.body_mass[1] = mass_02
        if mass_03 is not None:
            self.model.body_mass[2] = mass_03
        if mass_04 is not None:
            self.model.body_mass[3] = mass_04
        if mass_05 is not None:
            self.model.body_mass[4] = mass_05
        if mass_06 is not None:
            self.model.body_mass[5] = mass_06
        if mass_07 is not None:
            self.model.body_mass[6] = mass_07
        if mass_08 is not None:
            self.model.body_mass[7] = mass_08
        if mass_09 is not None:
            self.model.body_mass[8] = mass_09
        if mass_10 is not None:
            self.model.body_mass[9] = mass_10
        if mass_11 is not None:
            self.model.body_mass[10] = mass_11
        if mass_12 is not None:
            self.model.body_mass[11] = mass_12
        if mass_13 is not None:
            self.model.body_mass[12] = mass_13
        if mass_14 is not None:
            self.model.body_mass[13] = mass_14


        if thickness_head is not None:
            self.model.geom_size[1, 0] = thickness_head
        if thickness_03_y is not None:
            self.model.geom_size[2, 1] = thickness_03_y
        if thickness_04_y is not None:
            self.model.geom_size[3, 1] = thickness_04_y
        if thickness_05_y is not None:
            self.model.geom_size[4, 1] = thickness_05_y
        if thickness_06_y is not None:
            self.model.geom_size[5, 1] = thickness_06_y
        if thickness_07_y is not None:
            self.model.geom_size[6, 1] = thickness_07_y
        if thickness_08_y is not None:
            self.model.geom_size[7, 1] = thickness_08_y
        if thickness_09_y is not None:
            self.model.geom_size[8, 1] = thickness_09_y
        if thickness_10_y is not None:
            self.model.geom_size[9, 1] = thickness_10_y
        if thickness_11_y is not None:
            self.model.geom_size[10, 1] = thickness_11_y
        if thickness_12_y is not None:
            self.model.geom_size[11, 1] = thickness_12_y
        if thickness_13_y is not None:
            self.model.geom_size[12, 1] = thickness_13_y
        if thickness_14_y is not None:
            self.model.geom_size[13, 1] = thickness_14_y

        # Impossible env
        if disabled:
            mujoco_utils.set_as_impossible(self.model)

        if disable_leg1:
            self.model.actuator_ctrlrange[:2] = 0.
            self.model.actuator_gear[:2] = 0.
        if disable_leg2:
            self.model.actuator_ctrlrange[2:4] = 0.
            self.model.actuator_gear[2:4] = 0.
        if disable_leg3:
            self.model.actuator_ctrlrange[4:6] = 0.
            self.model.actuator_gear[4:6] = 0.
        if disable_leg4:
            self.model.actuator_ctrlrange[6:8] = 0.
            self.model.actuator_gear[6:8] = 0.





if __name__ == '__main__':

    x = AntED()
    print(x.model.body_mass)
    print(x.model.geom_size)
    print(x.model.actuator_ctrlrange)
