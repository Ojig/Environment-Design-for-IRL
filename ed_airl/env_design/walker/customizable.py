from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from env_design import mujoco_utils


class WalkerED(Walker2dEnv):
    def __init__(
        self,
        *args,

        gravity: float = -9.81,

        mass_02: float = None,  # 3.53
        mass_03: float = None,  # 3.93
        mass_04: float = None,  # 2.71
        mass_05: float = None,  # 2.94
        mass_06: float = None,  # 3.92
        mass_07: float = None,  # 2.71
        mass_08: float = None,  # 2.94
            
        thickness_02_y: float = None,  # 0.2
        thickness_03_y: float = None,  # 0.225
        thickness_04_y: float = None,  # 0.25
        thickness_05_y: float = None,  # 0.1
        thickness_06_y: float = None,  # 0.225
        thickness_07_y: float = None,  # 0.25
        thickness_08_y: float = None,  # 0.1

        disabled: bool = False,
        noisy_timelimit = True,

        **kwargs,
    ):



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

        if thickness_02_y is not None:
            self.model.geom_size[1, 1] = thickness_02_y
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

        # Impossible env
        if disabled:
            mujoco_utils.set_as_impossible(self.model)
        self.noisy_timelimit = noisy_timelimit



if __name__ == '__main__':

    x = Walker2dEnv()
    print(x.model.body_mass)
    print(x.model.geom_size)
    #print(x.model.geom_size, x.model.body_mass, x.model.opt)
