from gym.envs.mujoco.hopper_v3 import HopperEnv

from env_design import mujoco_utils


class HopperED(HopperEnv):
    def __init__(
        self,
        *args,

        gravity: float = -9.81,

        mass_02: float = None,  # 3.66519143
        mass_03: float = None,  # 4.05789051
        mass_04: float = None,  # 2.7813567
        mass_05: float = None,  # 5.31557477

        thickness_02_y: float = None,  # 0.05
        thickness_03_y: float = None,  # 0.05
        thickness_04_y: float = None,  # 0.04
        thickness_05_y: float = None,  # 0.06


        disabled: bool = False,
        noisy_timelimit = True,
        **kwargs,
    ):



        super().__init__(*args, **kwargs)
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

        if thickness_02_y is not None:
            self.model.geom_size[1, 1] = thickness_02_y
        if thickness_03_y is not None:
            self.model.geom_size[2, 1] = thickness_03_y
        if thickness_04_y is not None:
            self.model.geom_size[3, 1] = thickness_04_y
        if thickness_05_y is not None:
            self.model.geom_size[4, 1] = thickness_05_y

        # Impossible env
        if disabled:
            mujoco_utils.set_as_impossible(self.model)

        self.noisy_timelimit = noisy_timelimit


if __name__ == '__main__':
    x = HopperED()
    print(dir(x.model))
    print(x.model.actuator_ctrlrange)
    print(x.model.geom_size)
    print(x.model.body_mass)