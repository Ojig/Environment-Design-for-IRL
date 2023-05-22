from dataclasses import dataclass
from typing import NamedTuple

from env_design.env_params import EnvParams, EnvDesign
from utils.distribution import loguniform
from numpy.random import uniform, randint

@dataclass
class Params(EnvParams):
    gravity: float = -9.81 

    mass_02: float = None   # 34.6
    mass_03: float = None   # 34.6
    mass_04: float = None   # 34.6

    thickness_02_x: float = None   # 0.1
    thickness_03_x: float = None
    thickness_04_x: float = None

    thickness_02_y: float = None   # 0.5
    thickness_03_y: float = None
    thickness_04_y: float = None

    # actuator_01: float = None
    # actuator_02: float = None
    #
    # joint_axis_01: int = None
    # joint_axis_02: int = None
    # joint_axis_03: int = None
    #
    # joint_range_01: float = None
    # joint_range_02: float = None

    disabled: bool = False



@dataclass
class ParamsDefault(EnvParams):
    
    gravity: float = -9.81 

    mass_02: float = None   # 34.6
    mass_03: float = None   # 34.6
    mass_04: float = None   # 34.6

    thickness_02_x: float = None   # 0.1
    thickness_03_x: float = None 
    thickness_04_x: float = None 

    thickness_02_y: float = None   # 0.5
    thickness_03_y: float = None
    thickness_04_y: float = None

    # actuator_01: float = None
    # actuator_02: float = None
    #
    # joint_axis_01: int = None
    # joint_axis_02: int = None
    # joint_axis_03: int = None
    #
    # joint_range_01: float = None
    # joint_range_02: float = None

    disabled: bool = False


class Bounds(NamedTuple):
    gravity: tuple = uniform, -4, -14

    mass_02: tuple = loguniform, 10., 50.
    mass_03: tuple = loguniform, 10., 50.
    mass_04: tuple = loguniform, 10., 50.

    thickness_02_x: tuple = uniform, 0.025, 0.2  # 0.1
    thickness_03_x: tuple = uniform, 0.025, 0.2
    thickness_04_x: tuple = uniform, 0.025, 0.2

    thickness_02_y: tuple = uniform, 0.125, 1.  # 0.5
    thickness_03_y: tuple = uniform, 0.125, 1.
    thickness_04_y: tuple = uniform, 0.125, 1.

    # actuator_01: float = uniform, 0.8, 1.2
    # actuator_02: float = uniform, 0.8, 1.2
    #
    # joint_axis_01: int = randint, 0, 3
    # joint_axis_02: int = randint, 0, 3
    # joint_axis_03: int = randint, 0, 3
    #
    # joint_range_01: float = uniform, 0.5, 1.3
    # joint_range_02: float = uniform, 0.5, 1.3



class SwimmerDesign(EnvDesign):
    env = "SwimmerED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "swimmer/"

    EXPERT_EPISODE_RETURN = 300.