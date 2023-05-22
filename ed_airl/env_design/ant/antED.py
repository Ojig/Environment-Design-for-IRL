from dataclasses import dataclass
from typing import NamedTuple

from env_design.env_params import EnvParams, EnvDesign

@dataclass
class Params(EnvParams):
    gravity: float = -9.81

    mass_02: float = None  # 0.327
    mass_03: float = None  # 0.037
    mass_04: float = None  # 0.037
    mass_05: float = None  # 0.065
    mass_06: float = None  # 0.037
    mass_07: float = None  # 0.037
    mass_08: float = None  # 0.065
    mass_09: float = None  # 0.037
    mass_10: float = None  # 0.037
    mass_11: float = None  # 0.065
    mass_12: float = None  # 0.037
    mass_13: float = None  # 0.037
    mass_14: float = None  # 0.065

    thickness_head: float = None  # 0.25 # head
    thickness_03_y: float = None  # 0.14
    thickness_04_y: float = None  # 0.14
    thickness_05_y: float = None  # 0.28
    thickness_06_y: float = None  # 0.14
    thickness_07_y: float = None  # 0.14
    thickness_08_y: float = None  # 0.28
    thickness_09_y: float = None  # 0.14
    thickness_10_y: float = None  # 0.14
    thickness_11_y: float = None  # 0.28
    thickness_12_y: float = None  # 0.14
    thickness_13_y: float = None  # 0.14
    thickness_14_y: float = None  # 0.28

    disable_leg1: bool = False
    disable_leg2: bool = False
    disable_leg3: bool = False
    disable_leg4: bool = False

    disabled: bool = False
    

@dataclass
class ParamsDefault(EnvParams):
    
    gravity: float = -9.81 

    mass_02: float = None  # 0.327
    mass_03: float = None  # 0.037
    mass_04: float = None  # 0.037
    mass_05: float = None  # 0.065
    mass_06: float = None  # 0.037
    mass_07: float = None  # 0.037
    mass_08: float = None  # 0.065
    mass_09: float = None  # 0.037
    mass_10: float = None  # 0.037
    mass_11: float = None  # 0.065
    mass_12: float = None  # 0.037
    mass_13: float = None  # 0.037
    mass_14: float = None  # 0.065

    thickness_head: float = None  # 0.25 # head
    thickness_03_y: float = None  # 0.14
    thickness_04_y: float = None  # 0.14
    thickness_05_y: float = None  # 0.28
    thickness_06_y: float = None  # 0.14
    thickness_07_y: float = None  # 0.14
    thickness_08_y: float = None  # 0.28
    thickness_09_y: float = None  # 0.14
    thickness_10_y: float = None  # 0.14
    thickness_11_y: float = None  # 0.28
    thickness_12_y: float = None  # 0.14
    thickness_13_y: float = None  # 0.14
    thickness_14_y: float = None  # 0.28

    disable_leg1: bool = False
    disable_leg2: bool = False
    disable_leg3: bool = False
    disable_leg4: bool = False

    disabled: bool = False


class Bounds(NamedTuple):
    gravity: tuple = (-7, -18) # (-2, -15)

    mass_02: tuple = (0.03, 0.7)  # 0.327
    mass_03: tuple = (0.003, 0.07)  # 0.037
    mass_04: tuple = (0.003, 0.07)  # 0.037
    mass_05: tuple = (0.006, 0.14)  # 0.065
    mass_06: tuple = (0.003, 0.07)  # 0.037
    mass_07: tuple = (0.003, 0.07)  # 0.037
    mass_08: tuple = (0.006, 0.14)  # 0.065
    mass_09: tuple = (0.003, 0.07)  # 0.037
    mass_10: tuple = (0.003, 0.07)  # 0.037
    mass_11: tuple = (0.006, 0.14)  # 0.065
    mass_12: tuple = (0.003, 0.07)  # 0.037
    mass_13: tuple = (0.003, 0.07)  # 0.037
    mass_14: tuple = (0.006, 0.14)  # 0.065

    thickness_head: tuple = (0.025, 0.5)   # 0.25 # head
    thickness_03_y: tuple = (0.014, 0.28)  # 0.14
    thickness_04_y: tuple = (0.014, 0.28)  # 0.14
    thickness_05_y: tuple = (0.028, 0.56)  # 0.28
    thickness_06_y: tuple = (0.014, 0.28)  # 0.14
    thickness_07_y: tuple = (0.014, 0.28)  # 0.14
    thickness_08_y: tuple = (0.028, 0.56)  # 0.28
    thickness_09_y: tuple = (0.014, 0.28)  # 0.14
    thickness_10_y: tuple = (0.014, 0.28)  # 0.14
    thickness_11_y: tuple = (0.028, 0.56)  # 0.28
    thickness_12_y: tuple = (0.014, 0.28)  # 0.14
    thickness_13_y: tuple = (0.014, 0.28)  # 0.14
    thickness_14_y: tuple = (0.028, 0.56)  # 0.28



class AntDesign(EnvDesign):
    env = "AntED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "ant/"

    EXPERT_EPISODE_RETURN = 2408.