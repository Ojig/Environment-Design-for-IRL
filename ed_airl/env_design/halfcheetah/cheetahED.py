from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from env_design.env_params import EnvParams, EnvDesign

@dataclass
class Params(EnvParams):
    gravity: float = -9.81 

    mass_02: float = None   # 6.3
    mass_03: float = None   # 1.53
    mass_04: float = None   # 1.58
    mass_05: float = None   # 1.069
    mass_06: float = None   # 1.42
    mass_07: float = None   # 1.178
    mass_08: float = None   # 0.849

    thickness_02_y: float = None   # 0.5
    thickness_03_y: float = None   # 0.15
    thickness_04_y: float = None   # 0.145
    thickness_05_y: float = None   # 0.15
    thickness_06_y: float = None   # 0.094
    thickness_07_y: float = None   # 0.133
    thickness_08_y: float = None   # 0.106
    thickness_09_y: float = None   # 0.07

    disabled: bool = False


@dataclass
class ParamsDefault(EnvParams):
    
    gravity: float = -9.81 

    mass_02: float = None   # 6.3
    mass_03: float = None   # 1.53
    mass_04: float = None   # 1.58
    mass_05: float = None   # 1.069
    mass_06: float = None   # 1.42
    mass_07: float = None   # 1.178
    mass_08: float = None   # 0.849

    thickness_02_y: float = None   # 0.5
    thickness_03_y: float = None   # 0.15
    thickness_04_y: float = None  # 0.145
    thickness_05_y: float = None  # 0.15
    thickness_06_y: float = None  # 0.094
    thickness_07_y: float = None  # 0.133
    thickness_08_y: float = None  # 0.106
    thickness_09_y: float = None  # 0.07

    disabled: bool = False


class Bounds(NamedTuple):
    gravity: tuple = np.random.uniform, -7, -18 # (-2, -15

    mass_02: tuple = np.random.uniform, 0.6 , 1.26  # 6.3
    mass_03: tuple = np.random.uniform, 0.15 , 3.  # 1.53
    mass_04: tuple = np.random.uniform, 0.15 , 3.  # 1.58
    mass_05: tuple = np.random.uniform, 0.15 , 3.  # 1.069
    mass_06: tuple = np.random.uniform, 0.15 , 3.  # 1.42
    mass_07: tuple = np.random.uniform, 0.15 , 3.  # 1.178
    mass_08: tuple = np.random.uniform, 0.15 , 3.  # 0.849

    thickness_02_y: tuple = np.random.uniform, 0.1 , 0.8  # 0.5
    thickness_03_y: tuple = np.random.uniform, 0.02 , 0.45  # 0.15
    thickness_04_y: tuple = np.random.uniform, 0.02 , 0.45  # 0.145
    thickness_05_y: tuple = np.random.uniform, 0.02 , 0.45  # 0.15
    thickness_06_y: tuple = np.random.uniform, 0.015 , 0.3  # 0.094
    thickness_07_y: tuple = np.random.uniform, 0.02 , 0.4  # 0.133
    thickness_08_y: tuple = np.random.uniform, 0.015 , 0.3  # 0.106
    thickness_09_y: tuple = np.random.uniform, 0.01 , 0.2  # 0.07



class CheetahDesign(EnvDesign):
    env = "CheetahED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "halfcheetah/"

    EXPERT_EPISODE_RETURN = 3400.