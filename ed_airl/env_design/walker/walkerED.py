from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from env_design.env_params import EnvParams, EnvDesign

@dataclass
class Params(EnvParams):
    gravity: float = -9.81

    mass_02: float = None  # 3.53
    mass_03: float = None  # 3.93
    mass_04: float = None  # 2.71
    mass_05: float = None  # 2.94
    mass_06: float = None  # 3.92
    mass_07: float = None  # 2.71
    mass_08: float = None  # 2.94

    thickness_02_y: float = None  # 0.2
    thickness_03_y: float = None  # 0.225
    thickness_04_y: float = None  # 0.25
    thickness_05_y: float = None  # 0.1
    thickness_06_y: float = None  # 0.225
    thickness_07_y: float = None  # 0.25
    thickness_08_y: float = None  # 0.1

    disabled: bool = False
    

@dataclass
class ParamsDefault(EnvParams):
    
    gravity: float = -9.81

    mass_02: float = None  # 3.53
    mass_03: float = None  # 3.93
    mass_04: float = None  # 2.71
    mass_05: float = None  # 2.94
    mass_06: float = None  # 3.92
    mass_07: float = None  # 2.71
    mass_08: float = None  # 2.94

    thickness_02_y: float = None  # 0.2
    thickness_03_y: float = None  # 0.225
    thickness_04_y: float = None  # 0.25
    thickness_05_y: float = None  # 0.1
    thickness_06_y: float = None  # 0.225
    thickness_07_y: float = None  # 0.25
    thickness_08_y: float = None  # 0.1

    disabled: bool = False


class Bounds(NamedTuple):
    gravity: tuple = np.random.uniform, -7, -18 # (-2, -15

    mass_02: tuple = np.random.uniform, 0.35, 7.  # 3.53
    mass_03: tuple = np.random.uniform, 0.4, 8.  # 3.93
    mass_04: tuple = np.random.uniform, 0.27, 5.4  # 2.71
    mass_05: tuple = np.random.uniform, 0.29, 6.  # 2.94
    mass_06: tuple = np.random.uniform, 0.39, 8.  # 3.92
    mass_07: tuple = np.random.uniform, 0.27, 5.4  # 2.71
    mass_08: tuple = np.random.uniform, 0.29, 6.  # 2.94

    thickness_02_y: tuple = np.random.uniform, 0.02, 0.4  # 0.2
    thickness_03_y: tuple = np.random.uniform, 0.0225, 0.45  # 0.225
    thickness_04_y: tuple = np.random.uniform, 0.025, 0.5  # 0.25
    thickness_05_y: tuple = np.random.uniform, 0.01, 0.2  # 0.1
    thickness_06_y: tuple = np.random.uniform, 0.0225, 0.45  # 0.225
    thickness_07_y: tuple = np.random.uniform, 0.025, 0.5  # 0.25
    thickness_08_y: tuple = np.random.uniform, 0.01, 0.2



class WalkerDesign(EnvDesign):
    env = "WalkerED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "walker/"

    EXPERT_EPISODE_RETURN = 2673.