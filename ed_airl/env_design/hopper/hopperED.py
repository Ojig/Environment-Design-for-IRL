from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from env_design.env_params import EnvParams, EnvDesign

@dataclass
class Params(EnvParams):

    gravity: float = -9.81
    mass_02: float = None # 3.66519143
    mass_03: float = None # 4.05789051
    mass_04: float = None # 2.7813567
    mass_05: float = None # 5.31557477

    thickness_02_y: float = None  # 0.05
    thickness_03_y: float = None  # 0.05
    thickness_04_y: float = None  # 0.04
    thickness_05_y: float = None  # 0.06

    disabled: bool = False

    # friction_01: float = None
    # friction_02: float = None
    # friction_03: float = None
    # friction_04: float = None
    # friction_05: float = None


@dataclass
class ParamsDefault(EnvParams):
    
    gravity: float = -9.81

    mass_02: float = None # 3.66519143
    mass_03: float = None # 4.05789051
    mass_04: float = None # 2.7813567
    mass_05: float = None # 5.31557477

    thickness_02_y: float = None  # 0.05
    thickness_03_y: float = None  # 0.05
    thickness_04_y: float = None  # 0.04
    thickness_05_y: float = None  # 0.06

    disabled: bool = False

    # friction_01: float = None
    # friction_02: float = None
    # friction_03: float = None
    # friction_04: float = None
    # friction_05: float = None


class Bounds(NamedTuple):
    gravity: tuple = np.random.uniform, -7, -13

    mass_02: tuple = np.random.uniform, 0.36, 7  # 3.66519143
    mass_03: tuple = np.random.uniform, 0.4, 8  # 4.05789051
    mass_04: tuple = np.random.uniform, 0.27, 5  # 2.7813567
    mass_05: tuple = np.random.uniform, 0.53, 11  # 5.31557477

    thickness_02_y: tuple = np.random.uniform, 0.03, 0.6  # 0.3
    thickness_03_y: tuple = np.random.uniform, 0.02, 0.45  # 0.225
    thickness_04_y: tuple = np.random.uniform, 0.02, 0.5  # 0.25
    thickness_05_y: tuple = np.random.uniform, 0.02, 0.5  # 0.195



class HopperDesign(EnvDesign):
    env = "HopperED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "hopper/"

    EXPERT_EPISODE_RETURN = 2500.