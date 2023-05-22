import json
from abc import ABC
from dataclasses import dataclass, asdict
from typing import NamedTuple, Type, List

import numpy as np

from utils.distribution import loguniform


@dataclass
class EnvParams:
    params_id: int = 0

    def get(
            self
    ):
        d = asdict(self)
        del d['params_id']
        return d


class EnvDesign(ABC):
    params : Type[EnvParams]
    default : Type[EnvParams]
    bounds : Type[NamedTuple]
    path: str

    EXPERT_EPISODE_RETURN: np.float16

    def __init__(
            self
    ):
        pass

    def get_param_samples(
            self,
            n_samples_per_param,
            params_to_sample_from=None,
            random=True,
    ):
        '''
        samples n_samples env configs.
        '''

        bounds = self.bounds()
        env_params = [self.default(params_id=0)]  # add in the original params
        params_to_sample_from = bounds._asdict().keys() if params_to_sample_from is None else params_to_sample_from
        if random:
            for n in range(n_samples_per_param - 1):
                random_params = {
                    k: apply(*getattr(bounds, k)) for k in params_to_sample_from
                }
                random_params["params_id"] = n + 1
                env_params.append(
                    self.params(**random_params)
            )
        else:
            env_param_grid = {}

            if isinstance(n_samples_per_param, int):
                n_samples_per_param = np.full(len(params_to_sample_from), fill_value=n_samples_per_param)

            for num_samples, param in zip(n_samples_per_param, params_to_sample_from):
                env_param_grid[param] = np.linspace(*getattr(bounds, param), num=num_samples)


            n_samples = np.prod(n_samples_per_param)
            for env_num in range(n_samples):
                params = {
                    k: param[(env_num // (n_samples_per_param[i] ** i)) % n_samples_per_param[i]] for i, (k, param) in
                    enumerate(env_param_grid.items())
                }
                params["params_id"] = env_num + 1

                env_params.append(
                    self.params(
                        **params
                    )
                )

        return env_params

    def generate_config(
            self,
            n_samples: int,
            params_to_sample_from: List[str],
            name: str = "designs"
    ):
        params = self.get_param_samples(n_samples, params_to_sample_from)
        with open(f"env_design/{self.path}{name}.json", mode="w+") as f:
            json.dump(
                {i: asdict(param) for i, param in enumerate(params)}
                , f, indent=4)

    def load_config(
            self,
            config_name: str = "designs"
    ):

        with open(f"env_design/{self.path}{config_name}.json", mode="r") as f:
            config = json.load(f)
            params = [self.params(**conf) for conf in config.values()]

        return params


def apply(func, *args):
    return func(*args)