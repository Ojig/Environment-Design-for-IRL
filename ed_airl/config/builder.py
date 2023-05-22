from copy import deepcopy

from config.irl_config import IRLConfig
from config.rl_config import RLConfig
from env_design.envs import ENV_MAKERS

class ConfigBuilder(dict):

    def __init__(
            self,

            num_gpus=0,
            num_workers=0,

            rl_algo=None,
            irl_algo=None,
            env=None,

            # additional overriding args
            **kwargs
    ):
        super(ConfigBuilder, self).__init__()
        self.rl_algo = rl_algo
        self.irl_algo = irl_algo

        self.rl_config = RLConfig(env, rl_algo, irl_algo)
        self.irl_config = IRLConfig(env, irl_algo)

        self.update(
            num_gpus=num_gpus,
            num_workers=num_workers,
            env=env,
        )

        self.cli_args = kwargs

    def build_base_rl(
            self,
            env_params,
            **kwargs,
        ):
        base = self.rl_config.rl_config.pre_build()

        base.update(
            **self
        )
        if env_params is not None:
            base.update(env_config=env_params.get())

        base.update(**kwargs)


        for cli_arg in self.cli_args:
            if cli_arg in base:
                base[cli_arg] = self.cli_args[cli_arg]

        return base

    def build_base_irl(
            self
    ):

        base = self.irl_config.pre_build()
        base.update(
            **self
        )
        for cli_arg in self.cli_args:
            if cli_arg in base:
                base[cli_arg] = self.cli_args[cli_arg]

        base.postprocess()
        return base

    def build(
            self,
            env_params=None,  # Mandatory, to ensure proper initialization
            *args,
            **kwargs
    ):
        new = deepcopy(self)
        rl = self.rl_config.pre_build()
        if env_params is not None:
            rl.update(
                env_config=env_params.get()
            )
        irl = self.build_base_irl()
        new.update(**rl)
        new.update(**irl)

        return dict(new)
