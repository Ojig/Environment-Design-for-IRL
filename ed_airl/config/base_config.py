

class EnvAlgoConfig(dict):
    BASE = dict()

    def __init__(
            self,
            env,
            algo,
            *args,
    ):
        super(EnvAlgoConfig, self).__init__()
        if env is None or algo is None:
            self.config_name = ""
        else:
            self.config_name = env + "__" + algo



    def pre_build(
            self,
            config_name=None
    ):
        config_name = config_name if config_name is not None else self.config_name

        if self.config_name != "":
            self.update(self.BASE)
            if hasattr(self, config_name):
                self.update(getattr(self, config_name))
            else:
                print()
                print(f"No config found for '{config_name}'")
                print()
            self.postprocess()

        return self

    def postprocess(
            self
    ):
        pass
