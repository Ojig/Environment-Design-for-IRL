import functools
from pprint import pprint

import ray
import os, os.path
import errno

from config.builder import ConfigBuilder
from env_design.envs import register_ed_env


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class Default:

    def __init__(self, y=True, n_gpus=None, n_cpus=None):
        try:
            print()
            ray.init()
            ressources = ray.cluster_resources()
            pprint((ray.nodes(), ressources))
            print()
            if not y:
                try:
                    command = input("Press [ENTER] to run. ^D or ^C to cancel")
                except KeyboardInterrupt or EOFError:
                    exit(-1)
        except Exception as e:
            print(e)
            ressources = {}


        self.n_gpus = n_gpus if n_gpus is not None else (int(ressources["GPU"]) if "GPU" in ressources else 0)
        self.n_workers = n_cpus if n_cpus is not None else int(ressources["CPU"]) - 1

        self.config_to_build = None

    def setup(
            self,
            env=None,
            timelimit=1000,
            timelimit_noise=0,
            record_gifs=False,
            **kwargs
    ):
        if "num_gpus" not in kwargs:
            kwargs["num_gpus"] = self.n_gpus
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = self.n_workers

        assert env is not None, "need to specify env name"

        register_ed_env(
            env, timelimit, timelimit_noise, record_gifs
        )

        self.config_to_build = ConfigBuilder(
            env=env,
            **kwargs
        )


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')