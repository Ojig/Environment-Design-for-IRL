from typing import Type, Union, List

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import override
from ray.rllib.utils.typing import EnvCreator, MultiAgentDict
from ray.tune.registry import ENV_CREATOR, _global_registry

from env_design import envs
from env_design.env_params import EnvParams


def make_multi_config_env(
    env_name_or_creator: Union[str, EnvCreator],
    env_params: List[EnvParams]
) -> Type["MultiAgentEnv"]:

    class MultiEnv(MultiAgentEnv):
        def __init__(self, config: EnvContext = None):
            MultiAgentEnv.__init__(self)
            self.env = env_name_or_creator
            if config is None:
                config = {}

            if isinstance(env_name_or_creator, str):
                env_creator_op = _global_registry.get(ENV_CREATOR, env_name_or_creator)

            else:
                env_creator_op = lambda config: env_name_or_creator(config)

            self.agents = [env_creator_op(env_params_i.get()) for env_params_i in env_params]

            # if isinstance(env_name_or_creator, str):
            #     self.agents = [gym.make(env_name_or_creator, **configs[i]) for i in range(num)]
            # else:
            #     self.agents = [env_name_or_creator(configs[i]) for i in range(num)]

            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space
            self._agent_ids = set(range(len(env_params)))

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}

            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            return all(self.action_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            return all(self.observation_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        @override(MultiAgentEnv)
        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.agents[0].render(mode)

    return MultiEnv



if __name__ == '__main__':

    def env_creator(config):
        return gym.make("LunarLander-v2", **config)


    env_design = envs.DESIGNS["LunarLander-v2"]

    env_design_params = env_design.load_config("handpicked")
    all_configs = [params.get() for params in env_design_params]
    x = make_multi_config_env(env_creator, all_configs)

    xx = x(config={
        "num_agents": len(env_design_params)
    })
    print([agent.gravity for agent in xx.agents])