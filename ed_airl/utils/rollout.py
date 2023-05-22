import dataclasses
import logging
import warnings
from typing import Optional, List, Union, Dict

import numpy as np

from ray.rllib import Policy
from ray.rllib.execution import ParallelRollouts
from ray.rllib.utils import FilterManager
from ray.tune.registry import _Registry, ENV_CREATOR

import ray
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.policy.sample_batch import (
    SampleBatch,
    MultiAgentBatch,
    concat_samples,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)

import gym
from env_design import envs

_global_registry = _Registry(prefix="global")

def visualize(
        policy,
        n_rollouts=5,
):
    policy.config["render_env"] = False
    policy.config["num_workers"] = 0
    policy.config["num_envs_per_worker"] = 1
    make_rollouts(
        policy,
        env_creator=policy.config["env"],
        n_rollouts=n_rollouts,
        filtering=False,
        perf_top_percent=1,
    )


def make_rollouts(
        policy: Policy,
        n_rollouts=1,
        env_creator=None,
        filtering=True,
        perf_top_percent=0.25,
) -> List[SampleBatch]:
    print(policy.config["explore"])
    policy.config["batch_mode"] = "complete_episodes"
    policy.config["rollout_fragment_length"] = 1000
    policy.config["env_config"]["noisy_timelimit"] = False
    # policy.config["vtrace"] = False
    if isinstance(env_creator, str):
        env_creator_op = _global_registry.get(ENV_CREATOR, env_creator)
        # env_creator_op = lambda config: gym.make(env, **config)
        # gym.make(env, **env_params)
    else:
        env_creator_op = lambda config: env_creator(config)

    workers = WorkerSet(
        env_creator=env_creator_op,
        policy_class=policy.__class__,
        trainer_config=policy.config,
        num_workers=policy.config["num_workers"],
    )
    workers.local_worker().set_weights({'default_policy': policy.get_weights()})
    workers.sync_weights()
    workers.local_worker().filters = policy.config["filters"]
    FilterManager.synchronize(
        policy.config["filters"],
        workers.remote_workers(),
        update_remote=True,
        timeout_seconds=5,
    )
    rollouts = ParallelRollouts(
        workers,
        mode="async",
    )
    print(policy.config)
    samples: list[SampleBatch] = []
    if filtering:
        total_rollouts = n_rollouts / perf_top_percent
    else:
        total_rollouts = n_rollouts

    while len(samples) < total_rollouts:
        r = next(rollouts).split_by_episode()
        samples.extend(r)
    total_returns = [-np.sum(e[SampleBatch.REWARDS]) for e in samples]
    arg_sorted = np.argsort(total_returns)
    workers.stop()


    selected = np.array(samples)[arg_sorted][:n_rollouts]
    np.random.shuffle(selected)
    print([-np.sum(e[SampleBatch.REWARDS]) for e in selected])

    return list(selected)

    # while len(samples) < n_rollouts:
    #     print(len(samples), "/", n_rollouts)
    #     episodes = next(rollouts).split_by_episode()
    #     for episode in episodes:
    #         if (
    #                 not filtering
    #                 or
    #                 # Filtering bad trajectories
    #                 np.sum(episode[SampleBatch.REWARDS]) >= envs.DESIGNS[policy.config["env"]].EXPERT_EPISODE_RETURN * perf_top_percent
    #         ):
    #
    #             samples.append(episode)
    #         else:
    #             print(np.sum(episode[SampleBatch.REWARDS]), envs.DESIGNS[policy.config["env"]].EXPERT_EPISODE_RETURN * perf_top_percent)
    #
    # workers.stop()
    #
    # return samples[:n_rollouts]



def sample_batch_indexation_helper(
        sample_batch: SampleBatch,
        indexes: np.ndarray
):
    return SampleBatch(
        **{
            k: v[indexes] for k, v in sample_batch.items()
        }
    )


@ExperimentalAPI
def ma_synchronous_parallel_sample(
    *,
    worker_set: WorkerSet,
    policies=["default_policy"],
    max_steps_per_policy=1000,
    concat: bool = True,

) -> Union[Dict, SampleBatchType]:

    # Only allow one of `max_agent_steps` or `max_env_steps` to be defined.

    steps = {
        p: 0 for p in policies
    }
    all_sample_batches = {
        p: [] for p in policies
    }

    # Stop collecting batches as soon as one criterium is met.
    while min(steps.values())\
            < max_steps_per_policy:
        # No remote workers in the set -> Use local worker for collecting
        # samples.
        if not worker_set.remote_workers():
            sample_batches = [worker_set.local_worker().sample()]
        # Loop over remote workers' `sample()` method in parallel.
        else:
            sample_batches = ray.get(
                [worker.sample.remote() for worker in worker_set.remote_workers()]
            )
        # Update our counters for the stopping criterion of the while loop.
        for b in sample_batches:
            for policy, batch in b.policy_batches.items():
                if steps[policy] < max_steps_per_policy:
                    steps[policy] += batch.count
                    all_sample_batches[policy].append(batch)

    for p in all_sample_batches:
        all_sample_batches[p] = concat_samples(all_sample_batches[p])

    if concat is True:
        full_batch = MultiAgentBatch(all_sample_batches, env_steps=sum(steps.values()))
        # Discard collected incomplete episodes in episode mode.
        # if max_episodes is not None and episodes >= max_episodes:
        #    last_complete_ep_idx = len(full_batch) - full_batch[
        #        SampleBatch.DONES
        #    ].reverse().index(1)
        #    full_batch = full_batch.slice(0, last_complete_ep_idx)
        return full_batch
    else:
        return all_sample_batches






@dataclasses.dataclass(frozen=True)
class ImitTrajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, length trajectory_len."""

    terminal: bool
    """Does this trajectory (fragment) end in a terminal state?

    Episodes are always terminal. Trajectory fragments are also terminal when they
    contain the final state of an episode (even if missing the start of the episode).
    """

    def __len__(self) -> int:
        """Returns number of transitions, equal to the number of actions."""
        return len(self.acts)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ImitTrajectory):
            return False

        dict_self, dict_other = dataclasses.asdict(self), dataclasses.asdict(other)
        # Trajectory objects may still have different keys if different subclasses
        if dict_self.keys() != dict_other.keys():
            return False

        if len(self) != len(other):
            # Short-circuit: if trajectories are of different length, then unequal.
            # Redundant as later checks would catch this, but speeds up common case.
            return False

        for k, self_v in dict_self.items():
            other_v = dict_other[k]
            if k == "infos":
                # Treat None equivalent to sequence of empty dicts
                self_v = [{}] * len(self) if self_v is None else self_v
                other_v = [{}] * len(other) if other_v is None else other_v
            if not np.array_equal(self_v, other_v):
                return False

        return True

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1",
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action: "
                f"{len(self.infos)} != {len(self.acts)}",
            )
        if len(self.acts) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

    def __setstate__(self, state):
        if "terminal" not in state:
            warnings.warn(
                "Loading old version of Trajectory."
                "Support for this will be removed in future versions.",
                DeprecationWarning,
            )
            state["terminal"] = True
        self.__dict__.update(state)