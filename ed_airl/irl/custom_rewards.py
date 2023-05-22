from typing import Dict, Tuple, Optional

import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils import override
from ray.rllib.utils.typing import AgentID, PolicyID

from irl.discriminator import Discriminator
from irl.reward_function import RewardFunction, RewardEnsemble
from utils.norm import RunningNorm
from utils.python import partialclass


class CustomRewardCallBack(DefaultCallbacks):
    """
    Keep track of the actual rewards signal received by the learning policy
    """

    @override(DefaultCallbacks)
    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Episode,
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:

        super().on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id,
                                          policies=policies,postprocessed_batch=postprocessed_batch,
                                          original_batches=original_batches, **kwargs)

        episode.custom_metrics["GAN_rewards_normalized/"+policy_id] = np.sum(postprocessed_batch[SampleBatch.REWARDS])

        # This is not working with irl because we can't concat expert trajs which do not have reward distance.
        if "reward_distance" in postprocessed_batch:
            episode.custom_metrics["reward_distance"] = np.mean(postprocessed_batch["reward_distance"])


def wrap_rewards(policy_cls, shaped, compute_reward_distance=False):
    class CustomRewardsPolicy(policy_cls):
        def __init__(
                self,
                observation_space,
                action_space,
                config,
                existing_model=None,
                existing_inputs=None,
                rew_config=None,
                rew_weights=None

        ):

            self.shaped = shaped
            self.compute_reward_distance = compute_reward_distance
            if self.compute_reward_distance:
                self.normalizer = RunningNorm()


            self.rew_config = rew_config
            if self.shaped:
                self.reward_function = Discriminator(state_shape=observation_space.shape, **self.rew_config)
                if rew_weights is not None:
                    self.reward_function.set_weights(rew_weights)
            else:
                self.reward_function = RewardEnsemble(from_params=rew_weights)



            super(CustomRewardsPolicy, self).__init__(
                observation_space,
                action_space,
                config,
                existing_model,
                existing_inputs,
            )

        def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, Tuple["Policy", SampleBatch]]] = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
            # estimated_r = self.discriminator.compute_training_reward(
            #     np.concatenate([sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.NEXT_OBS][-1:]]),
            #     sample_batch[SampleBatch.ACTION_LOGP]
            #     )
            if not self.shaped:
                estimated_r = self.reward_function(
                    sample_batch[SampleBatch.OBS]
                )
            else:
                all_states = np.concatenate([sample_batch[SampleBatch.OBS], [sample_batch[SampleBatch.NEXT_OBS][-1]]])
                estimated_r = self.reward_function.compute_training_reward(
                    all_states, sample_batch[SampleBatch.ACTION_LOGP]
                )

            estimated_r = np.array(estimated_r, np.float32)
            if self.compute_reward_distance:
                sample_batch["reward_distance"] = np.abs(estimated_r-self.normalizer(sample_batch[SampleBatch.REWARDS]))

            sample_batch[SampleBatch.REWARDS] = estimated_r



            return super(CustomRewardsPolicy, self).postprocess_trajectory(
                sample_batch, other_agent_batches, episode
            )

    return CustomRewardsPolicy


def wrap_algo_with_custom_rewards(
        algo,
        policy_cls,
        reward_config,
        reward_params,
        shaped=False,
        compute_reward_distance=False,
):
    custom_reward_cls = wrap_rewards(
        policy_cls,
        shaped,
        compute_reward_distance
    )

    class CustomRewardsAlgo(algo):

        def get_default_policy_class(self, config):
            return partialclass(custom_reward_cls,
                                rew_config=reward_config,
                                rew_weights=reward_params)

    return CustomRewardsAlgo