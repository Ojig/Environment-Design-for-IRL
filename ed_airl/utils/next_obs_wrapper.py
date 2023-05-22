from typing import Optional, Dict, Tuple

from ray.rllib import SampleBatch
from ray.rllib.utils.typing import AgentID


def wrap_policy(cls):

    class NEXT_OBS_Wrapper(cls):


        def postprocess_trajectory(
                self,
                sample_batch: SampleBatch,
                other_agent_batches: Optional[Dict[AgentID, Tuple["Policy", SampleBatch]]] = None,
                episode: Optional["Episode"] = None,
        ) -> SampleBatch:
            # view requirement
            # just access the field once

            sample_batch[SampleBatch.NEXT_OBS]

            return super(NEXT_OBS_Wrapper, self).postprocess_trajectory(
                sample_batch, other_agent_batches, episode
            )

        __name__ = cls.__name__ + "_NEXT_OBS_Wrapped"

    

    return NEXT_OBS_Wrapper



def wrap_algo(
        algo
):
    class EnforceNextObs(algo):
        def get_default_policy_class(self, config):
            return wrap_policy(super(EnforceNextObs, self).get_default_policy_class(config))


    return EnforceNextObs
