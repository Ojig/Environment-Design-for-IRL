from typing import Dict, Tuple

import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils import try_import_tf, override
from ray.rllib.utils.typing import AgentID, PolicyID

tf1, tf, _ = try_import_tf()

class RunningNorm:
    def __init__(
            self,
            frozen=False,
            runing_mean=0.,
            running_var=1.,
    ):
        self.running_mean = runing_mean
        self.running_var = running_var
        self.eps = 1e-4
        self.count = 0
        self.frozen = frozen

    def update(self, estimated):
        batch_mean = tf.reduce_mean(estimated)
        batch_var = tf.math.reduce_variance(estimated)
        batch_count = tf.cast(estimated.shape[0], tf.float32)

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean += delta * batch_count / tot_count
        self.running_var *= self.count
        self.running_var += batch_var * batch_count
        self.running_var += tf.math.square(delta) * self.count * batch_count / tot_count
        self.running_var /= tot_count

        self.count += batch_count

    def __call__(self, estimated, update=True):
        if not self.frozen and update:
            self.update(estimated)
        return (estimated - self.running_mean) / tf.math.sqrt(self.running_var + self.eps)


class RunningNorm2:
    def __init__(self):
        self.frozen = False
        self.eps = 1e-8
        self.count = 0
        self.size = 100
        self.means = np.full((self.size,), fill_value=np.nan, dtype=np.float32)
        self.stds = np.full((self.size,), fill_value=np.nan, dtype=np.float32)

    @property
    def running_mean(self):
        return np.nanmean(self.means)

    @property
    def running_std(self):
        return np.nanmean(self.stds)


    def update(self, estimated):
        batch_mean = tf.reduce_mean(estimated)
        batch_std = tf.math.sqrt(tf.math.reduce_variance(estimated))
        self.means[self.count % self.size] = batch_mean
        self.stds[self.count % self.size] = batch_std
        self.count += 1

    def __call__(self, estimated):
        if not self.frozen:
            self.update(estimated)
        return (estimated - self.running_mean) / (self.running_std + self.eps)



class RewardNormalizerCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()

        self.norm = RunningNorm()


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
        postprocessed_batch[SampleBatch.REWARDS][:] = self.norm(postprocessed_batch[SampleBatch.REWARDS])

