from time import time

import numpy as np
from ray.rllib.policy.sample_batch import MultiAgentBatch


class RolloutQueue:
    def __init__(self, n_samples):
        self.rollouts = []
        self.sizes = []
        self.n_max = n_samples

    def __getitem__(self, item):
        return self.rollouts[item]

    def put(self, item):
        if isinstance(item, list):

            self.rollouts.extend(item)
            self.sizes.append(len(item))
        else:

            self.rollouts.append(item)
            self.sizes.append(1)

        if self.length() > self.n_max:
            del self.rollouts[:self.sizes[0]]
            self.sizes.pop(0)

    def length(self):
        return len(self.sizes)

    def size(self):
        return sum(self.sizes)

    def __repr__(self):
        return self.rollouts.__repr__()

    def concatenated(self):
        if self.length() == 0:
            return []
        elif self.length() == 1:
            return self.rollouts[0]
        else:
            return concat_samples(self.rollouts)


from ray.rllib.policy.sample_batch import concat_samples, SampleBatch


class ConcatenatedRolloutQueue:
    def __init__(self, n_samples):
        self.rollouts = None
        self.sizes = []
        self.n_max = n_samples

    def __getitem__(self, item):
        return self.rollouts[item]

    def put(self, item):
        # Samples should be concatenated already
        if isinstance(item, list):
            print("samples not concatenated ?")
        else:
            if self.rollouts is None:
                self.rollouts = item
            else:
                self.rollouts = concat_samples([self.rollouts, item])

            if isinstance(item, MultiAgentBatch):
                self.sizes.append([
                    i.count for i in item.policy_batches.values()])
            else:
                self.sizes.append(item.count)

        if self.length() > self.n_max:

            if isinstance(self.rollouts, MultiAgentBatch):
                for i, k in enumerate(self.rollouts.policy_batches):
                    self.rollouts.policy_batches[k] = self.rollouts.policy_batches[k][self.sizes[0][i]:]
                self.rollouts.count = sum([b.count for b in self.rollouts.policy_batches.values()])
            else:
                self.rollouts = self.rollouts[self.sizes[0]:]

            self.sizes.pop(0)

    def length(self):
        return len(self.sizes)

    def size(self):
        return self.rollouts.count

    def size_bytes(self):
        return self.rollouts.size_bytes()

    def __repr__(self):
        return self.rollouts.__repr__()

    def concatenated(self):
        return self.rollouts


if __name__ == '__main__':

    q = ConcatenatedRolloutQueue(100)

    for _ in range(110):
        x = SampleBatch(obs=np.random.random(np.random.randint(100,128)))

        # x = MultiAgentBatch(
        #     {
        #         str(i): xx for i, xx in enumerate(x)
        #     },
        #     env_steps = sum([len(xx) for xx in x])
        # )

        t = time()
        q.put(x)

        # print(q.rollouts.count, q.size_bytes(), time()-t)
        # print([h.count for h in q.rollouts.policy_batches.values()])

        print(q.rollouts["obs"][0])





