import pickle

import numpy as np
from keras.optimizers import Adam
from ray.rllib import SampleBatch
from ray.rllib.utils import try_import_tf
from tensorflow.python.keras.layers import Dense

from utils.norm import RunningNorm

tf1, tf, tfv = try_import_tf()


class Discriminator(tf.keras.Model):

    def __reduce__(self):
        deserializer = Discriminator
        serialized = (self.state_shape, self.name, self.layer_dims, self.default_activation, self.lr, self.get_weights(),
                      self.normalize, self.n_envs)
        return deserializer, serialized

    def __init__(
            self,
            state_shape,
            name: str = 'Discriminator',
            layer_dims=[32, 32],
            default_activation='relu',
            gamma=0.99,
            lr=1e-3,
            weights=None,
            normalize=True,
            n_envs=1,
        ):
        super().__init__(name=name)
        self.normalizer = RunningNorm() if normalize else None
        self.state_shape = state_shape
        self.layer_dims = layer_dims
        self.default_activation = default_activation
        self.lr = lr
        self.n_envs = n_envs

        self.gamma = gamma
        self.reward_core = [tf.keras.layers.Dense(dim, activation=default_activation, dtype='float32') for dim in layer_dims]
        self.value_core = [[tf.keras.layers.Dense(dim, activation=default_activation, dtype='float32') for dim in layer_dims]
                            for _ in range(n_envs)]
        self.g = Dense(1, activation='linear', dtype='float32', name="g")
        self.h = [Dense(1, activation='linear', dtype='float32', name="h") for _ in range(n_envs)]

        self.optim = Adam(lr)

        #  Init values for eager execution
        for i in range(self.n_envs):
            self(
                tf.zeros((100,) + state_shape, dtype=tf.float32),
                tf.zeros((99), dtype=tf.float32),
                idx=i
            )

        if weights is not None:
            self.set_weights(weights)


    def __call__(
            self,
            features,
            logprobs,
            idx=0,
    ):
        """
        features [timesteps, state]
        """
        rewards = self.compute_reward(features[:-1])
        values = self.compute_value(idx, features)

        # Define log p_tau(a|s) = r + gamma * V(s') - V(s)

        log_p_tau = rewards + self.gamma * values[1:] - values[:-1]

        log_pq = tf.reduce_logsumexp([log_p_tau, logprobs], axis=0)

        return tf.exp(log_p_tau - log_pq)

    def compute_reward(self, states):
        for layer in self.reward_core:
            states = layer(states)

        r = self.g(states)[:, 0]
        #if self.normalizer is not None:
        #    r = self.normalizer(r, update=False)
        return r

    def compute_training_reward(self,
                                state_all,
                                action_logp,
                                ):
        rewards = self.compute_reward(state_all[:-1])
        # values = self.compute_value(state_all)
        r = rewards # + self.gamma * values[1:] - values[:-1] - action_logp
        return r

    def compute_value(self, idx, states):
        for layer in self.value_core[idx]:
            states = layer(states)
        return self.h[idx](states)[:, 0]

    def train(
            self,
            input_dicts,
            labels,
            gpu,
            log=False,
    ):
        loss, losses = self._train(
            input_dicts,
            labels,
            gpu,
        )

        if log:
            tf.summary.scalar(name="GAN_loss", data=loss)

        return loss, losses

    def loss(
            self,
            idx,
            states,
            next_states,
            logprobs,
            labels,
            **kwargs,
    ):
        rewards = self.compute_reward(states)
        values = self.compute_value(idx, states)
        next_values = self.compute_value(idx, next_states)

        # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
        log_p_tau = rewards + self.gamma * next_values - values

        log_pq = tf.reduce_logsumexp([log_p_tau, logprobs], axis=0)

        loss = -tf.reduce_mean(
            labels * (log_p_tau - log_pq) + (1 - labels) * (logprobs - log_pq)
        )

        return loss

    @tf.function
    def _train(
            self,
            input_dicts,
            labels,
            gpu,
    ):
        """
        https://github.com/justinjfu/inverse_rl/blob/9609933389459a3a54f5c01d652114ada90fa1b3/inverse_rl/models/airl_state.py#L7
        """
        losses = []
        with tf.GradientTape() as tape:
            with tf.device("/device:GPU:{}".format(gpu) if gpu >= 0 else "/device:CPU:0"):
                for i, input_dict in enumerate(input_dicts):
                    loss = self.loss(
                        i, **input_dict, labels=labels
                    )
                    losses = tf.concat([losses, [loss]], axis=0)
            loss_mean = tf.reduce_mean(losses)
            grad = tape.gradient(loss_mean, self.trainable_variables)

            self.optim.apply_gradients(zip(grad, self.trainable_variables))

        return loss_mean, losses

    def get_reward_weights(self):
        weights = []
        for layer in self.reward_core:
            weights.extend(layer.get_weights())

        weights.extend(self.g.get_weights())

        return weights


    def get_params(self):
        return {
            "state_shape": self.state_shape,
            "name": self.name,
            "layer_dims": self.layer_dims,
            "default_activation": self.default_activation,
            "weights": self.get_reward_weights(),
            #"norm_params": (self.normalizer.running_mean, self.normalizer.running_var)
        }

    def set_norm(self, m, v):
        self.normalizer.running_mean = m
        self.normalizer.running_var = v

    def save_as_reward_function(self, path):

        params = self.get_params()

        with open(path + '.pkl',
                  'wb+') as f:

            pickle.dump(params, f)