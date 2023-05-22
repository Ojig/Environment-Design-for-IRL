import pickle

from tensorflow.python.keras.layers import Dense

from irl.discriminator import tf

from utils.norm import RunningNorm


class RewardFunction(tf.keras.Model):

    def __reduce__(self):
        deserializer = RewardFunction
        serialized = (self.state_shape, self.name, self.layer_dims, self.default_activation, self.get_weights(), None,
                      self.normalize, self.frozen, (self.normalizer.running_mean, self.normalizer.running_var))
        return deserializer, serialized

    def __init__(
            self,
            state_shape=None,
            name: str = 'RewardFunction',
            layer_dims=[32, 32],
            default_activation='relu',
            weights=None,
            from_ckpt=None,
            normalize=True,
            frozen=False,
            norm_params=(0, 1),

            # Ignore other arguments that may be provided
            *args,
            **kwargs,
    ):
        mean, var = norm_params

        self.normalizer = RunningNorm(
            frozen=frozen,
            runing_mean=tf.cast(mean, tf.float32),
            running_var=tf.cast(var, tf.float32)
        ) if normalize else None

        if from_ckpt is not None:
            self.load_ckpt(from_ckpt)
        else:

            super().__init__(name=name)

            self.state_shape = state_shape
            self.layer_dims = layer_dims
            self.default_activation = default_activation

            self.reward_core = [tf.keras.layers.Dense(dim, activation=default_activation, dtype='float32') for dim in layer_dims]
            self.g = Dense(1, activation='linear', dtype='float32', name="g")

            #  Init values for eager execution
            self(
                tf.zeros((100,) + state_shape, dtype=tf.float32),
            )

            if weights is not None:
                self.set_weights(weights)

            self.config = {
                    "name": self.name,
                    "layer_dims": self.layer_dims,
                    "default_activation": self.default_activation
            }

    def __call__(self, states, normalize=True):
        """
        :param states:
        :return: estimated rewards from the states
        """

        for layer in self.reward_core:
            states = layer(states)
        r = self.g(states)[:, 0]
        if normalize and self.normalizer is not None:
            r = self.normalizer(r)

        return r

    def load_ckpt(self, path):

        with open(path + ".pkl", 'rb') as f:
            params = pickle.load(f)
        self.__init__(
            **params
        )

    def get_params(self):
        return {
            "state_shape"       : self.state_shape,
            "name"              : self.name,
            "layer_dims"        : self.layer_dims,
            "default_activation": self.default_activation,
            "weights"           : self.get_weights(),
            "frozen"            : self.normalizer.frozen,
            "norm_params"       : (self.normalizer.running_mean.numpy(), self.normalizer.running_var.numpy())

        }
    def get_norm(self):
        return (self.normalizer.running_mean.numpy(), self.normalizer.running_var.numpy())


class RewardEnsemble:
    
    def __init__(
            self,
            name: str = 'RewardFunctionEnsemble',
            from_ckpt=None,
            from_params=None,
            from_list=None,
            dont_normalize=False,
    ):
        self.reward_funcs = []
        self.dont_normalize = dont_normalize
        if from_ckpt is not None:
            self.load_ckpt(from_ckpt)
        elif from_params:
            self.set_params(from_params)
        elif from_list:
            self.reward_funcs = from_list
        else:
            pass

    @property
    def config(self):
        return self.reward_funcs[0].config
    def get_params(self):
        return [f.get_params() for f in self.reward_funcs]

    def get_norms(self):
        return [f.get_norm() for f in self.reward_funcs]

    def update_params(self, params):
        if len(self.reward_funcs) > 0:
            for p, rew_func in zip(params, self.reward_funcs):
                rew_func.set_weights(p["weights"])
        else:
            self.set_params(params)

    def set_params(self, params):
        self.reward_funcs = []
        for params_f in params:
            if self.dont_normalize:
                params_f["frozen"] = True
                params_f["norm_params"] = (0., 1.)
            self.reward_funcs.append(RewardFunction(**params_f))
    def __call__(self, states, normalize=True):
        """
        :param states:
        :return: estimated rewards from the states
        """
        if len(self.reward_funcs) == 0:
            return tf.zeros(len(states), dtype = tf.float32)
        r = 0.
        for f in self.reward_funcs :
            r += f(states, normalize=normalize)

        return r / tf.cast(len(self.reward_funcs), tf.float32) #+ np.random.normal(0., 1, size=r.shape)


    def load_ckpt(self, path):
        # Loads list of rewards
        with open(path + ".pkl", 'rb') as f:
            reward_list = pickle.load(f)
        self.set_params(reward_list)

