import fire
import numpy as np


from env_design import envs
from irl.discriminator import Discriminator
from irl.reward_function import RewardFunction, RewardEnsemble
from rl.expert import generate_rollouts
from rl.with_estimated_rewards import eval_reward_function
from utils.python import Default

import ray


class Test(Default):

    def visualize_policy(
            self,
            rl_algo="PPO",
            env="HopperED",
            design="designs=3",
            n_rollouts=1,
            deterministic=True,
            full_path=None,
            name="visualization",
            **kwargs,
    ):

        self.setup(
            rl_algo=rl_algo,
            env=env,
            explore=not deterministic,
            record_gifs=name,
            **kwargs,
        )

        env_design = envs.DESIGNS[env]
        config_name, params_id = design.split(sep='=')
        params = env_design.load_config(config_name)[int(params_id)]

        generate_rollouts(
            params,
            self.config_to_build,
            n_rollouts,
            filtering=False,
            render=True,
            full_path=full_path
        )

    def eval_reward_function(
            self,
            env="HopperED",
            env_designs="handpicked",
            reward_design_id=0,
            design_id=0,
            irl_algo="AIRL",
            rl_algo="PPO",
            from_path=None,
            max_iter=350,

            **kwargs
    ):
        self.setup(
            env=env,
            rl_algo=rl_algo,
            irl_algo=irl_algo,

            **kwargs
        )

        env_design = envs.DESIGNS[env]
        params = env_design.load_config(env_designs)[design_id]

        eval_reward_function(
            params,
            reward_design_id,
            config=self.config_to_build,
            max_iter=max_iter,
            from_path=from_path
        )

    def eval_on(
            self,
            env="HopperED",
            env_set="demo",
            irl_algo="AIRL",
            rl_algo="PPO",
            from_path=None,
            max_iter=220,
            name="single_env_demo",
            which=[],
            **kwargs
    ):
        self.setup(
            env=env,
            rl_algo=rl_algo + "_estimated",
            irl_algo=irl_algo,

            **kwargs
        )

        env_design = envs.DESIGNS[env]
        params = env_design.load_config(env_set)
        for id_, p in enumerate(params):
            if len(which) == 0 or id_ in which:
                if not p.disabled:
                    eval_reward_function(
                        p,
                        id_,
                        config=self.config_to_build,
                        max_iter=max_iter,
                        from_path=from_path,
                        name=name
                    )


    def reward_heatmap(
            self,
            path: str,
            precision=20,
            name="heatmap"
    ):

        # Maze env specific

        import matplotlib.pyplot as plt

        # coords_x = np.linspace(-0.355, 0.255, precision)
        # -0.1, 0.6
        coords_y = np.linspace(-0.1, 0.6, precision)

        xy, yx = np.meshgrid(
            coords_y, coords_y
        )

        all_tests = np.zeros(
            (precision, precision, 5), dtype=np.float32
        )
        if path.endswith(".pkl"):
            path = path[:-4]
        reward_function = RewardEnsemble(
            from_ckpt=path
        )
        # reward_function.reward_funcs = reward_function.reward_funcs[-5:]
        for r in reward_function.reward_funcs:
            r.normalizer.frozen = True

        def make_heatmap_array(left=0., middle=0., right=0.):

            all_tests[:, :, 0] = xy
            all_tests[:, :, 1] = yx

            # all_tests[:, :, 2] = np.random.normal(0, 0.2, (precision, precision))
            # all_tests[:, :, 3] = np.random.normal(0, 0.2, (precision, precision))

            all_tests[:, :, 4-2] = left  # left
            all_tests[:, :, 5-2] = right  # right
            all_tests[:, :, 6-2] = middle  # middle

            r = np.reshape(all_tests, (precision ** 2, 5))

            # def redo_r(r):
            #     is_there = np.logical_and(r[:, 0]< 0.2, r[:, 0]>0.0)
            #     is_there_2 = np.logical_and(r[:, 1]< 0.05, r[:, 1]>0.)
            #
            #     return np.float32(np.logical_and(is_there, is_there_2))

            rewards = reward_function(r, normalize=False).numpy()
            value = np.zeros((precision, precision))

            # for i in range(precision):
            #     for j in range(precision):
            #         print(all_tests[i, j, :2])
            #         value[i, j] = reward_function(all_tests[i, j][np.newaxis]).numpy()

            for n, reward in enumerate(rewards):
                i = n % precision
                j = n // precision
                value[-j - 1, i] = reward

            return value

        def plot_heatmap(value, plot_idx='', vmin=0, vmax=1):
            fig, ax = plt.subplots()
            im = ax.imshow(value, interpolation="quadric", cmap="plasma", vmin=vmin, vmax=vmax)

            dx, dy = np.gradient(value)
            ax.contour(value, colors="k", linestyles="dashed", linewidths=0.7)
            ax.quiver(dy, -dx)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im)
            fig.savefig(name + plot_idx)

        to_test = [
            ((0, 0, 0), "_000"),
            ((0, 1, 0), "_010"),
            ((1, 0, 0), "_100"),
            ((0, 0, 1), "_001"),
            ((1, 1, 0), "_110"),
            ((0, 1, 1), "_011"),
            ((1, 0, 1), "_101"),
            ((1, 1, 1), "_111")
        ]

        v = []
        vmin = np.inf
        vmax = -np.inf
        for t, idx in to_test:
            vi = make_heatmap_array(*t)
            vmin = min(vmin, np.min(vi))
            vmax = max(vmax, np.max(vi))
            v.append(vi)

        print(vmin, vmax)

        for (t, idx), vi in zip(to_test, v):
            plot_heatmap(vi, idx, vmin, vmax)

    def airl_model_tests(
            self,
    ):
        # Import tf at the top of the file for this test
        s_shape = (10,)

        disc = Discriminator(
            s_shape
        )
        x = disc.get_weights()
        for xx in x:
            if not isinstance(xx, list):
                xx += np.random.normal(0, 3, xx.shape)
        disc.set_weights(x)
        disc.save_as_reward_function("checkpoints/estimated_reward_functions/AIRL/test")
        reward_func = RewardFunction(
            s_shape
        )
        reward_func.load_ckpt(
            "checkpoints/estimated_reward_functions/AIRL/test"
        )

        for _ in range(10):
            dummy_s = np.random.uniform(0, 1, (100,) + s_shape)
            r1, r2 = disc.compute_reward(dummy_s), reward_func(dummy_s)
            print(r1, r2)
            assert np.sum(np.abs(r1 - r2)) < 1e-3, "Reward function not restored properly !"


if __name__ == '__main__':
    exit(fire.Fire(Test))
