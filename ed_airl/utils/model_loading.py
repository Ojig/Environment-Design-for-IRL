import os
from copy import copy

from ray.rllib.algorithms.registry import ALGORITHMS

from irl import custom_rewards


def get_folders(path):
    p = os.sep.join(path)
    if os.path.exists(p):
        return next(os.walk(p))[1]
    else:
        return []


def get_latest_from(path, f):
    return os.path.getctime(os.sep.join(path + [f]))


def get_ckpt_path(
        parent_folder
):
    path = copy(parent_folder)
    dirs = get_folders(path)
    if len(dirs) == 0:
        return os.sep.join(path)

    latest = max(dirs, key=lambda d: get_latest_from(path, d))
    path.append(latest)
    dirs = get_folders(path)
    if len(dirs) == 0:
        return os.sep.join(path)

    latest = max(dirs, key=lambda d: get_latest_from(path, d))
    path.append(latest)

    return os.sep.join(path)


def load_policy(
        path_or_object,
        used_algo,
        config,
        keep_algo_instance=False,
):
    if isinstance(path_or_object, list):
        ckpt_path = get_ckpt_path(path_or_object)
    else:
        ckpt_path = path_or_object

    try:
        algo_cls = ALGORITHMS[used_algo]()[0]
        algo = algo_cls(config)
        if ckpt_path is not None:
            print()
            print('Loading model at', ckpt_path, '...')
            print()
            try:
                algo.restore(ckpt_path)
            except ValueError as e:
                print()
                print('No models found at', ckpt_path)
                print('Using random policy')
                print()
    except AttributeError as e:
        print(e)
        policy_cls = algo.get_default_policy_class(config)
        algo.cleanup()
        algo = custom_rewards.wrap_algo_with_custom_rewards(
            algo_cls,
            policy_cls,
            {},
            None,
            compute_reward_distance=True
        )(config)

        if ckpt_path is not None:
            print()
            print('Loading model at', ckpt_path, '...')
            print()
            algo.restore(ckpt_path)
        else:
            print()
            print('No models found at', ckpt_path)
            print('Using random policy')
            print()

    policy = algo.get_policy()
    policy.config["filters"] = algo.workers.local_worker().filters

    if keep_algo_instance:
        return policy, algo
    else:
        algo.cleanup()
        return policy
