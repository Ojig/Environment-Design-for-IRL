import json
from datetime import datetime

import numpy as np
from os import listdir
from os.path import isfile, join, isdir


def onlyfiles(path, endswith="", has=""):
    return [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(endswith) and has in f]


def onlydirs(path, endswith="", has=""):
    return [f for f in listdir(path) if isdir(join(path, f)) and f.endswith(endswith) and has in f]


def detect_env_set(path, env):
    all_runs = onlydirs(path, has=env)
    return set([int(d.split("_")[-1]) for d in all_runs])


def get_run_day_time(dirname):
    day_time_str = " ".join(dirname.split("_")[-2:])
    # 2023-02-14 23-09-48
    return datetime.strptime(day_time_str, '%Y-%m-%d %H-%M-%S')


def load_expert_results(expert_path, env_id, env):
    directory = onlydirs(expert_path, endswith="_"+str(env_id), has=env)[0]
    expert_path = join(expert_path, directory)
    # run folder
    directory = onlydirs(expert_path, has=env)[0]
    # sort by date

    expert_path = join(expert_path, directory, "result.json")

    for line in open(expert_path, 'r'):
        pass
    r = json.loads(line)

    hist = r["hist_stats"]["episode_reward"]
    print("expert", env_id, np.std(hist), np.mean(hist))

    return np.array([r["sampler_results"]["episode_reward_mean"]])


def load_trained_results(path, env_id, env):

    directory = onlydirs(path, endswith="_"+str(env_id), has=env)[0]
    trained_path = join(path, directory)
    # run folder
    # Oldest to newest
    run_directories = list(sorted(onlydirs(trained_path, has=env), key=get_run_day_time))
    rs = []
    for directory in run_directories:
        subpath = join(trained_path, directory, "result.json")
        for line in open(subpath, 'r'):
            pass
        r = json.loads(line)

        rs.append(r["sampler_results"]["episode_reward_mean"])

    return np.array(rs)


def load_random_results(path, env_id, env):

    directory = onlydirs(path, endswith="_"+str(env_id), has=env)[0]
    trained_path = join(path, directory)
    # run folder
    run_directories = onlydirs(trained_path, has=env)
    rs = []
    for directory in run_directories:
        subpath = join(trained_path, directory, "result.json")
        for line in open(subpath, 'r'):
            r = json.loads(line)
            if not np.isnan(r["sampler_results"]["episode_reward_mean"]):
                break
            pass

        hist = r["hist_stats"]["episode_reward"]
        print("random", env_id, np.std(hist), np.mean(hist))

        rs.append(r["sampler_results"]["episode_reward_mean"])

    return np.array(rs)
