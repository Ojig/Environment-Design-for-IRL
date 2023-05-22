from pprint import pprint

import matplotlib.pyplot as plt

import numpy as np
from fire import Fire

from utils.results_loading import detect_env_set, load_random_results, load_trained_results, load_expert_results

class Results:
    BASE_ENV_ID = 0
    NUM_DEMO = 20
    EXPERT = "expert"
    FROM_ESTIMATED = "from_estimated"
    RANDOM = "random"

    def load_for_env(self, expert_path, path_demo, path_test, env_id, env):
        if env_id >= Results.NUM_DEMO:
            path = path_test
        else:
            path = path_demo

        return {
            Results.RANDOM        : load_random_results(expert_path, env_id, env),
            Results.EXPERT        : load_expert_results(expert_path, env_id, env),
            Results.FROM_ESTIMATED: load_trained_results(path, env_id, env)
        }

    def compute_ratios(self, env, expert_path, trained_path_demo=None, trained_path_test=None,
                       name="Domain Randomization"):

        if trained_path_demo is not None:
            envs_tested = detect_env_set(trained_path_demo, env)
            if trained_path_test is not None:
                tests = detect_env_set(trained_path_test, env)
               # print(tests)
                envs_tested |= tests

        else:
            envs_tested = {i for i in range(30) if i != 19}  # all but uninformative env

        res_dict = {}
        for env_id in envs_tested:
            #print(env_id)
            res_dict[env_id] = self.load_for_env(expert_path, trained_path_demo, trained_path_test, env_id, env)

        demo_envs = [k for k in res_dict if (k < Results.NUM_DEMO)]
        test_envs = list(set(res_dict.keys()) - ({Results.BASE_ENV_ID} | set(demo_envs)))

        #print(demo_envs, test_envs, envs_tested)
        #pprint(res_dict)

        normalized_ratios = {
            "base_env"   : (res_dict[Results.BASE_ENV_ID][Results.FROM_ESTIMATED]
                            - np.mean(res_dict[Results.BASE_ENV_ID][Results.RANDOM]))
                           / (np.mean(res_dict[Results.BASE_ENV_ID][Results.EXPERT])
                              - np.mean(res_dict[Results.BASE_ENV_ID][Results.RANDOM]))
            ,

            "demo"       : [((res_dict[idx][Results.FROM_ESTIMATED]
                              - np.mean(res_dict[idx][Results.RANDOM]))
                             / (np.mean(res_dict[idx][Results.EXPERT])
                                - np.mean(res_dict[idx][Results.RANDOM])))[np.newaxis] for idx in demo_envs],

            "random_demo": [[np.mean(res_dict[idx][Results.RANDOM])] for idx in demo_envs],
        }

        if 20 in envs_tested:
            normalized_ratios["random_test"] = [[np.mean(res_dict[idx][Results.RANDOM])] for idx in test_envs]
            normalized_ratios["test"] = [((res_dict[idx][Results.FROM_ESTIMATED]
                                           - np.mean(res_dict[idx][Results.RANDOM]))
                                          / (np.mean(res_dict[idx][Results.EXPERT])
                                             - np.mean(res_dict[idx][Results.RANDOM])))[np.newaxis] for idx in
                                         test_envs]

        for k, v in normalized_ratios.items():
            if isinstance(v, list):
                normalized_ratios[k] = np.concatenate(v, axis=0)

        normalized_ratios["base_mean"] = np.mean(normalized_ratios["base_env"])
        normalized_ratios["bas_std"] = np.std(normalized_ratios["base_env"])
        run_types = ["demo"]
        if 20 in envs_tested:
            run_types += ["test"]
        for run in run_types:
            normalized_ratios[f"{run}_mean"] = np.nanmean(normalized_ratios[run])
            normalized_ratios[f"{run}_max"] = np.mean(np.nanmax(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_min"] = np.mean(np.nanmin(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_top25"] = np.mean(np.quantile(normalized_ratios[run], 0.75, axis=0))
            normalized_ratios[f"{run}_bottom25"] = np.mean(np.quantile(normalized_ratios[run], 0.25, axis=0))
            normalized_ratios[f"{run}_min"] = np.mean(np.nanmin(normalized_ratios[run], axis=0))

            normalized_ratios[f"{run}_mean_std"] = np.std(np.nanmean(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_max_std"] = np.std(np.nanmax(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_min_std"] = np.std(np.nanmin(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_top25_std"] = np.std(np.quantile(normalized_ratios[run], 0.75, axis=0))
            normalized_ratios[f"{run}_bottom25_std"] = np.std(np.quantile(normalized_ratios[run], 0.25, axis=0))

            normalized_ratios[f"{run}_mean_min"] = np.min(np.nanmean(normalized_ratios[run], axis=0))
            normalized_ratios[f"{run}_mean_max"] = np.max(np.nanmean(normalized_ratios[run], axis=0))

            normalized_ratios[f"{run}_argmax"] = np.argmax(normalized_ratios[run], axis=0) + 1
            normalized_ratios[f"{run}_argmin"] = np.argmin(normalized_ratios[run], axis=0) + 1

            # normalized_ratios[f"per_env_{run}"] = np.mean(normalized_ratios[run], axis=1)

        pprint(normalized_ratios)

        scores = np.concatenate([normalized_ratios["base_env"][np.newaxis], normalized_ratios["demo"]], axis=0)
        labels = ["Base"] + [str(i) for i in range(1, len(scores))]
        x = np.arange(len(labels)) + 1
        plt.boxplot(scores.T, labels=labels, sym="", whis=[0, 100])
        plt.title(name + " (demo set)")
        plt.ylim([-1.6, 1.2])
        plt.ylabel("Normalized score")
        plt.xlabel("Half Cheetah demo environments")
        plt.draw()
        plt.savefig(f'boxplot_demo_{name}.pdf')
        plt.clf()

        scores_test = normalized_ratios["test"]
        labels = [str(i) for i in range(1, len(scores_test) + 1)]
        x = np.arange(len(labels)) + 1

        plt.boxplot(scores_test.T, labels=labels, sym="", whis=[0, 100])
        plt.title(name + " (test set)")
        plt.ylim([-1.6, 1.2])
        plt.ylabel("Normalized score")
        plt.xlabel("Half Cheetah test environments")
        plt.draw()
        plt.show()
        plt.savefig(f'boxplot_test_{name}.pdf')

        plt.clf()


if __name__ == '__main__':
    Fire(Results)
