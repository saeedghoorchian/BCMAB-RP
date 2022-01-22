import numpy as np
import timeit
import json

from data_loading import get_r6b_pickle_data, get_movielens_data
from evaluation import evaluate_policy_on_r6b, evaluate_policy_on_movielens
from policies import policy_generation


def run_evaluation(trials, num_rep, reduct_matrix, config_file, dataset_type):
    print(f"Running each algorithm for {num_rep} repetitions")

    with open(config_file, "r") as f:
        experiment_config = json.load(f)

    if dataset_type == "r6b":
        data = get_r6b_pickle_data()
    elif dataset_type == "movielens":
        data = get_movielens_data()
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    times = trials
    # conduct analyses

    results = {}
    cum_reward = {}
    cum_ctr = {}
    time_all_dict = {}

    for experiment in experiment_config:
        bandit_name = experiment["algo"]

        params_list = experiment.get("params", [])
        if not params_list:
            params_list = [{}]
        for params in params_list:
            reward_all, ctr_all, final_rew_all, final_ctr_all, time_all = [], [], [], [], []
            for j in range(num_rep):
                timeBegin = timeit.default_timer()

                policy = policy_generation(bandit_name, reduct_matrix, params)

                print(f"{policy.name} repetition {j+1}")

                if dataset_type == "r6b":
                    seq_reward, seq_ctr = evaluate_policy_on_r6b(policy, bandit_name, data, times)
                elif dataset_type == "movielens":
                    # TODO Implement movielens evaluation
                    streaming_batch, user_feature, actions, reward_list, action_context = data
                    action_features = None
                    seq_reward = evaluate_policy_on_movielens(policy, bandit_name, streaming_batch, user_feature, reward_list,
                                                  actions, action_features, times)
                    seq_ctr = None

                timeEnd = timeit.default_timer()

                reward_all.append(seq_reward)
                ctr_all.append(seq_ctr)
                time_all.append(timeEnd - timeBegin)
                final_rew_all.append(seq_reward[times - 1])
                final_ctr_all.append(seq_ctr[times - 1])
                print(f"This took {timeEnd - timeBegin:.4f} seconds.\n")

            results[policy.name] = {"Time":  np.mean(time_all)}
            results[policy.name]["Total reward mean"] = np.mean(final_rew_all)
            results[policy.name]["Total reward std"] = np.std(final_rew_all)
            results[policy.name]["Total CTR mean"] = np.mean(final_ctr_all)
            results[policy.name]["Total CTR std"] = np.std(final_ctr_all)

            time_all_dict[policy.name] = time_all
            cum_reward[policy.name] = reward_all
            cum_ctr[policy.name] = ctr_all

    results = [(name, result_dict) for name, result_dict in results.items()]
    results = sorted(results, key=lambda x: x[1]["Total reward mean"], reverse=True)

    return results, cum_reward, cum_ctr, time_all_dict