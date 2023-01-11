import numpy as np
import timeit
import json

from data_loading import get_amazon_data, get_jester_data, get_movielens_data
from evaluation import evaluate_policy_on_amazon, evaluate_policy_on_jester, evaluate_policy_on_movielens
from evaluation import evaluate_policy
from policies import policy_generation


def run_evaluation(
        trials, num_rep, reduct_matrix, config_file, dataset_type, feature_flag, tune=False, non_stationarity=False
):
    print(f"Running each algorithm for {num_rep} repetitions")

    with open(config_file, "r") as f:
        experiment_config = json.load(f)

    if dataset_type == 'amazon':
        dataset = get_amazon_data(trials, tune)
    elif dataset_type == "jester":
        dataset = get_jester_data(trials, tune)
    elif dataset_type == "movielens":
        dataset = get_movielens_data(trials, tune)
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    times = trials
    # conduct analyses

    results = {}
    cum_reward = {}
    cum_ndcg = {}
    time_all_dict = {}
    policies = {}

    for experiment in experiment_config:
        bandit_name = experiment["algo"]

        params_list = experiment.get("params", [])
        if not params_list:
            params_list = [{}]
        for params in params_list:
            reward_all, ndcg_all, final_rew_all, time_all = [], [], [], []
            for j in range(num_rep):
                time_begin = timeit.default_timer()

                policy = policy_generation(bandit_name, reduct_matrix, params)

                print(f"{policy.name} repetition {j+1}")

                if dataset_type == "amazon":
                    seq_reward, seq_ndcg = evaluate_policy(policy, times, dataset, tune, introduce_nonstationarity=non_stationarity)
                elif dataset_type == "jester":
                    seq_reward, seq_ndcg = evaluate_policy(policy, times, dataset, tune, introduce_nonstationarity=non_stationarity)
                elif dataset_type == "movielens":
                    seq_reward, seq_ndcg = evaluate_policy(policy, times, dataset, tune, introduce_nonstationarity=non_stationarity)

                time_end = timeit.default_timer()

                reward_all.append(seq_reward)
                ndcg_all.append(seq_ndcg)
                time_all.append(time_end - time_begin)
                final_rew_all.append(seq_reward[times - 1])
                print(f"This took {time_end - time_begin:.4f} seconds.\n")

            policies[policy.name] = policy

            results[policy.name] = {
                "Time mean":  np.mean(time_all),
                "Time std": np.std(time_all),
                "Total reward mean": np.mean(final_rew_all),
                "Total reward std": np.std(final_rew_all),
            }

            time_all_dict[policy.name] = time_all
            cum_reward[policy.name] = reward_all
            cum_ndcg[policy.name] = ndcg_all

    results = [(name, result_dict) for name, result_dict in results.items()]
    results = sorted(results, key=lambda x: x[1]["Total reward mean"], reverse=True)

    return results, cum_reward, cum_ndcg, time_all_dict, policies