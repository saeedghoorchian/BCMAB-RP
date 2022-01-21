import argparse
import numpy as np
import timeit
import json
import pickle

from data_loading import get_r6b_data, get_r6b_pickle_data, get_movielens_data
from evaluation import evaluate_policy_on_r6b, evaluate_policy_on_movielens
from policies import policy_generation
from reduct_matrix import get_reduct_matrix


def run_evaluation(trials, num_rep, reduct_matrix, dataset_type):
    print(f"Running each algorithm for {num_rep} repetitions")
    if dataset_type == "r6b":
        data = get_r6b_pickle_data()
    elif dataset_type == "movielens":
        data = get_movielens_data()
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    times = trials
    # conduct analyses
    experiment_bandit = ['BCMABRP', 'CBRAP', 'LinearTS', 'LinUCB', 'random']
    results = {}
    cum_reward = {}
    cum_ctr = {}
    time_all_dict = {}

    for bandit_name in experiment_bandit:
        reward_all, ctr_all, final_rew_all, final_ctr_all, time_all = [], [], [], [], []
        for j in range(num_rep):
            print(f"{bandit_name} repetition {j+1}")
            timeBegin = timeit.default_timer()

            policy = policy_generation(bandit_name, reduct_matrix)
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

        results[bandit_name]["Time"] = np.mean(time_all)
        results[bandit_name]["Total reward mean"] = np.mean(final_rew_all)
        results[bandit_name]["Total reward std"] = np.std(final_rew_all)
        results[bandit_name]["Total CTR mean"] = np.mean(final_ctr_all)
        results[bandit_name]["Total CTR std"] = np.std(final_ctr_all)

        time_all_dict[bandit_name] = time_all
        cum_reward[bandit_name] = reward_all
        cum_ctr[bandit_name] = ctr_all

    return results, cum_reward, cum_ctr, time_all_dict


def save_results(results, t, n, d):
    with open(f"results/results_t_{t}_n_{n}_d_{d}.pickle", "wb") as f:
        pickle.dump(results, f)

    with open(f"results/results_t_{t}_n_{n}_d_{d}.json", "w") as f:
        json.dump(results[0], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trials",
        metavar="TRIALS",
        type=int,
        help="Number of trials",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        help="Dimension of reduction matrix",
        required=True,
    )

    parser.add_argument(
        "--num-rep",
        type=int,
        help="Number of repetitions for each algorithms, default = 5",
        default=5,
    )

    parser.add_argument(
        "--load-old-reduct-matrix",
        dest="load_old_reduct_matrix",
        action="store_true",
        default=False,
        help="Load reduction matrix of needed dimension from file. "
             "Will throw error if no file exists."
    )

    args = parser.parse_args()

    reduct_matrix = get_reduct_matrix(args.dimension, args.load_old_reduct_matrix)

    timeBegin = timeit.default_timer()
    results = run_evaluation(args.trials, args.num_rep, reduct_matrix, dataset_type="r6b")

    print("Saving results")
    save_results(results, args.trials, args.num_rep, args.dimension)

    timeEnd = timeit.default_timer()
    print(f"Done.\nThe whole experiment took {timeEnd - timeBegin:.2f} seconds.")


