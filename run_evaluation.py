import argparse
import numpy as np
import timeit

from evaluation import evaluate_policy_on_r6b, evaluate_policy_on_movielens
from policies import policy_generation
from reduct_matrix import get_reduct_matrix


def run_evaluation(trials, num_rep, reduct_matrix, dataset_type):
    print(num_rep)
    if dataset_type == "r6b":
        data = get_r6b_data()
    elif dataset_type == "movielens":
        data = get_movielens_data()
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    times = trials
    seq_rew = np.arange(times) + 1
    seq_rew = seq_rew.reshape(times, 1)
    # conduct analyses
    experiment_bandit = ['BCMABRP', 'CBRAP', 'LinearTS', 'LinUCB', 'random']
    results = {}
    cum_results = {}
    time_all_dict = {}

    i = 0
    for bandit in experiment_bandit:
        print(bandit)
        regret_all, reward_all, final_rew_all, time_all = [], [], [], []
        for j in range(num_rep):
            timeBegin = timeit.default_timer()

            if dataset_type == "r6b":
            
            elif dataset_type == "movielens":
                # TODO Implement movielens evaluation
                streaming_batch, user_feature, actions, reward_list, action_context = data
                action_features = None
                policy = policy_generation(bandit, actions, reduct_matrix)
                seq_error = evaluate_policy_on_movielens(policy, bandit, streaming_batch, user_feature, reward_list,
                                              actions, action_features, times)

            regret_all.append(seq_error)
            timeEnd = timeit.default_timer()

            time_all.append(timeEnd - timeBegin)

            final_rew_all.append(times - seq_error[times - 1])

        results[bandit] = [np.mean(time_all)]
        results[bandit].append(np.mean(final_rew_all))  # average of final regret for n rep
        results[bandit].append(np.var(final_rew_all))

        time_all_dict[bandit] = time_all
        cum_results[bandit] = regret_all

    print("Done!")
    return results, cum_results, time_all_dict


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

    run_evaluation(args.trials, args.num_rep, reduct_matrix, dataset_type="r6b")
