import numpy as np
from datetime import datetime

from sklearn.metrics import ndcg_score

from config.cofig import NDCG_SCORE_K


def evaluation_nonstationarity_function(trial, arm, num_of_arms):
    """Takes trial and arm index as input and returns index of arm with which to swap."""
    N_ARMS = num_of_arms
    SHIFT_SIZE = int(0.25 * N_ARMS)
    intervals = [0, 10000, 20000, 50000, 80000, 100000]
    for i, (start, end) in enumerate(zip(intervals, intervals[1:])):
        if start <= trial < end:
            return (arm + i * SHIFT_SIZE) % N_ARMS
    return arm


def tuning_nonstationarity_function(trial, arm, num_of_arms):
    """Takes trial and arm index as input and returns index of arm with which to swap."""
    N_ARMS = num_of_arms
    SHIFT_SIZE = int(0.25 * N_ARMS)
    intervals = [0, 10000, 20000, 30000]
    for i, (start, end) in enumerate(zip(intervals, intervals[1:])):
        if start <= trial < end:
            return (arm + i * SHIFT_SIZE) % N_ARMS
    return arm


def get_reward_and_ndcg_for_user(policy, trial, dataset, user_data, nonstationarity_function=None):
    context, reward_vector, score_true, missing_vector = user_data

    action_t, score_dict = policy.get_action(context, trial)
    action_ind = dataset.actions_index_by_action_id[action_t]

    if nonstationarity_function is not None:
        action_ind = nonstationarity_function(trial, action_ind, len(dataset.actions))
        score_reindex = [
            nonstationarity_function(trial, arm_ind, len(dataset.actions)) for arm_ind, _ in enumerate(dataset.actions)
        ]
        score_true[0, :] = score_true[0, score_reindex]

    if missing_vector is not None:
        # For Jester Dataset we need to skip the missing values
        if missing_vector[action_ind] == 1:
            # This means this user has not rated this item in the original data (the value is missing).
            return None, None

    reward = reward_vector[action_ind]

    score_predicted = np.zeros((1, len(dataset.actions)))
    for i, action_id in enumerate(dataset.actions):
        score_predicted[:, i] = score_dict[action_id]

    ndcg = ndcg_score(y_true=score_true, y_score=score_predicted, k=NDCG_SCORE_K[1])
    if ndcg < 0:
        raise ValueError()

    return reward, ndcg


def evaluate_policy(
        policy,
        times,
        dataset,
        tune=False,
        nonstationarity_function=None,
        introduce_nonstationarity=False,
):
    if tune:
        print("Using tuning dataset")
    else:
        print("Using evaluation dataset")

    if introduce_nonstationarity:
        print("Introducing non-stationarity")

    seq_reward = np.zeros(shape=(times, 1))
    seq_ndcg = np.zeros(shape=(times, 1))

    users_generator = dataset.generate_users(times, tune)

    if nonstationarity_function is None and introduce_nonstationarity:
        nonstationarity_function = tuning_nonstationarity_function if tune else evaluation_nonstationarity_function

    for t, user_at_t_data in enumerate(users_generator):

        reward_t, ndcg_t = get_reward_and_ndcg_for_user(
            policy, t, dataset, user_at_t_data, nonstationarity_function
        )

        policy.reward(reward_t)
        seq_reward[t] = reward_t
        seq_ndcg[t] = ndcg_t

        if t % 500 == 0:
            print(t)

    cumulative_reward = np.cumsum(seq_reward, axis=0)
    cumulative_ndcg = np.cumsum(seq_ndcg, axis=0)
    return cumulative_reward, cumulative_ndcg


def evaluate_policy_on_test_set(
        policy,
        times,
        dataset,
        tune=False,
):
    """Evaluate NDCG of policy at each step on a fully labeled separate test set(not used for learning).

    Reward is reported on the train set, as usual.
    Only NDCG needs to be computed on a subset of users who have all the ratings.
    """
    seq_reward = np.zeros(shape=(times, 1))
    seq_ndcg = np.zeros(shape=(times, 1))

    users_generator = dataset.generate_users_until_no_left(tune)

    t = 0
    user_ind = 0
    for user_at_t_data in users_generator:
        user_ind += 1
        if t == times:
            break

        reward_t, ndcg_t = get_reward_and_ndcg_for_user(
            policy, t, dataset, user_at_t_data
        )

        if reward_t is None and ndcg_t is None:
            # Skipped user due to missing value in original data.
            continue

        # Train policy with reward for train user.
        policy.reward(reward_t)

        # Evaluate policy on a separate (fixed for all t) set of users.
        test_users_data = dataset.test_users_data
        rewards_and_ndcgs = [
            get_reward_and_ndcg_for_user(policy, t, dataset, test_user_data)
            for test_user_data in test_users_data
        ]
        test_ndcgs = [x[1] for x in rewards_and_ndcgs]

        # Report train reward,
        seq_reward[t] = reward_t
        # but test ndcg.
        seq_ndcg[t] = np.mean(test_ndcgs)

        if t % 5000 == 0:
            print(t)
        t += 1

    print(f"Total {t} users considered in this experiment.")
    print(f"The algorithm has seen {user_ind} users, of them {user_ind - t} were skipped.")

    cumulative_reward = np.cumsum(seq_reward, axis=0)
    cumulative_ndcg = np.cumsum(seq_ndcg, axis=0)
    return cumulative_reward, cumulative_ndcg


def create_offline_dataset(times, actions, action_features, user_stream, user_features, reward_list):
    """Create a dataset for evaluation of offline algorithms (CTR prediction)"""
    action_context_dict = {}
    for action_id in actions:
        action_context = np.array(action_features[action_features['item_id'] == action_id])[0][1:]
        action_context_dict[action_id] = action_context.astype(np.float64)
        action_context_len = len(action_context)

    user_features = user_features.set_index("user_id")
    user_features_array = user_features.to_numpy(dtype=np.float32)

    # This is an optimization because pandas indexing is slower.
    user_id_to_index = {
        uid: ind for ind, uid in enumerate(user_features.index)
    }

    reward_list = reward_list.set_index("user_id")
    watched_list_series = reward_list.groupby('user_id')['item_id'].agg(set=set).set
    user_id_to_watched_list_index = {
        uid: ind for ind, uid in enumerate(watched_list_series.index)
    }

    contexts = np.zeros(shape=(times, len(actions), user_features.shape[1] + action_context_len))
    rewards = np.zeros(shape=(times, len(actions), 1))

    # Use last `times` users in the user stream. We do this because later data is more dense, so we get all
    # users from a smaller time window this way.
    assert times <= user_stream.shape[0], "Not enough users in user stream for given --times parameter"
    ind_start = user_stream.shape[0] - times
    ind_end = user_stream.shape[0] - 1
    if "timestamp" in user_stream.columns:
        print(f"First user in exp from {datetime.fromtimestamp(user_stream.timestamp[ind_start])}")
        print(f"Last user in exp from {datetime.fromtimestamp(user_stream.timestamp[ind_end])}")
    t = 0
    user_ind = ind_start
    while t < ind_end - ind_start:
        user_ind += 1
        user_t = user_stream.iloc[user_ind, 0]

        try:
            user_feature_index = user_id_to_index[user_t]
        except KeyError:
            # User with no features (see preprocessing code for details).
            continue
        user_feature = user_features_array[user_feature_index]

        # Create full context by concatenating user and item features.
        full_context_t = np.asarray(
            [
                np.append(action_context_dict[action_id], user_feature)
                for action_id in actions
            ]
        )  # (actions, full_context_dim)

        try:
            watched_list_index = user_id_to_watched_list_index[user_t]
            watched_list = watched_list_series.iloc[watched_list_index]
        except KeyError:
            watched_list = set()

        rewards_t = np.asmatrix(
            [
                1 if action_id in watched_list else 0
                for action_id in actions
            ]
        ).T

        # contexts[t] = full_context_t
        rewards[t] = rewards_t

        if t % 10000 == 0:
            print(t)

        t = t + 1
    # offline_contexts = np.concatenate(contexts, axis=0)
    # offline_rewards = np.concatenate(rewards, axis=0)
    offline_contexts = None
    offline_rewards = None

    # return offline_contexts, offline_rewards, contexts, rewards
    return offline_contexts, offline_rewards, contexts, rewards
