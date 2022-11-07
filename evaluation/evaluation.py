import numpy as np
from datetime import datetime

from sklearn.metrics import ndcg_score

from config.cofig import NDCG_SCORE_K


def get_reward_and_ndcg_for_user(policy, trial, actions, user_data):
    context, watched_list, score_true = user_data

    action_t, score_dict = policy.get_action(context, trial)

    reward = 1 if action_t in watched_list else 0

    score_predicted = np.zeros((1, len(actions)))
    for i, action_id in enumerate(actions):
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
):
    if tune:
        print("Using tuning dataset")
    else:
        print("Using evaluation dataset")
    seq_reward = np.zeros(shape=(times, 1))
    seq_ndcg = np.zeros(shape=(times, 1))

    users_generator = dataset.generate_users(times, tune)

    for t, user_at_t_data in enumerate(users_generator):

        reward_t, ndcg_t = get_reward_and_ndcg_for_user(
            policy, t, dataset.actions, user_at_t_data
        )

        policy.reward(reward_t)
        seq_reward[t] = reward_t
        seq_ndcg[t] = ndcg_t

        if t % 5000 == 0:
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
    """Evaluate policy at each step on a fully labeled separate test set(not used for learning)."""
    seq_reward = np.zeros(shape=(times, 1))
    seq_ndcg = np.zeros(shape=(times, 1))

    users_generator = dataset.generate_users(times, tune)

    for t, user_at_t_data in enumerate(users_generator):

        reward_t, ndcg_t = get_reward_and_ndcg_for_user(
            policy, t, dataset.actions, user_at_t_data
        )

        # Train policy with reward for train user.
        policy.reward(reward_t)

        # Evaluate policy on a separate (fixed for all t) set of users.
        test_users_data = dataset.test_users_data
        rewards_and_ndcgs = [
            get_reward_and_ndcg_for_user(policy, t, dataset.actions, test_user_data)
            for test_user_data in test_users_data
        ]
        test_rewards = [x[0] for x in rewards_and_ndcgs]
        test_ndcgs = [x[1] for x in rewards_and_ndcgs]

        seq_reward[t] = np.mean(test_rewards)
        seq_ndcg[t] = np.mean(test_ndcgs)

        if t % 5000 == 0:
            print(t)

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
