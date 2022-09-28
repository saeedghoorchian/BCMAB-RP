import numpy as np
from sklearn.metrics import ndcg_score

NDCG_SCORE_K = [3, 5, 7]


def evaluate_policy_on_amazon(policy, times, actions, action_features, user_stream, user_features, reward_list):
    print("Using OLD evaluation function")
    seq_reward = np.zeros(shape=(times, 1))

    action_context_dict = {}
    for action_id in actions:
        action_context_dict[action_id] = np.array(action_features[action_features['item_id'] == action_id])[0][1:]

    t = 0
    user_ind = 0
    while t < times:
        user_ind += 1
        user_t = user_stream.iloc[user_ind, 0]
         
        user_feature_query = user_features[user_features["user_id"] == user_t]
        if len(user_feature_query) == 0:
            # User with no features (see preprocessing code for details).
            continue
        user_feature = user_feature_query.drop(columns=["user_id"]).iloc[0].to_numpy()

        # Create full context by concatenating user and item features.
        full_context = {}
        for action_id in actions:
            full_context[action_id] = np.append(action_context_dict[action_id], user_feature)

        action_t = policy.get_action(full_context, t)

        watched_list = np.array(reward_list[reward_list["user_id"] == user_t]["item_id"])

        if action_t not in watched_list:
            policy.reward(0.0)
            if t == 0:
                seq_reward[t] = 0.0
            if t > 0:
                seq_reward[t] = seq_reward[t - 1]

        else:
            policy.reward(1.0)
            if t == 0:
                seq_reward[t] = 1.0
            else:
                seq_reward[t] = seq_reward[t - 1] + 1.0

        if t % 1000 == 0:
            print(t)

        t = t + 1

    return seq_reward


def evaluate_policy_on_amazon_new(
        policy,
        times,
        actions,
        action_features,
        action_biases,
        user_stream,
        user_features,
        user_biases,
        reward_list,
        ratings_list
):
    print("\nWARNING !!!\nFeature flag is ON, using NEW evaluation function\n")
    seq_reward = np.zeros(shape=(times, 1))
    seq_ndcg = np.zeros(shape=(times, 1))

    action_context_dict = {}
    action_bias_dict = {}
    for action_id in actions:
        action_context = np.array(action_features[action_features['item_id'] == action_id])[0][1:]
        action_bias = action_biases[action_biases["item_id"] == action_id].iloc[0][1]
        action_context_dict[action_id] = action_context.astype(np.float64)
        action_bias_dict[action_id] = action_bias

    user_features = user_features.set_index("user_id")
    user_features_array = user_features.to_numpy(dtype=np.float32)

    user_biases = user_biases.set_index("user_id")
    user_biases_array = user_biases.to_numpy(dtype=np.float32)

    assert (user_features.index == user_biases.index).all(), "Indexes of user features and biases dont coincide"
    # This is an optimization because pandas indexing is slower.
    user_id_to_index = {
        uid: ind for ind, uid in enumerate(user_features.index)
    }

    reward_list = reward_list.set_index("user_id")
    watched_list_series = reward_list.groupby('user_id')['item_id'].agg(set=set).set
    user_id_to_watched_list_index = {
        uid: ind for ind, uid in enumerate(watched_list_series.index)
    }

    user_item_to_rating = {}
    for (user_id, item_id, rating) in ratings_list.to_numpy():
        user_item_to_rating[(user_id, item_id)] = rating

    t = 0
    user_ind = 0
    while t < times:
        user_ind += 1
        user_t = user_stream.iloc[user_ind, 0]

        try:
            user_feature_index = user_id_to_index[user_t]
        except KeyError:
            # User with no features (see preprocessing code for details).
            continue
        user_feature = user_features_array[user_feature_index]
        user_bias = user_biases_array[user_feature_index]

        try:
            watched_list_index = user_id_to_watched_list_index[user_t]
            watched_list = watched_list_series.iloc[watched_list_index]
        except KeyError:
            watched_list = set()

        # Create full context by concatenating user and item features.
        full_context = {}
        # True ratings used to compute the NDCG score.
        score_true = np.zeros((1, len(actions)))
        for i, action_id in enumerate(actions):
            action_context = action_context_dict[action_id]
            action_bias = action_bias_dict[action_id]
            full_context[action_id] = np.append(action_context, user_feature)
            user_item_tuple = (user_t, action_id)
            # If item is rated by user - use true rating as score.
            if user_item_tuple in user_item_to_rating:
                rating = user_item_to_rating[user_item_tuple]
            else:
                # Otherwise use rating estimated by matrix factorization.
                # See https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
                rating = user_bias + action_bias + np.dot(action_context, user_feature)

            score_true[:, i] = rating

        action_t, score_dict = policy.get_action(full_context, t)

        if action_t not in watched_list:
            policy.reward(0.0)
            seq_reward[t] = 0.0
        else:
            policy.reward(1.0)
            seq_reward[t] = 1.0

        score_predicted = np.zeros((1, len(actions)))
        for i, action_id in enumerate(actions):
            score_predicted[:, i] = score_dict[action_id]

        seq_ndcg[t] = ndcg_score(y_true=score_true, y_score=score_predicted, k=NDCG_SCORE_K[1])

        if t % 1000 == 0:
            print(t)

        t = t + 1

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

    t = 0
    user_ind = 0
    while t < times:
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

        contexts[t] = full_context_t
        rewards[t] = rewards_t

        if t % 1000 == 0:
            print(t)

        t = t + 1
    contexts = np.concatenate(contexts, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    return contexts, rewards