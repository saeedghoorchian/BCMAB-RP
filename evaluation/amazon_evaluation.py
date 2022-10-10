import numpy as np


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

