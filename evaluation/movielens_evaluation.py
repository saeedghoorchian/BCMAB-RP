import numpy as np


def evaluate_policy_on_movielens(policy, times, streaming_batch, user_feature, actions, action_features, reward_list):
    seq_reward = np.zeros(shape=(times, 1))

    action_context_dict = {}
    for action_id in actions:  # movie_id :)
        action_context_dict[action_id] = np.array(action_features[action_features['movieid'] == action_id])[0][1:]
    if policy.name != 'random':

        j = 0  #
        t = 0  #
        while t < times:  #
            j = j + 1  #
            if len(np.array(user_feature[user_feature.index == streaming_batch.iloc[j, 0]])) == 0:  #
                # print("time with no user:" + str(j))
                continue
            feature = np.array(user_feature[user_feature.index == streaming_batch.iloc[j, 0]])[0]  #
            full_context = {}
            for action_id in actions:  # movie_id :)
                full_context[action_id] = np.append(action_context_dict[action_id], feature)  # feature
            # history_m, action_t = policy.get_action(full_context, 1)
            action_t = policy.get_action(full_context, t)
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[j, 0]]  #
            if action_t not in list(watched_list['movie_id']):
                policy.reward(0.0)
                if t == 0:
                    seq_reward[t] = 0.0
                else:
                    seq_reward[t] = seq_reward[t - 1]
            else:
                policy.reward(1.0)
                if t == 0:
                    seq_reward[t] = 1.0
                else:
                    seq_reward[t] = seq_reward[t - 1] + 1.0

            t = t + 1  #

            if t % 1000 == 0:
                print(t)

        print("jj = " + str(j))
    elif policy.name == 'random':

        j = 0
        t = 0
        while t < times:
            j = j + 1
            if len(np.array(user_feature[user_feature.index == streaming_batch.iloc[j, 0]])) == 0:
                print("time in random with no user:" + str(j))
                continue
            action_t = actions[np.random.randint(0, len(actions) - 1)]
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[j, 0]]
            if action_t not in list(watched_list['movie_id']):
                if t == 0:
                    seq_reward[t] = 1.0
                else:
                    seq_reward[t] = seq_reward[t - 1] + 1.0
            else:
                if t > 0:
                    seq_reward[t] = seq_reward[t - 1]
            t = t + 1
    return seq_reward
