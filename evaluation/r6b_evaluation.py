import numpy as np


def evaluate_policy_on_r6b(policy, bandit_name, data, trials):
    trial = 0

    total_reward = 0
    reward = []
    ctr = []

    events = data.events

    for event in events:

        # Keys of full context are action ids. In R6B dataset action ids are indices of articles in the pool.
        full_context = {}
        for action_id in event.pool_indices:
            full_context[action_id] = event.user_features

        # action_t is index of article relative to the pool
        action_t = policy.get_action(full_context)

        if action_t == event.displayed_pool_index:
            trial += 1
            policy.reward(event.user_click)

            total_reward += event.user_click
            reward.append(total_reward)
            ctr.append(total_reward / trial)

        else:
            # On r6b evaluation is on logged data. If the chosen article was not chosen in log data, this
            # datapoint is just skipped. See "Unbiased Offline Evaluation of Contextual-bandit-based News
            # Article Recommendation Algorithms" Li et. al.
            pass

        if trial >= trials:
            break

    return reward, ctr
