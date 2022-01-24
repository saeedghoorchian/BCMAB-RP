from collections import deque
import six
import numpy as np


class EGreedy:

    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

        self.name = f"E-Greedy (epsilon={self.epsilon})"

        self.history_memory = deque(maxlen=1)

        self.q = None  # average reward for each arm
        self.n = None  # number of times each arm was chosen

    def initialization(self):
        pass

    def update_history(self, hst):  # recommendation_id
        self.history_memory.append(hst)

    def get_score(self, context, trial):
        action_ids = list(six.viewkeys(context))

        estimated_reward_dict = {}

        for action_id in action_ids:
            estimated_reward_dict[action_id] = float(self.q[action_id])

        return estimated_reward_dict

    def get_action(self, context, trial):
        # Initialize upon seeing the set of arms for the first time.
        action_ids = list(six.viewkeys(context))
        if self.q is None and self.n is None:
            self.q, self.n = {}, {}
            for action_id in action_ids:
                self.q[action_id] = 0
                self.n[action_id] = 1

        p = np.random.rand()
        if p > self.epsilon:
            # Choose arm with best estimated reward with probability (1-epsilon)
            estimated_reward = self.get_score(context, trial)
            recommendation_id = max(estimated_reward, key=estimated_reward.get)
        else:
            # Choose random arm with probability epsilon
            actions = list(context.keys())
            recommendation_id = actions[np.random.randint(0, len(actions) - 1)]

        self.update_history(recommendation_id)
        return recommendation_id

    def reward(self, reward_t):
        recommendation_id = self.history_memory[0]

        self.n[recommendation_id] += 1
        self.q[recommendation_id] += (reward_t - self.q[recommendation_id]) / self.n[
            recommendation_id
        ]
