from collections import deque
import six
import numpy as np


class UCB:

    def __init__(self, alpha=0.5):
        self.alpha = alpha

        self.name = f"UCB (alpha={self.alpha})"

        self.history_memory = deque(maxlen=1)

        self.q = None  # average reward for each arm
        self.n = None  # number of times each arm was chosen

    def initialization(self):
        pass

    def update_history(self, hst):  # recommendation_id
        self.history_memory.append(hst)

    def get_score(self, context, trial):
        action_ids = list(six.viewkeys(context))

        # Initialize upon seeing the set of arms for the first time.
        if self.q is None and self.n is None:
            self.q, self.n = {}, {}
            for action_id in action_ids:
                self.q[action_id] = 0
                self.n[action_id] = 1

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}

        for action_id in action_ids:
            estimated_reward_dict[action_id] = float(self.q[action_id])
            uncertainty_dict[action_id] = float(np.sqrt(self.alpha * np.log(trial + 1) / self.n[action_id]))
            score_dict[action_id] = (estimated_reward_dict[action_id] + uncertainty_dict[action_id])

        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context, trial):

        estimated_reward, uncertainty, score = self.get_score(context, trial)
        recommendation_id = max(score, key=score.get)
        self.update_history(recommendation_id)
        return recommendation_id

    def reward(self, reward_t):
        recommendation_id = self.history_memory[0]

        self.n[recommendation_id] += 1
        self.q[recommendation_id] += (reward_t - self.q[recommendation_id]) / self.n[
            recommendation_id
        ]
