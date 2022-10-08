import numpy as np


class RandomPolicy:
    def __init__(self):
        self.name = "RandomPolicy"

    def get_action(self, context, trial):
        score = {}
        for action_id in context.keys():
            score[action_id] = np.random.uniform(low=0, high=1)
        recommendation_id = max(score, key=score.get)
        return recommendation_id, score

    def reward(self, reward_t):
        pass
