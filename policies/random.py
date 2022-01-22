import numpy as np


class RandomPolicy:
    def __init__(self):
        self.name = "RandomPolicy"

    def initialization(self):
        pass

    def get_action(self, context):
        actions = list(context.keys())
        return actions[np.random.randint(0, len(actions) - 1)]

    def reward(self, reward_t):
        pass
