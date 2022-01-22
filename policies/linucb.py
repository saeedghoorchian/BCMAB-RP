from collections import deque
import six
import numpy as np


class LinUCB:

    def __init__(self, context_dimension, alpha=0.5):
        super(LinUCB, self).__init__()
        self.context_dimension = context_dimension
        self.alpha = alpha

        self.model_param_memory = deque(maxlen=1)
        self.history_memory = deque(maxlen=1)

        self.name = f"LinUCB (alpha={self.alpha})"

    def update_history(self, hst):  # (context, recommendatin_id)
        self.history_memory.append(hst)

    def update_model_param(self, param):  # (A, b, inv_A)
        self.model_param_memory.append(param)

    def initialization(self):
        A = np.identity(self.context_dimension)
        b = np.zeros((self.context_dimension, 1))
        inv_A = np.linalg.inv(A)
        self.update_model_param((A, b, inv_A))

    def get_score(self, context):
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])

        A = self.model_param_memory[0][0]
        b = self.model_param_memory[0][1]
        inv_A = self.model_param_memory[0][2]
        A = A.astype(np.float32)
        theta = (inv_A).dot(b).astype(np.float32)
        theta = np.reshape(theta, (-1, 1))
        estimated_reward_array = context_array.dot(theta)

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}

        for action_id, estimated_reward in zip(action_ids, estimated_reward_array):
            action_context = np.reshape(context[action_id], (-1, 1))
            action_context = action_context.astype(np.float32)

            estimated_reward_dict[action_id] = float(estimated_reward)
            uncertainty_dict[action_id] = float(self.alpha * np.sqrt(action_context.T.dot(inv_A).dot(action_context)))
            score_dict[action_id] = (estimated_reward_dict[action_id] + uncertainty_dict[action_id])

        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context):
        # if not isinstance(context, dict):
        #     raise ValueError( "LinUCB requires context dict for all actions!")

        estimated_reward, uncertainty, score = self.get_score(context)
        recommendation_id = max(score, key=score.get)
        self.update_history((context, recommendation_id))
        return recommendation_id

    def reward(self, reward_t):
        context = self.history_memory[0][0]
        recommendation_id = self.history_memory[0][1]
        A = self.model_param_memory[0][0]
        b = self.model_param_memory[0][1]

        context_t = np.reshape(context[recommendation_id], (-1, 1))
        A = A + context_t.dot(context_t.T)
        A = A.astype(np.float32)

        b = b + reward_t * context_t

        inv_A = np.linalg.inv(A)

        self.update_model_param((A, b, inv_A))

