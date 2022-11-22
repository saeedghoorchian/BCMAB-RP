from collections import deque
import six
import numpy as np

from policies.reduction_matrix import get_reduction_matrix


class CBRAP:

    def __init__(self, context_dimension, red_dim, alpha, scale=False):
        super(CBRAP, self).__init__()
        self.context_dimension = context_dimension
        self.red_dim = red_dim
        self.alpha = alpha

        self.model_param_memory = deque(maxlen=1)
        self.history_memory = deque(maxlen=1)

        reduct_matrix = get_reduction_matrix(context_dimension, red_dim)
        if scale:
            self.reduction_matrix = np.zeros(reduct_matrix.shape)
            for k in range(red_dim):
                self.reduction_matrix[:, k] = reduct_matrix[:, k] / np.linalg.norm(reduct_matrix[:, k])
        else:
            self.reduction_matrix = reduct_matrix

        self.name = f"CBRAP (alpha={self.alpha})"

        # Initialization
        A = np.identity(self.red_dim)
        # theta = np.zeros((self.red_dim, 1))
        b = np.zeros((self.red_dim, 1))
        inv_A = np.linalg.inv(A)  # .astype(np.float32)
        # print(self.model_param_memory)
        self.update_model_param((A, b, inv_A))
        # print(self.model_param_memory)

    def update_history(self, hst):  # (context, recommendatin_id)
        self.history_memory.append(hst)
        # return self.history_memory

    def update_model_param(self, param):  # (A, b, inv_A)
        self.model_param_memory.append(param)

    def get_score(self, context, trial):
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])
        context_array = context_array.dot(self.reduction_matrix)

        A = self.model_param_memory[0][0]
        b = self.model_param_memory[0][1]
        inv_A = self.model_param_memory[0][2]
        # mu_hat = np.matmul(np.linalg.inv(B), f)
        A = A.astype(np.float32)
        theta = (inv_A).dot(b).astype(np.float32)  # (np.linalg.inv(A)).dot(b).astype(np.float32)
        theta = np.reshape(theta, (-1, 1))
        estimated_reward_array = context_array.dot(theta)
        # score_array = context_array.dot()

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        # for action_id in action_ids:
        for action_id, estimated_reward, action_context in zip(action_ids, estimated_reward_array, context_array):
            # action_context = np.reshape(context[action_id], (-1, 1))
            # action_context = context[action_id].dot(self.reduction_matrix)
            action_context = np.reshape(action_context, (-1, 1))
            action_context = action_context.astype(np.float32)

            # estimated_reward[action_id] = float(theta[action_id].T.dot(action_context))
            estimated_reward_dict[action_id] = float(estimated_reward)
            # uncertainty_dict[action_id] = float(self.alpha * np.sqrt(action_context.T.dot(np.linalg.inv(A)).dot(action_context)))
            uncertainty_dict[action_id] = float(self.alpha * np.sqrt(action_context.T.dot(inv_A).dot(action_context)))
            score_dict[action_id] = (estimated_reward_dict[action_id] + uncertainty_dict[action_id])

        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context, trial):
        # if not isinstance(context, dict):
        #     raise ValueError( "LinUCB requires context dict for all actions!")

        estimated_reward, uncertainty, score = self.get_score(context, trial)
        recommendation_id = max(score, key=score.get)
        self.update_history((context, recommendation_id))
        return recommendation_id, score

    # def reward(self, history_m, rewards):
    def reward(self, reward_t):
        # context = history_m[0][0]
        # recommendation_id = history_m[0][1]
        context = self.history_memory[0][0]
        recommendation_id = self.history_memory[0][1]
        A = self.model_param_memory[0][0]
        b = self.model_param_memory[0][1]
        # inv_A = self.model_param_memory[0][2]

        # for action_id, reward in six.viewitems(rewards):
        # context_t = np.reshape(context[recommendation_id], (-1, 1))
        context_t = context[recommendation_id].dot(self.reduction_matrix)
        context_t = np.reshape(context_t, (-1, 1))
        A = A + context_t.dot(context_t.T)
        A = A.astype(np.float32)

        b = b + reward_t * context_t
        # theta = np.linalg.inv(B).dot(f)
        inv_A = np.linalg.inv(A)  # .astype(np.float32)
        # print(self.model_param_memory)
        self.update_model_param((A, b, inv_A))
        # print(self.model_param_memory)
        # print("#############################################")
