from collections import deque
import numpy as np
import scipy as sp
import six

from policies.reduction_matrix import get_reduction_matrix


class BCMAB_RP:
    def __init__(self, context_dimension, red_dim, a, gamma, lambda_param=1, seed=None):
        self.context_dimension = context_dimension
        self.red_dim = red_dim

        reduction_matrix = get_reduction_matrix(context_dimension, red_dim)
        assert reduction_matrix.shape == (context_dimension, red_dim)
        self.reduction_matrix = reduction_matrix
        self.lambda_param = lambda_param
        self.a = a
        self.gamma = gamma

        self.random_state = np.random.RandomState(seed)

        assert 0 < self.gamma < 1, "Parameter gamma must be in (0; 1)"
        assert self.a > 0, "Parameter a must be > 0"

        self.model_param_memory = deque(maxlen=1)
        self.history_memory = deque(maxlen=1)

        self.name = f"DLinTS (lambda={self.lambda_param}, gamma={self.gamma}, a={self.a})"

        # Initialization
        self.Z = self.lambda_param * np.identity(self.red_dim)
        self.Z_inv = np.linalg.inv(self.Z)
        self.Z_tilde = self.lambda_param * np.identity(self.red_dim)
        self.b = np.zeros((self.red_dim, 1))
        self.psi_hat = np.zeros((self.red_dim, 1))

        self.name = f"BCMAB-RP (d={self.red_dim}, gamma={self.gamma}, a={self.a})"

    def update_history(self, hst):  # (context, recommendation_id)
        self.history_memory.append(hst)

    def get_score(self, context, trial):
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])  # (actions, context_dim)
        context_array = context_array.dot(self.reduction_matrix)  # (actions, red_dim), z_{a,t} in paper

        Z_tilde_sqrt = sp.linalg.sqrtm(self.Z_tilde)

        with np.testing.suppress_warnings() as sup:
            sup.filter(np.ComplexWarning)
            Z_tilde_sqrt = Z_tilde_sqrt.astype(np.float64)

        W = self.random_state.multivariate_normal(
            np.zeros(self.red_dim), self.a ** 2 * np.identity(self.red_dim)
        )
        W = W.reshape((-1, 1))

        psi_tilde = self.psi_hat + self.Z_inv @ Z_tilde_sqrt @ W

        estimated_reward_array = context_array.dot(self.psi_hat)
        # self.rewards[trial, :] = estimated_reward_array.flatten()
        score_array = context_array.dot(psi_tilde)

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        for action_id, estimated_reward, score in zip(action_ids, estimated_reward_array, score_array):
            estimated_reward_dict[action_id] = float(estimated_reward)
            score_dict[action_id] = float(score)
            uncertainty_dict[action_id] = float(score - estimated_reward)
        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context, trial):
        estimated_reward, uncertainty, score = self.get_score(context, trial)
        recommendation_id = max(score, key=score.get)
        self.update_history((context, recommendation_id))
        return recommendation_id, score

    def reward(self, reward_t):
        context = self.history_memory[0][0]
        recommendation_id = self.history_memory[0][1]

        context_t = np.reshape(context[recommendation_id], (-1, 1))
        context_t = self.reduction_matrix.T.dot(context_t)

        self.Z = (
                self.gamma * self.Z
                + context_t.dot(context_t.T)
                + (1 - self.gamma) * self.lambda_param * np.identity(self.red_dim)
        )

        self.Z_inv = np.linalg.inv(self.Z)

        self.Z_tilde = (
            self.gamma ** 2 * self.Z_tilde
            + context_t.dot(context_t.T)
            + (1 - self.gamma ** 2) * self.lambda_param * np.identity(self.red_dim)
        )

        self.b = self.gamma * self.b + context_t * reward_t
        self.psi_hat = self.Z_inv @ self.b
