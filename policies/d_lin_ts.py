from collections import deque
import six
import numpy as np
import scipy as sp


class DLinTS:
    """Discounted Linear Thompson Sampling

    From paper:
    Randomized Exploration for Non-Stationary Stochastic Linear Bandits, Kim et al.
    http://proceedings.mlr.press/v124/kim20a/kim20a-supp.pdf
    """

    def __init__(self, context_dimension, lambda_param=1, gamma=0.5, a=None, seed=None):
        super(DLinTS, self).__init__()
        self.context_dimension = context_dimension
        self.random_state = np.random.RandomState(seed)

        self.lambda_param = lambda_param
        self.gamma = gamma
        if a is None:
            print("\n\nSetting a manually and tuning it works better\n\n")
            # Default setting of parameter as in Corollary 9 of the paper.
            T = 100000
            c_1 = np.sqrt(
                2 * np.log(T) + context_dimension * np.log(
                    1 + (
                            (1 - np.power(gamma, 2 * (T - 1)))
                            / (lambda_param * context_dimension * (1 - gamma**2))
                    )
                )
            ) + np.sqrt(lambda_param)

            self.a = np.sqrt(14) * c_1
        else:
            self.a = a

        assert self.lambda_param >= 1, "Parameter lambda_param must be >= 1"
        assert 0 < self.gamma < 1, "Parameter gamma must be in (0; 1)"
        assert self.a > 0, "Parameter a must be > 0"

        self.model_param_memory = deque(maxlen=1)
        self.history_memory = deque(maxlen=1)

        self.name = f"DLinTS (lambda={self.lambda_param}, gamma={self.gamma}, a={self.a})"

        # Initialization
        self.W = self.lambda_param * np.identity(self.context_dimension)
        self.W_inv = np.linalg.inv(self.W)
        self.W_tilde = self.lambda_param * np.identity(self.context_dimension)
        self.b_hat = np.zeros((self.context_dimension, 1))
        self.theta_hat = np.zeros((self.context_dimension, 1))

        self.rewards = np.zeros((10000, 1000))

    def update_history(self, hst):  # (context, recommendatin_id)
        self.history_memory.append(hst)

    def update_model_param(self, param):  # (B, f, inv_B)
        self.model_param_memory.append(param)

    def get_score(self, context, trial):
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])

        W_tilde_sqrt = sp.linalg.sqrtm(self.W_tilde)

        with np.testing.suppress_warnings() as sup:
            sup.filter(np.ComplexWarning)
            W_tilde_sqrt = W_tilde_sqrt.astype(np.float64)

        Z = self.random_state.multivariate_normal(
            np.zeros(self.context_dimension), self.a ** 2 * np.identity(self.context_dimension)
        )
        Z = Z.reshape((-1, 1))

        theta_tilde = self.theta_hat + self.W_inv @ W_tilde_sqrt @ Z

        estimated_reward_array = context_array.dot(self.theta_hat)
        # self.rewards[trial, :] = estimated_reward_array.flatten()
        score_array = context_array.dot(theta_tilde)

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

        self.W = (
                self.gamma * self.W
                + context_t.dot(context_t.T)
                + (1 - self.gamma) * self.lambda_param * np.identity(self.context_dimension)
        )

        self.W_inv = np.linalg.inv(self.W)

        self.W_tilde = (
            self.gamma ** 2 * self.W_tilde
            + context_t.dot(context_t.T)
            + (1 - self.gamma ** 2) * self.lambda_param * np.identity(self.context_dimension)
        )

        self.b_hat = self.gamma * self.b_hat + context_t * reward_t
        self.theta_hat = self.W_inv @ self.b_hat
