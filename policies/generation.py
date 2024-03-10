from policies.dlintsrp import D_LinTS_RP
from policies.cbrap import CBRAP
from policies.deepfm import DeepFM_OnlinePolicy
from policies.egreedy import EGreedy
from policies.linear_ts import LinearTS
from policies.d_lin_ts import DLinTS
from policies.linucb import LinUCB
from policies.random import RandomPolicy
from policies.ucb import UCB
from config import cofig


def policy_generation(bandit, reduct_matrix, params):
    org_dim, red_dim = reduct_matrix.shape
    if "intervals" in params:
        intervals = params["intervals"]
        assert type(intervals) == list
        cofig.NON_STATIONARITY_INTERVALS = intervals

    if "shift_size" in params:
        shift_size = params["shift_size"]
        assert 0.0 <= shift_size <= 1.0
        cofig.SHIFT_SIZE = shift_size

    if bandit == 'D_LinTS_RP':
        gamma = params.get("gamma", 0.999)
        a = params.get("a", 0.2)
        red_dim_param = params.get("red_dim", red_dim)
        seed = params.get("seed", None)
        policy = D_LinTS_RP(org_dim, red_dim_param,  a=a, gamma=gamma, seed=seed)
    elif bandit == 'CBRAP':
        alpha = params.get("alpha", 0.5)
        red_dim_param = params.get("red_dim", red_dim)
        policy = CBRAP(org_dim, red_dim_param, alpha=alpha)
    elif bandit == 'DeepFM':
        param_index = params.get("param_index", 0)
        policy = DeepFM_OnlinePolicy(org_dim, param_index)
    elif bandit == 'LinearTS':
        nu = params.get("nu", 0.5)
        policy = LinearTS(org_dim, delta=0.5, R=0.01, epsilon=0.5, nu=nu)
    elif bandit == 'DLinTS':
        gamma = params.get("gamma", 0.999)
        a = params.get("a", 0.2)
        policy = DLinTS(org_dim, gamma=gamma, a=a)
    elif bandit == 'LinUCB':
        alpha = params.get("alpha", 0.5)
        policy = LinUCB(org_dim, alpha=alpha)
    elif bandit == 'UCB':
        alpha = params.get("alpha", 0.5)
        policy = UCB(alpha)
    elif bandit == 'EGreedy':
        epsilon = params.get("epsilon", 0.2)
        policy = EGreedy(epsilon)
    elif bandit == 'RandomPolicy':
        policy = RandomPolicy()
    else:
        raise ValueError(f"Policy {bandit} is not supported")

    return policy
