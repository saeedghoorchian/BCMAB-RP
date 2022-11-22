from policies.bcmab_rp import BCMAB_RP
from policies.bcmabrp_old import BCMABRP_Old
from policies.cbrap import CBRAP
from policies.egreedy import EGreedy
from policies.linear_ts import LinearTS
from policies.d_lin_ts import DLinTS
from policies.linucb import LinUCB
from policies.random import RandomPolicy
from policies.ucb import UCB


def policy_generation(bandit, reduct_matrix, params):
    org_dim, red_dim = reduct_matrix.shape
    if bandit == 'BCMAB_RP':
        gamma = params.get("gamma", 0.999)
        a = params.get("a", 0.2)
        policy = BCMAB_RP(org_dim, red_dim, reduct_matrix, lambda_param=1, gamma=gamma, a=a)
    elif bandit == 'BCMABRP_Old':
        nu = params.get("nu", 0.5)
        policy = BCMABRP_Old(org_dim, red_dim, reduct_matrix, delta=0.5, R=0.01, lambd=0.5, nu=nu)
    elif bandit == 'CBRAP':
        alpha = params.get("alpha", 0.5)
        policy = CBRAP(org_dim, red_dim, alpha=alpha)
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

    return policy
