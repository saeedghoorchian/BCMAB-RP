from policies.bcmabrp import BCMABRP
from policies.cbrap import CBRAP
from policies.linear_ts import LinearTS
from policies.linucb import LinUCB
from policies.random import RandomPolicy
from policies.ucb import UCB


def policy_generation(bandit, reduct_matrix, params):
    org_dim, red_dim = reduct_matrix.shape
    if bandit == 'BCMABRP':
        nu = params.get("nu", 0.5)
        policy = BCMABRP(org_dim, red_dim, reduct_matrix, delta=0.5, R=0.01, lambd=0.5, nu=nu)
    elif bandit == 'CBRAP':
        alpha = params.get("alpha", 0.5)
        policy = CBRAP(org_dim, red_dim, reduct_matrix, alpha=alpha)
    elif bandit == 'LinearTS':
        nu = params.get("nu", 0.5)
        policy = LinearTS(org_dim, delta=0.5, R=0.01, epsilon=0.5, nu=nu)
    elif bandit == 'LinUCB':
        alpha = params.get("alpha", 0.5)
        policy = LinUCB(org_dim, alpha=alpha)
    elif bandit == 'UCB':
        alpha = params.get("alpha", 0.5)
        policy = UCB(alpha)
    elif bandit == 'random':
        policy = RandomPolicy()

    policy.initialization()

    return policy
