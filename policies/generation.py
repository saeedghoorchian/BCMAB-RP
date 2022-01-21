from policies.bcmabrp import BCMABRP
from policies.cbrap import CBRAP
from policies.linear_ts import LinearTS
from policies.linucb import LinUCB


def policy_generation(bandit, actions, reduct_matrix):
    org_dim = 120
    red_dim = 60
    if bandit == 'BCMABRP':
        policy = BCMABRP(org_dim, red_dim, reduct_matrix, delta=0.5, R=0.01, lambd=0.5)
        policy.initialization()
    elif bandit == 'CBRAP':
        policy = CBRAP(org_dim, red_dim, reduct_matrix, alpha=0.5)
        policy.initialization()
    elif bandit == 'LinearTS':
        policy = LinearTS(org_dim, delta=0.5, R=0.01, epsilon=0.5)
        policy.initialization()
    elif bandit == 'LinUCB':
        policy = LinUCB(org_dim, alpha=0.5)
        policy.initialization()
    elif bandit == 'random':
        policy = 0

    return policy
