import numpy as np


def get_reduction_matrix(context_dimension, red_dim):
    rng = np.random.default_rng(seed=42)

    # Default value for kappa**2 = 1 / d. If this will work badly I will tune it.
    kappa = np.sqrt(1 / context_dimension)

    reduction_matrix = np.zeros((context_dimension, red_dim))
    for i in range(context_dimension):
        for j in range(red_dim):
            random_value = rng.normal(loc=0, scale=kappa)  # standard random matrix with N(0, 1)
            reduction_matrix[i, j] = random_value

    return reduction_matrix
