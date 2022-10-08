import numpy as np
from config.cofig import PROJECT_DIR, AMAZON_CONTEXT_DIMENSION

JESTER_ORIGINAL_DIMENSION = 300
MOVIELENS_ORIGINAL_DIMENSION = 120


dataset_type_to_original_dimension = {
    "amazon": AMAZON_CONTEXT_DIMENSION,
    "jester": JESTER_ORIGINAL_DIMENSION,
    "movielens": MOVIELENS_ORIGINAL_DIMENSION,
}


def get_reduct_matrix(dataset_type, dimension, load_old_reduct_matrix):
    original_dimension = dataset_type_to_original_dimension[dataset_type]
    reduced_dimension = dimension
    if load_old_reduct_matrix:
        reduct_matrix = np.load(f"{PROJECT_DIR}/matrices/reduct_matrix_{dimension}.npy")
    else:
        # Alternative solution
        # reduct_matrix =  np.random.normal(loc=0, scale=1, size=(original_dimension, reduced_dimension))
        reduct_matrix = np.zeros((original_dimension, reduced_dimension))
        for i in range(original_dimension):
            for j in range(reduced_dimension):
                random_value = np.random.normal(loc=0, scale=1)  # standard random matrix with N(0, 1)
                reduct_matrix[i, j] = random_value
        np.save(f"{PROJECT_DIR}/matrices/reduct_matrix_{dimension}", reduct_matrix)

    return reduct_matrix
