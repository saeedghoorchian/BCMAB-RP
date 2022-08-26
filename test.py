import math
import itertools


def powerset(s):
    len_s = len(s)
    p_set = []
    for length in range(len_s+1):
        for comb in itertools.combinations(s, length):
            p_set.append(list(comb))
    return p_set


def shapley_from_values(n, values):
    assert 2**n == len(values), "values should be length"
    N = list(range(n))
    S = powerset(N)
    print("Values")
    values_dict = {}
    for s_i, value in zip(S, values):
        print(s_i, value)
        values_dict[str(sorted(s_i))] = value

    shapley = []
    for i in range(n):
        shapley_i = 0
        for coalition in S:
            if i in coalition:
                continue
            s = len(coalition)
            factor = (1/(math.factorial(n))) * math.factorial(s) * math.factorial(n - s - 1)
            coalition_value = values_dict[str(sorted(coalition))]
            new_coalition = coalition + [i]
            new_coalition_value = values_dict[
                str(sorted(new_coalition))
            ]
            marginal_contribution = new_coalition_value - coalition_value
            shapley_i += factor * marginal_contribution

        shapley.append(shapley_i)

    print("Shapley values are:\n", shapley)


shapley_from_values(
    3,
    [0, 0, 0, 0, 90, 100, 120, 220],
)