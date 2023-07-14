"""We're going to simulate and check if E(X+Y) equals E(X) + E(Y).
For a detailed mathmatical proof, check out this resource below:
https://math.stackexchange.com/questions/430727/what-does-expected-value-of-sum-of-two-discrete-random-variables-mean
"""
import numpy as np


def calculate_sample_average_for_two_indepedent_random_variables() -> float:
    size = 10_000
    x = np.random.normal(0, 1, size=size)
    y = np.random.normal(10, 6, size=size)

    # sample average
    x_expected_value = x.sum() / size
    y_expected_value = y.sum() / size

    expectation_sum = x_expected_value + y_expected_value
    sum_of_expectation = (x + y).sum() / size

    assert np.allclose(sum_of_expectation, expectation_sum)
    assert 9 < sum_of_expectation < 11

    return sum_of_expectation


def calculate_sample_average_for_two_depedent_random_variables() -> float:
    # Simulate two correlated random variables
    # https://oscarnieves100.medium.com/simulating-correlated-random-variables-in-python-c3947f2dbb10
    # X = mu_x + sigma_x * S1
    # Y = mu_y + sigma_y * (ro * S1 + sqrt(1 - ro ** 2) * S2)
    size = 10_000
    s1 = np.random.normal(0, 1, size=size)
    s2 = np.random.normal(2, 3, size=size)

    ro = 0.8
    x = 4 + 4 * s1
    y = 5 + 5 * (ro * s1 + np.sqrt(1 - ro**2) * s2)

    # sample average
    x_expected_value = x.sum() / size
    y_expected_value = y.sum() / size

    expectation_sum = x_expected_value + y_expected_value
    sum_of_expectation = (x + y).sum() / size

    assert np.allclose(sum_of_expectation, expectation_sum)

    return sum_of_expectation


if __name__ == "__main__":
    calculate_sample_average_for_two_indepedent_random_variables()
    calculate_sample_average_for_two_depedent_random_variables()
