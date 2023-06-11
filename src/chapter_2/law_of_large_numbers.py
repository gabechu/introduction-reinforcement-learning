"""Strong law of large numbers vs weak law of large numbers"""
from typing import List

import numpy as np


def sample_normal_growing_sizes(sizes: List[int], mean: float, variance: float) -> List[float]:
    """Sample from a normal distribution with growing sample sizes."""
    return [np.random.normal(mean, variance, size).mean() for size in sizes]


def sample_poisson_growing_sizes(sizes: List[int], lambda_: float) -> List[float]:
    """Sample from a possion distribution with growing sample sizes."""
    return [np.random.poisson(lambda_, size).mean() for size in sizes]


def repeated_sampling_normal_fixed_size(repetitions: List[int], size: int, mean: float, variance: float) -> List[float]:
    """Repeatedly sample from a fixed-sized normal distribution."""
    repeated_experiments = [[np.random.normal(mean, variance, size) for _ in range(rep)] for rep in repetitions]
    mean_repeated_experiments = [np.array(experiment).mean() for experiment in repeated_experiments]
    return mean_repeated_experiments


def repeated_sampling_poisson_fixed_size(repetitions: List[int], size: int, lambda_: float) -> List[float]:
    """Repeatedly sample from a fixed-sized possion distribution."""
    repeated_experiments = [[np.random.poisson(lambda_, size) for _ in range(rep)] for rep in repetitions]
    mean_repeated_experiments = [np.array(experiment).mean() for experiment in repeated_experiments]
    return mean_repeated_experiments


def main():
    """Main function."""
    sample_sizes = [10, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000]
    print(sample_normal_growing_sizes(sample_sizes, mean=3, variance=10))
    print(sample_poisson_growing_sizes(sample_sizes, lambda_=5))

    reps = [10, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000]
    print(repeated_sampling_normal_fixed_size(repetitions=reps, size=10, mean=3, variance=10))
    print(repeated_sampling_poisson_fixed_size(repetitions=reps, size=10, lambda_=5))


if __name__ == "__main__":
    main()
