import matplotlib.pyplot as plt
import numpy as np


def reciprocal_function(n: int) -> float:
    return 1 / n


def cumsum_of_reciprocal(steps: int) -> np.ndarray:
    return np.array([reciprocal_function(i + 1) for i in range(steps)]).cumsum()


def cumsum_of_square_of_reciprocal(steps: int) -> np.ndarray:
    return np.array([reciprocal_function(i + 1) ** 2 for i in range(steps)]).cumsum()


def plot_array(array: np.ndarray, label: str):
    plt.plot(array, label=label)
    plt.legend()


def run(steps: int):
    first_condition = cumsum_of_reciprocal(steps)
    second_condition = cumsum_of_square_of_reciprocal(steps)

    plot_array(first_condition, "First Condition of Robbins Monro Algo")
    plot_array(second_condition, "Second Condition of Robbins Monro Algo")


if __name__ == "__main__":
    run(steps=100_000)
    plt.show()
