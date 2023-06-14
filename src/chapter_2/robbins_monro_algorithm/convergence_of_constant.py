import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def cumsum_of_constant(steps: int, constant: int) -> float:
    return np.array([constant] * steps).cumsum()


def cumsum_of_square_of_constant(steps: int, constant) -> float:
    return np.array([constant**2 for _ in range(steps)]).cumsum()


def plot_array(array: npt.NDArray, label: str):
    plt.plot(array, label=label)
    plt.legend()


def run(steps: int, constant: float):
    first_condition = cumsum_of_constant(steps, constant)
    second_condition = cumsum_of_square_of_constant(steps, constant)

    plot_array(first_condition, "First Condition of Robbins Monro Algo")
    plot_array(second_condition, "Second Condition of Robbins Monro Algo")


if __name__ == "__main__":
    run(steps=100_000, constant=0.3)
    plt.show()
