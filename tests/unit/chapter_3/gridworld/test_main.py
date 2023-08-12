import numpy as np
from pytest import fixture

from src.chapter_3.gridworld.run import calculate_new_state_value
from src.chapter_3.gridworld.state import State


@fixture
def state_value_matrix() -> np.ndarray:
    return np.array(
        [
            [3.3, 8.8, 4.4, 5.3, 1.5],
            [1.5, 3.0, 2.3, 1.9, 0.5],
            [0.1, 0.7, 0.7, 0.4, -0.4],
            [-1.0, 0.4, -0.4, -0.6, -1.2],
            [-1.9, -1.3, -1.2, -1.4, -2.0],
        ]
    )


def test_calculate_new_state_value_for_state_A(state_value_matrix):
    actual = calculate_new_state_value(State(0, 1), state_value_matrix, 0.9)
    assert actual == 8.83


def test_calculate_new_state_value_for_state_B(state_value_matrix):
    actual = calculate_new_state_value(State(0, 3), state_value_matrix, 0.9)
    assert actual == 5.36


def test_calculate_new_state_value_for_state_00(state_value_matrix):
    actual = calculate_new_state_value(State(0, 0), state_value_matrix, 0.9)
    assert actual == 3.3025


def test_calculate_new_state_value_for_state_04(state_value_matrix):
    actual = calculate_new_state_value(State(0, 4), state_value_matrix, 0.9)
    assert actual == 1.48


def test_calculate_new_state_value_for_state_11(state_value_matrix):
    actual = calculate_new_state_value(State(1, 1), state_value_matrix, 0.9)
    assert actual == 2.9925


