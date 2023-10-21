import numpy as np
from pytest import fixture

from src.chapter_4.gridworld.policy import Policy
from src.chapter_4.gridworld.run import evaluate_policy
from src.chapter_4.gridworld.state import State


@fixture
def state_value_matrix() -> np.ndarray:
    return np.array([[0.0, -14, -20, -22], [-14, -18, -20, -20], [-20, -20, -18, -14], [-22, -20, -14, -0.0]])


def test_evaluate_policy_for_terminal_state_00(state_value_matrix):
    result = evaluate_policy(
        State(x=0, y=0), state_value_matrix=state_value_matrix, discount_factor=1.0, policy=Policy.RANDOM_POLICY
    )
    expected_result = 0.0

    assert result == expected_result


def test_evaluate_policy_for_terminal_state_33(state_value_matrix):
    result = evaluate_policy(
        State(x=3, y=3), state_value_matrix=state_value_matrix, discount_factor=1.0, policy=Policy.RANDOM_POLICY
    )
    expected_result = 0.0

    assert result == expected_result


def test_evaluate_policy_for_regular_state_01(state_value_matrix):
    result = evaluate_policy(
        State(x=0, y=1), state_value_matrix=state_value_matrix, discount_factor=1.0, policy=Policy.RANDOM_POLICY
    )
    expected_result = -14

    assert result == expected_result