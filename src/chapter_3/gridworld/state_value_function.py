import numpy as np

from src.chapter_3.gridworld.state import State


def get_state_value(state: State, state_value_matrix: np.ndarray) -> float:
    """Fetches the value of a state from a 2D state-value matrix."""
    return state_value_matrix[state.x][state.y]


def update_state_value_matrix(
    state: State,
    new_state_value: float,
    state_value_matrix: np.ndarray,
):
    """Updates state value matrix by mutating the pointed state with the new value."""
    state_value_matrix[state.x][state.y] = new_state_value
