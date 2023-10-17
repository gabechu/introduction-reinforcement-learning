from dataclasses import dataclass

import numpy as np

from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.grid import GRID_SIZE


@dataclass
class State:
    x: int
    y: int

    @property
    def is_terminal_state(self) -> bool:
        return (self.x, self.y) in ((0, 0), (3, 3))

    @property
    def is_off_grid(self) -> bool:
        return not (0 <= self.x <= (GRID_SIZE - 1) and 0 <= self.y <= (GRID_SIZE - 1))


def _regularize_state(new_state: State, old_state: State) -> State:
    """Fixes state when the agent is taken to a location off the grid or hit a wall."""
    if new_state.is_terminal_state:
        return old_state
    elif new_state.is_off_grid:
        return old_state
    else:
        return new_state


def _state_transition(current_state: State, action: Action) -> State:
    match action:
        case action.UP:
            return State(x=current_state.x - 1, y=current_state.y)
        case action.DOWN:
            return State(x=current_state.x + 1, y=current_state.y)
        case action.RIGHT:
            return State(x=current_state.x, y=current_state.y + 1)
        case action.LEFT:
            return State(x=current_state.x, y=current_state.y - 1)


def get_next_state(current_state: State, action: Action) -> State:
    """Get the location of the agent after taking the action, regardless of whether
    the agent goes off the grid or not."""
    # terminal states
    if current_state.is_terminal_state:
        return current_state

    # state transitions
    new_state = _state_transition(current_state, action)
    return _regularize_state(new_state=new_state, old_state=current_state)


def build_states() -> list[State]:
    return [State(x=x, y=y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]


def build_state_value_matrix() -> np.ndarray:
    return np.zeros(shape=(GRID_SIZE, GRID_SIZE))


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
