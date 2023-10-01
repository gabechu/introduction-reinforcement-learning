from dataclasses import dataclass

import numpy as np

from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.grid import GRID_SIZE


@dataclass
class State:
    x: int
    y: int

    def is_wall(self) -> bool:
        if self.x == 0 and self.y == 0:
            return True
        elif self.x == 3 and self.y == 3:
            return True

    def is_off_grid(self) -> bool:
        if 0 <= self.x <= (GRID_SIZE - 1) and 0 <= self.y <= (GRID_SIZE - 1):
            return True
        return False


def _regularize_state(new_state: State, old_state: State) -> State:
    """Fixes state when the agent is taken to a location off the grid or hit a wall."""
    match new_state:
        case new_state.is_wall:
            return old_state
        case new_state.is_off_grid:
            return old_state
        case _:
            return new_state


def _normal_state_transition(current_state: State, action: Action) -> State:
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
    # special state transitions -- shortcuts in the game
    if current_state.is_wall:
        return current_state

    # normal state transitions
    new_state = _normal_state_transition(current_state, action)
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
