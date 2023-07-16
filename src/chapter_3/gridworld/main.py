from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class State:
    x: int
    y: int


class Action(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


def get_next_state(action: Action, current_state: State) -> State:
    # special state transitions
    if current_state == State(x=0, y=1):
        return State(x=4, y=1)
    elif current_state == State(x=0, y=3):
        return State(x=2, y=3)

    # normal state transitions
    if action == action.NORTH:
        return State(x=current_state.x - 1, y=current_state.y)
    elif action == action.SOUTH:
        return State(x=current_state.x + 1, y=current_state.y)
    elif action == action.EAST:
        return State(x=current_state.x, y=current_state.y + 1)
    elif action == action.WEST:
        return State(x=current_state.x, y=current_state.y - 1)


def calculate_next_reward(current_state: State) -> int:
    # special states
    if current_state == State(x=0, y=1):
        return 10
    elif current_state == State(x=0, y=3):
        return 5

    # invalid states
    if current_state.x < 0 or current_state.x > 4:
        return -1
    elif current_state.y < 0 or current_state.y > 4:
        return -1

    # normal states
    return 0


def get_state_value(state: State, value_functions: np.ndarray) -> float:
    return value_functions[state.x][state.y]


def regularize_state(state: State) -> State:
    """Fix the state when agent is taking to a location off the grid."""
    if state.x < 0:
        return State(x=0, y=state.y)
    elif state.x > 4:
        return State(x=4, y=state.y)
    elif state.y < 0:
        return State(x=state.x, y=0)
    elif state.y > 4:
        return State(x=state.x, y=4)
    else:
        return state


def update_value_functions(
    state: State,
    new_state_value: float,
    value_functions: np.ndarray,
):
    value_functions[state.x][state.y] = new_state_value


def compute_new_state_value(
    state: State,
    value_functions: np.ndarray,
    gamma: float = 0.8,
) -> float:
    actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
    action_probability = 1 / len(actions)

    current_state_value = 0
    for action in actions:
        next_state = regularize_state(get_next_state(action, state))
        next_state_value = action_probability * (
            calculate_next_reward(state)
            + gamma * get_state_value(next_state, value_functions)
        )
        current_state_value += next_state_value

    return current_state_value


if __name__ == "__main__":
    # a 5 x 5 grid
    value_functions = np.zeros(shape=(5, 5))
    states = [State(x=x, y=y) for x in range(5) for y in range(5)]

    for i in range(5000):
        for state in states:
            new_state_value = compute_new_state_value(state, value_functions, 0.9)
            update_value_functions(
                state=state,
                new_state_value=new_state_value,
                value_functions=value_functions,
            )

    print(value_functions)
