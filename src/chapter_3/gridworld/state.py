from dataclasses import dataclass

from src.chapter_3.gridworld.action import Action


@dataclass
class State:
    x: int
    y: int


def get_next_state(action: Action, current_state: State) -> State:
    """Get the location of the agent after taking the action, regardless of whether
    the agent goes off the grid or not."""
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


def regularize_state(state: State) -> State:
    """Fixes state when the agent is taken to a location off the grid."""
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
