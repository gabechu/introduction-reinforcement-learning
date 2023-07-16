from src.chapter_3.gridworld.action import Action
from src.chapter_3.gridworld.state import State, get_next_state, is_state_valid


def evaluate_reward(current_state: State, action: Action) -> int:
    # special states
    if current_state == State(x=0, y=1):
        return 10
    elif current_state == State(x=0, y=3):
        return 5

    next_state = get_next_state(action=action, current_state=current_state)

    if is_state_valid(next_state) is False:
        return -1
    return 0
