from src.chapter_3.gridworld.state import State


def evaluate_next_state_reward(state: State) -> int:
    # special states
    if state == State(x=0, y=1):
        return 10
    elif state == State(x=0, y=3):
        return 5

    # invalid states
    if state.x < 0 or state.x > 4:
        return -1
    elif state.y < 0 or state.y > 4:
        return -1

    # normal states
    return 0
