from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.state import State


def calculate_reward(current_state: State, action: Action) -> int:
    if current_state.is_terminal_state:
        return 0
    return -1
