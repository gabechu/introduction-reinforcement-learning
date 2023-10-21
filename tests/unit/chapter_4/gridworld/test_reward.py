from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.reward import calculate_reward
from src.chapter_4.gridworld.state import State


def test_calculate_reward_for_valid_state():
    result = calculate_reward(State(x=2, y=2), action=Action.DOWN)
    expected_result = -1

    assert result == expected_result


def test_calculate_reward_for_terminal_state_one():
    result = calculate_reward(State(x=0, y=0), action=Action.DOWN)
    expected_result = 0

    assert result == expected_result


def test_calculate_reward_for_terminal_state_two():
    result = calculate_reward(State(x=3, y=3), action=Action.DOWN)
    expected_result = 0

    assert result == expected_result
