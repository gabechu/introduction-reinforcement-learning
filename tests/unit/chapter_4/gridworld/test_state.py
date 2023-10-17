from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.state import State, get_next_state


def test_terminal_state_one():
    result = State(x=0, y=0)
    assert result.is_terminal_state is True


def test_terminal_state_two():
    result = State(x=3, y=3)
    assert result.is_terminal_state is True


def test_offgrid_state():
    result = State(x=3, y=4)
    assert result.is_off_grid is True


def test_regular_state():
    result = State(x=1, y=1)
    assert result.is_terminal_state is False
    assert result.is_off_grid is False


def test_get_next_state_for_terminal_state():
    result = get_next_state(State(x=0, y=0), Action.UP)
    expected_result = State(x=0, y=0)

    assert result == expected_result


def test_get_next_state_off_grid():
    result = get_next_state(State(x=3, y=2), Action.DOWN)
    expected_result = State(x=3, y=2)

    assert result == expected_result


def test_get_next_state_on_left_move_within_grid():
    result = get_next_state(State(x=1, y=1), Action.LEFT)
    expected_result = State(x=1, y=0)

    assert result == expected_result


def test_get_next_state_on_right_move_within_grid():
    result = get_next_state(State(x=1, y=1), Action.RIGHT)
    expected_result = State(x=1, y=2)

    assert result == expected_result


def test_get_next_state_on_up_move_within_grid():
    result = get_next_state(State(x=1, y=1), Action.UP)
    expected_result = State(x=0, y=1)

    assert result == expected_result


def test_get_next_state_on_down_move_within_grid():
    result = get_next_state(State(x=1, y=1), Action.DOWN)
    expected_result = State(x=2, y=1)

    assert result == expected_result
