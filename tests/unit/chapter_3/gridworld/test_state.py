from src.chapter_3.gridworld.action import Action
from src.chapter_3.gridworld.state import (
    State,
    get_next_state,
    is_state_valid,
    regularize_state,
)


def test_get_next_state_with_action_go_north_within_grid():
    actual = get_next_state(action=Action.NORTH, current_state=State(3, 3))
    assert actual == State(2, 3)


def test_get_next_state_with_action_go_south_within_grid():
    actual = get_next_state(action=Action.SOUTH, current_state=State(3, 3))
    assert actual == State(4, 3)


def test_get_next_state_with_action_go_east_within_grid():
    actual = get_next_state(action=Action.EAST, current_state=State(3, 3))
    assert actual == State(3, 4)


def test_get_next_state_with_action_go_west_within_grid():
    actual = get_next_state(action=Action.WEST, current_state=State(3, 3))
    assert actual == State(3, 2)


def test_get_next_state_with_action_go_north_off_grid():
    actual = get_next_state(action=Action.NORTH, current_state=State(0, 0))
    assert actual == State(-1, 0)


def test_get_next_state_with_action_go_south_off_grid():
    actual = get_next_state(action=Action.SOUTH, current_state=State(4, 4))
    assert actual == State(5, 4)


def test_get_next_state_with_special_state_A():
    state_A = State(0, 1)

    assert get_next_state(action=Action.NORTH, current_state=state_A) == State(4, 1)
    assert get_next_state(action=Action.SOUTH, current_state=state_A) == State(4, 1)
    assert get_next_state(action=Action.EAST, current_state=state_A) == State(4, 1)
    assert get_next_state(action=Action.WEST, current_state=state_A) == State(4, 1)


def test_get_next_state_with_special_state_B():
    state_B = State(0, 3)

    assert get_next_state(action=Action.NORTH, current_state=state_B) == State(2, 3)
    assert get_next_state(action=Action.SOUTH, current_state=state_B) == State(2, 3)
    assert get_next_state(action=Action.EAST, current_state=state_B) == State(2, 3)
    assert get_next_state(action=Action.WEST, current_state=state_B) == State(2, 3)


def test_regularize_state_for_normal_state():
    assert regularize_state(State(0, 0)) == State(0, 0)
    assert regularize_state(State(3, 3)) == State(3, 3)
    assert regularize_state(State(4, 4)) == State(4, 4)


def test_regularize_state_for_state_x_off_grid():
    assert regularize_state(State(-1, 3)) == State(0, 3)
    assert regularize_state(State(5, 3)) == State(4, 3)


def test_regularize_state_for_state_y_off_grid():
    assert regularize_state(State(3, -1)) == State(3, 0)
    assert regularize_state(State(3, 5)) == State(3, 4)


def test_is_state_valid_for_valid_state():
    assert is_state_valid(State(3, 3)) is True


def test_is_state_valid_for_invalid_state():
    assert is_state_valid(State(-1, 0)) is False
