from src.chapter_4.gridworld.state import _regularize_state, State


def test__regularize_state():
    actual = _regularize_state(new_state=State(0, 4), old_state=State(0, 3))
    assert actual == State(0, 3)
