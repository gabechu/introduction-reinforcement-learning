from src.chapter_3.gridworld.reward import evaluate_next_state_reward
from src.chapter_3.gridworld.state import State


def test_evaluate_next_state_reward_for_normal_state():
    assert evaluate_next_state_reward(State(0, 0)) == 0
    assert evaluate_next_state_reward(State(2, 2)) == 0


def test_evaluate_next_state_reward_for_off_grid_state():
    assert evaluate_next_state_reward(State(-1, 0)) == -1
    assert evaluate_next_state_reward(State(5, 0)) == -1


def test_evaluate_next_state_reward_for_state_A():
    state_A = State(0, 1)
    assert evaluate_next_state_reward(state_A) == 10


def test_evaluate_next_state_reward_for_state_B():
    state_B = State(0, 3)
    assert evaluate_next_state_reward(state_B) == 5
