import numpy as np

from src.chapter_2.ten_armed_bandits.greedy_agent import choose_action_with_greedy_epsilon


def test_choose_action_with_epsilon_greedy():
    actual = choose_action_with_greedy_epsilon(estimated_action_rewards=np.arange(1000), epsilon=0.0)
    assert actual == 999


def test_choose_action_with_epsilon_random():
    actual = choose_action_with_greedy_epsilon(estimated_action_rewards=np.arange(1000), epsilon=0.999)
    assert actual != 999
