import unittest

from src.chapter_2.ten_armed_bandits.action_reward_simulator import (
    NormalActionRewardSimulator,
    PoissonActionRewardSimulator,
)


class TestNormalActionRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_get_reward_action_zero(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(0)
        assert -3 < actual < 3

    def test_get_reward_last_action(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(4)
        assert -3 < actual < 3

    def test__get_optimal_reward(self):
        instance = NormalActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_reward()
        assert actual >= instance._true_action_rewards[0]
        assert actual >= instance._true_action_rewards[1]
        assert actual >= instance._true_action_rewards[2]

    def test__get_optimal_action(self):
        instance = NormalActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action()
        for i, value in enumerate(instance._true_action_rewards):
            if value == instance.get_optimal_reward():
                assert actual == i


class TestPoissonActionRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_get_reward_action_zero(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(0)
        assert actual > 0

    def test_get_reward_last_action(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(4)
        assert actual > 0

    def test__get_optimal_reward(self):
        instance = PoissonActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_reward()
        assert actual >= instance._true_action_rewards[0]
        assert actual >= instance._true_action_rewards[1]
        assert actual >= instance._true_action_rewards[2]

    def test__get_optimal_action(self):
        instance = PoissonActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action()
        optimal_actions = [
            action
            for action, value in enumerate(instance._true_action_rewards)
            if value == instance.get_optimal_reward()
        ]
        assert actual in optimal_actions
