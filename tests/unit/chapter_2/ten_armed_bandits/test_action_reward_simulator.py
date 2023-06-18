import unittest
import numpy as np

from src.chapter_2.ten_armed_bandits.action_reward_simulator import (
    NormalActionRewardSimulator,
    PoissonActionRewardSimulator,
    RandomWalkRewardSimulator,
)


class TestNormalActionRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_generate_reward_for_action_zero(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(0)
        assert -3 < actual < 3

    def test_generate_reward_for_last_action(self):
        instance = NormalActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(4)
        assert -3 < actual < 3

    def test_get_optimal_reward(self):
        instance = NormalActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_reward()
        assert actual >= instance._true_action_rewards[0]
        assert actual >= instance._true_action_rewards[1]
        assert actual >= instance._true_action_rewards[2]

    def test_get_optimal_action(self):
        instance = NormalActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action()
        for i, value in enumerate(instance._true_action_rewards):
            if value == instance.get_optimal_reward():
                assert actual == i


class TestRandomWalkRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = RandomWalkRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_initialised_equal_rewards(self):
        instance = RandomWalkRewardSimulator(num_actions=5)
        assert np.all(instance._true_action_rewards == 0.0)

    def test_generate_reward_for_action_zero(self):
        instance = RandomWalkRewardSimulator(num_actions=5)
        actual = instance.generate_reward(0)
        assert -3 < actual < 3

    def test_generate_reward_for_last_action(self):
        instance = RandomWalkRewardSimulator(num_actions=5)
        actual = instance.generate_reward(4)
        assert -3 < actual < 3

    def test_true_action_rewards_with_random_walk(self):
        instance = RandomWalkRewardSimulator(num_actions=5)
        instance.generate_reward(0)
        assert not np.all(instance._true_action_rewards == 0.0)

    def test_get_optimal_reward(self):
        instance = RandomWalkRewardSimulator(num_actions=3)
        actual = instance.get_optimal_reward()
        assert actual >= instance._true_action_rewards[0]
        assert actual >= instance._true_action_rewards[1]
        assert actual >= instance._true_action_rewards[2]

    def test_get_optimal_action(self):
        instance = RandomWalkRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action()
        optimal_actions = [
            action
            for action, value in enumerate(instance._true_action_rewards)
            if value == instance.get_optimal_reward()
        ]
        assert actual in optimal_actions


class TestPoissonActionRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_generate_reward_for_action_zero(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(0)
        assert actual > 0

    def test_generate_reward_for_last_action(self):
        instance = PoissonActionRewardSimulator(num_actions=5)
        actual = instance.generate_reward(4)
        assert actual > 0

    def test_get_optimal_reward(self):
        instance = PoissonActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_reward()
        assert actual >= instance._true_action_rewards[0]
        assert actual >= instance._true_action_rewards[1]
        assert actual >= instance._true_action_rewards[2]

    def test_get_optimal_action(self):
        instance = PoissonActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action()
        optimal_actions = [
            action
            for action, value in enumerate(instance._true_action_rewards)
            if value == instance.get_optimal_reward()
        ]
        assert actual in optimal_actions
