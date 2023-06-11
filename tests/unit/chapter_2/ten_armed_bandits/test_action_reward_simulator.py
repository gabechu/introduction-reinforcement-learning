import unittest

from src.chapter_2.ten_armed_bandits.action_reward_simulator import ActionRewardSimulator


class TestActionRewardSimulator(unittest.TestCase):
    def test_initialised_action_size(self):
        instance = ActionRewardSimulator(num_actions=5)
        actual = len(instance._true_action_rewards)
        assert actual == 5

    def test_get_reward_action_zero(self):
        instance = ActionRewardSimulator(num_actions=5)
        actual = instance.get_reward(0)
        assert -3 < actual < 3

    def test_get_reward_last_action(self):
        instance = ActionRewardSimulator(num_actions=5)
        actual = instance.get_reward(4)
        assert -3 < actual < 3

    def test__get_optimal_action_and_reward_verify_key_values(self):
        instance = ActionRewardSimulator(num_actions=5)
        actual = instance.get_optimal_action_and_reward()
        assert list(actual.keys()) == ["action", "reward_value"]

    def test__get_optimal_action_and_reward_verify_optimal_value(self):
        instance = ActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action_and_reward()
        assert actual["reward_value"] >= instance._true_action_rewards[0]
        assert actual["reward_value"] >= instance._true_action_rewards[1]
        assert actual["reward_value"] >= instance._true_action_rewards[2]

    def test__get_optimal_action_and_reward_verify_optimal_action(self):
        instance = ActionRewardSimulator(num_actions=3)
        actual = instance.get_optimal_action_and_reward()
        for i, value in enumerate(instance._true_action_rewards):
            if value == actual["reward_value"]:
                assert actual["action"] == i
