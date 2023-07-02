import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.chapter_2.ten_armed_bandits.reward_tracker import RewardTracker, RewardTrackerPool


class TestRewardTracker(unittest.TestCase):
    def test_add_one_reward(self):
        instance = RewardTracker()
        instance.add_current_reward(0.0)
        assert instance.rewards == [0.0]

    def test_add_two_rewards(self):
        instance = RewardTracker()
        instance.add_current_reward(1)
        instance.add_current_reward(2)
        assert instance.rewards == [1, 2]

    def test_add_one_mean_reward(self):
        instance = RewardTracker()
        instance.add_mean_reward(0.0)
        actual = instance.mean_rewards
        assert actual == [0.0]

    def test_add_two_mean_reward(self):
        instance = RewardTracker()
        instance.add_mean_reward(0.0)
        instance.add_mean_reward(1.0)
        assert instance.mean_rewards == [0.0, 1.0]


class TestRewardTrackerPool(unittest.TestCase):
    def test_initialised_action_tracker_pool_size(self):
        size = 10
        instance = RewardTrackerPool(num_bandits=size)
        actual = len(instance.trackers)
        assert actual == size

    def test_get_reward_tracker(self):
        instance = RewardTrackerPool(num_bandits=3)
        assert isinstance(instance.get_reward_tracker(0), RewardTracker)
        assert isinstance(instance.get_reward_tracker(1), RewardTracker)
        assert isinstance(instance.get_reward_tracker(2), RewardTracker)

    def test_calcualte_mean_of_mean_rewards(self):
        instance = RewardTrackerPool(num_bandits=3)
        # mean rewards [1, 2, 1, 2]
        instance.trackers[0].add_mean_reward(1)
        instance.trackers[0].add_mean_reward(2)
        instance.trackers[0].add_mean_reward(1)
        instance.trackers[0].add_mean_reward(2)
        # mean rewards [3, 4, 3, 4]
        instance.trackers[1].add_mean_reward(3)
        instance.trackers[1].add_mean_reward(4)
        instance.trackers[1].add_mean_reward(3)
        instance.trackers[1].add_mean_reward(4)
        # mean rewards [5, 6, 5, 6]
        instance.trackers[2].add_mean_reward(5)
        instance.trackers[2].add_mean_reward(6)
        instance.trackers[2].add_mean_reward(5)
        instance.trackers[2].add_mean_reward(6)

        actual = instance.calcualte_mean_of_mean_rewards()
        assert_array_equal(actual, np.array([3.0, 4.0, 3.0, 4.0]))
