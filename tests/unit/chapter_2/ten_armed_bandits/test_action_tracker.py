import unittest

from numpy.testing import assert_array_equal

from src.chapter_2.ten_armed_bandits.action_tracker import ActionTracker, ActionTrackerPool


class TestActionCounter(unittest.TestCase):
    def test_initialised_action_counter_size(self):
        instance = ActionTracker(num_actions=5)
        actual = len(instance.counts)
        assert actual == 5

    def test_add_action(self):
        instance = ActionTracker(num_actions=3)
        instance.add_action(2)
        instance.add_action(1)
        actual = instance.actions
        assert actual == [2, 1]

    def test_count_add_one_for_first_action(self):
        instance = ActionTracker(num_actions=3)
        instance.update_action_count(0)
        assert instance.get_action_count(0) == 1
        assert instance.get_action_count(1) == 0
        assert instance.get_action_count(2) == 0

    def test_count_add_one_for_the_last_action(self):
        instance = ActionTracker(num_actions=3)
        instance.update_action_count(2)
        assert instance.get_action_count(0) == 0
        assert instance.get_action_count(1) == 0
        assert instance.get_action_count(2) == 1

    def test_count_add_two_actions(self):
        instance = ActionTracker(num_actions=3)
        instance.update_action_count(0)
        instance.update_action_count(1)
        assert instance.get_action_count(0) == 1
        assert instance.get_action_count(1) == 1
        assert instance.get_action_count(2) == 0

    def test_count_add_same_action_ten_times(self):
        instance = ActionTracker(num_actions=3)
        for _ in range(10):
            instance.update_action_count(0)
        assert instance.get_action_count(0) == 10
        assert instance.get_action_count(1) == 0
        assert instance.get_action_count(2) == 0


class TestActionTrackerPool(unittest.TestCase):
    def test_initialised_action_tracker_pool_size(self):
        size = 10
        instance = ActionTrackerPool(num_bandits=size, num_actions=3)
        actual = len(instance.trackers)
        assert actual == size

    def test_get_action_tracker(self):
        instance = ActionTrackerPool(num_bandits=10, num_actions=3)
        assert isinstance(instance.get_action_tracker(0), ActionTracker)
        assert isinstance(instance.get_action_tracker(1), ActionTracker)
        assert isinstance(instance.get_action_tracker(2), ActionTracker)

    def test_calculate_percentage_of_optimal_action_for_one_bandit(self):
        instance = ActionTrackerPool(num_actions=3, num_bandits=1)
        # actions [0, 2, 0, 2]
        instance.trackers[0].add_action(0)
        instance.trackers[0].add_action(2)
        instance.trackers[0].add_action(0)
        instance.trackers[0].add_action(2)

        actual = instance.calculate_percentage_of_optimal_action(optimal_action=2)
        assert_array_equal(actual, [0, 1, 0, 1])

    def test_calculate_percentage_of_optimal_action_for_multiple_bandit(self):
        instance = ActionTrackerPool(num_actions=3, num_bandits=2)
        # actions [0, 2, 0, 2]
        instance.trackers[0].add_action(0)
        instance.trackers[0].add_action(2)
        instance.trackers[0].add_action(0)
        instance.trackers[0].add_action(2)
        # actions [0, 1, 0, 2]
        instance.trackers[1].add_action(0)
        instance.trackers[1].add_action(1)
        instance.trackers[1].add_action(0)
        instance.trackers[1].add_action(2)

        actual = instance.calculate_percentage_of_optimal_action(optimal_action=2)
        assert_array_equal(actual, [0.0, 0.5, 0.0, 1.0])

    def test_calculate_percentage_of_exploit(self):
        instance = ActionTrackerPool(num_actions=3, num_bandits=4)
        # actions [1, 2, 1, 2]
        instance.trackers[0].add_action(1)
        instance.trackers[0].add_action(2)
        instance.trackers[0].add_action(1)
        instance.trackers[0].add_action(2)
        # actions [0, 1, 0, 2]
        instance.trackers[1].add_action(0)
        instance.trackers[1].add_action(1)
        instance.trackers[1].add_action(0)
        instance.trackers[1].add_action(2)
        # actions [1, 1, 1, 1]
        instance.trackers[2].add_action(1)
        instance.trackers[2].add_action(1)
        instance.trackers[2].add_action(1)
        instance.trackers[2].add_action(1)
        # actions [2, 2, 1, 1]
        instance.trackers[3].add_action(2)
        instance.trackers[3].add_action(2)
        instance.trackers[3].add_action(1)
        instance.trackers[3].add_action(1)

        actual = instance.calculate_percentage_of_exploit()
        assert_array_equal(actual, [0.5, 0.25, 0.5])
