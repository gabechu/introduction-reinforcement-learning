from typing import List

import matplotlib.pyplot as plt
import numpy as np


class ActionTracker(object):
    """Counting occurrences of actoins. Actions are initialised with a value of 0."""

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.counts = [0] * num_actions
        self.actions: List[int] = []

    def add_action(self, action: int):
        """Add the action to action list."""
        if action >= self.num_actions:
            raise ValueError(f"Invalid action. Action value must be smaller than {self.num_actions - 1}")
        self.actions.append(action)

    def update_action_count(self, action: int):
        """Increase the count for a specific action by one."""
        self.counts[action] += 1

    def get_action_count(self, action) -> int:
        """Return the count for the asked action."""
        return self.counts[action]


class ActionTrackerPool(object):
    """A pool of action trackers."""

    def __init__(self, num_bandits: int, num_actions) -> None:
        self._trackers = [ActionTracker(num_actions) for _ in range(num_bandits)]

    def get_action_tracker(self, bandit: int) -> ActionTracker:
        """Fetch action tracker for the specific bandit problem."""
        return self._trackers[bandit]

    def calculate_percentage_of_optimal_action(self, optimal_action: int):
        """Calculate the percentage of optimal action taken per step."""
        all_actions = np.array([tracker.actions for tracker in self._trackers])

        percentage_by_step = []
        num_steps = all_actions.shape[1]
        for i in range(num_steps):
            subset_actions = all_actions[:, i]
            percentage = (subset_actions == optimal_action).sum() / subset_actions.size
            percentage_by_step.append(percentage)
        return percentage_by_step

    def plot_percentage_of_optimal_action(self, optimal_action: int, label: str):
        """Plot mean rewards across different steps."""
        plt.plot(self.calculate_percentage_of_optimal_action(optimal_action), label=label)
        plt.legend()
