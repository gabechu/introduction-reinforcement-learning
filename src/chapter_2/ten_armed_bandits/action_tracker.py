from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes


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
        self._num_bandits = num_bandits
        self.trackers = [ActionTracker(num_actions) for _ in range(num_bandits)]

    def get_action_tracker(self, bandit: int) -> ActionTracker:
        """Fetch action tracker for the specific bandit problem."""
        return self.trackers[bandit]

    def _coalesce_actions(self) -> npt.NDArray:
        """Combine actions from different bandits into a matrix with the shape (num_bandits, num_steps)."""
        return np.array([tracker.actions for tracker in self.trackers])

    def calculate_percentage_of_optimal_action(self, optimal_action: int) -> npt.NDArray:
        """Calculate the percentage of optimal action taken per step."""
        all_actions = self._coalesce_actions()
        return (all_actions == optimal_action).sum(axis=0) / self._num_bandits

    def plot_percentage_of_optimal_action(self, ax: Axes, optimal_action: int, label: str):
        """Plot mean rewards across steps."""
        ax.plot(self.calculate_percentage_of_optimal_action(optimal_action), label=label)
        ax.legend()

    def calculate_percentage_of_exploit(self) -> npt.NDArray:
        """Calculate the percentage of exploit behaviors per step. The exploit behavior is identified when the
        current action is the same as the previous action."""
        all_actions = self._coalesce_actions()
        num_steps = all_actions.shape[1]
        step_n_actions = all_actions[:, : num_steps - 1]
        step_n_plus_one_actions = all_actions[:, 1:]

        return (step_n_actions == step_n_plus_one_actions).sum(axis=0) / self._num_bandits

    def plot_percentage_of_exploit(self, ax: Axes, label: str):
        """Plot the percentage of exploit behaviors across steps."""
        ax.plot(self.calculate_percentage_of_exploit(), label=label)
        ax.legend()
