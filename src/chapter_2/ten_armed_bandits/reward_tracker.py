import numpy as np
import numpy.typing as npt


class RewardTracker(object):
    """Track the reward obtained at each time step.
    It allows for the monitoring and analysis of the reward progress over time.
    """

    def __init__(self):
        self.rewards = []
        self.mean_rewards = []

    def add_current_reward(self, reward: float):
        """Add the current reward to reward list."""
        self.rewards.append(reward)

    def add_mean_reward(self, reward: float):
        """Add the mean reward for the action being taken."""
        self.mean_rewards.append(reward)


class RewardTrackerPool(object):
    """A pool of reward trackers."""

    def __init__(self, num_bandits: int) -> None:
        self.trackers = [RewardTracker() for _ in range(num_bandits)]

    def get_reward_tracker(self, bandit: int) -> RewardTracker:
        """Fetch a reward tracker for the specific bandit problem."""
        return self.trackers[bandit]

    def calcualte_mean_of_mean_rewards(self) -> npt.NDArray:
        """Calculate mean rewards for each step across different runs of bandits."""
        all_mean_rewards = np.array([tracker.mean_rewards for tracker in self.trackers])
        return all_mean_rewards.mean(axis=0)

    def plot_mean_of_mean_rewards(self, ax, label: str):
        """Plot mean rewards across different steps."""
        ax.plot(self.calcualte_mean_of_mean_rewards(), label=label)
        ax.legend()
