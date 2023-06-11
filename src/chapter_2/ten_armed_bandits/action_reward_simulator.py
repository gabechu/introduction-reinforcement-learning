from typing import Dict

import numpy as np
import numpy.typing as npt


class ActionRewardSimulator(object):
    """Simulate reward for every action."""

    def __init__(self, num_actions: int, seed: int = 22) -> None:
        self._num_actions = num_actions
        self._random_generator = np.random.RandomState(seed)
        self._true_action_rewards = self._set_rewards()

    def _set_rewards(self) -> npt.NDArray:
        return self._random_generator.standard_normal(size=self._num_actions)

    def get_optimal_action_and_reward(self) -> Dict:
        """Get the optimal action and reward for the underlying distribution."""
        return {"action": np.argmax(self._true_action_rewards), "reward_value": np.max(self._true_action_rewards)}

    def get_reward(self, action: int) -> float:
        """Generate reward for the current time step."""
        return self._random_generator.normal(self._true_action_rewards[action], 1, size=1).item()
