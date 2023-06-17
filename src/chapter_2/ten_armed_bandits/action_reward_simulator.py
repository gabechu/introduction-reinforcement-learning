from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ActionRewardSimulator(ABC):
    """Base class for defining action reward distributions."""

    def __init__(self, num_actions: int, seed: int = 22) -> None:
        self._seed = seed
        self._num_actions = num_actions
        self._true_action_rewards = self._set_true_rewards()

    @property
    def _stateful_generator(self):
        return np.random.RandomState(self._seed)  # pylint: disable=E1101

    @abstractmethod
    def _set_true_rewards(self) -> npt.NDArray:
        ...

    @abstractmethod
    def generate_reward(self, action: int) -> float:
        """Simulate a reward using the provided distribution for the action."""

    def get_optimal_action(self) -> int:
        """Get the optimal action based on the true action rewards."""
        return np.argmax(self._true_action_rewards).item()

    def get_optimal_reward(self) -> float:
        """Get the optimal reward value based on the true action rewards."""
        return np.max(self._true_action_rewards)


class NormalActionRewardSimulator(ActionRewardSimulator):
    """Simulating action rewards from normal distributions."""

    def _set_true_rewards(self) -> npt.NDArray:
        return self._stateful_generator.standard_normal(size=self._num_actions)

    def generate_reward(self, action: int):
        return self._stateful_generator.normal(self._true_action_rewards[action], 1, size=1).item()


class PoissonActionRewardSimulator(ActionRewardSimulator):
    """Simulating action rewards from Poisson distributions."""

    def _set_true_rewards(self) -> npt.NDArray:
        print("Execute once")
        return self._stateful_generator.poisson(3, size=self._num_actions)

    def generate_reward(self, action: int) -> float:
        return self._stateful_generator.poisson(self._true_action_rewards[action], size=1).item()
