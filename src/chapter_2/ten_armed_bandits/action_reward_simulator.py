from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt


class ActionRewardSimulator(ABC):
    """Base class for defining action reward distributions."""

    def __init__(self, num_actions: int, seed: Optional[int] = 22) -> None:
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

    @cached_property
    def _cache(self) -> Dict[int, List]:
        return {
            action: list(self._stateful_generator.normal(reward, 1, size=50_000))
            for action, reward in enumerate(self._true_action_rewards)
        }

    def _set_true_rewards(self) -> npt.NDArray:
        return self._stateful_generator.standard_normal(size=self._num_actions)

    def generate_reward(self, action: int) -> float:
        if len(self._cache[action]) == 0:
            del self.__dict__["_cache"]
        return self._cache[action].pop()


class PoissonActionRewardSimulator(ActionRewardSimulator):
    """Simulating action rewards from Poisson distributions."""

    @cached_property
    def _cache(self) -> Dict[int, List]:
        return {
            action: list(self._stateful_generator.poisson(reward, size=50_000))
            for action, reward in enumerate(self._true_action_rewards)
        }

    def _set_true_rewards(self) -> npt.NDArray:
        return self._stateful_generator.poisson(3, size=self._num_actions)

    def generate_reward(self, action: int) -> float:
        if len(self._cache[action]) == 0:
            del self.__dict__["_cache"]
        return self._cache[action].pop()


class RandomWalkRewardSimulator(ActionRewardSimulator):
    """Simulating action rewards from normal distributions, where each action reward samples from
    a distribution with a mean value that follows a random walk.
    Note: running this simulator coudl be slow as it has not been optimised yet.
    """

    def _set_true_rewards(self) -> npt.NDArray:
        return np.zeros(self._num_actions)

    def _update_true_rewards(self):
        random_factors = self._stateful_generator.normal(0, 0.01, size=self._num_actions)
        self._true_action_rewards += random_factors

    def generate_reward(self, action: int) -> float:
        reward = self._stateful_generator.normal(self._true_action_rewards[action], 1, size=1).item()
        self._update_true_rewards()
        return reward
