from dataclasses import dataclass

import numpy as np

from src.chapter_3.pole_balancing.discretization_config import DiscretizationConfig


@dataclass
class IndexedState:
    position_index: int
    angle_index: int


@dataclass
class State:
    position_buckets: list[tuple[float]]
    angle_buckets: list[tuple[float]]

    def _retrieve_position_bucket_index(self, position: float) -> int:
        for i, bucket in enumerate(self.position_buckets):
            if bucket[0] <= position < bucket[1]:
                return i
        raise Exception("Invalid position.")

    def _retrieve_angle_bucket_index(self, angle: float) -> int:
        for i, bucket in enumerate(self.angle_buckets):
            if bucket[0] <= angle < bucket[1]:
                return i
        raise Exception("Invalid angle.")

    def get_indexed_state(self, observation: list[float]) -> IndexedState:
        position = observation[0]
        angle = observation[2]

        position_index = self._retrieve_position_bucket_index(position)
        angle_index = self._retrieve_angle_bucket_index(angle)

        return IndexedState(position_index=position_index, angle_index=angle_index)


class StateBuilder:
    def __init__(self, discretization_config: DiscretizationConfig) -> None:
        self._discretization_config = discretization_config
        self.state = self.build_state()

    def discretize_positions(self) -> list[tuple[float]]:
        # https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space
        terminate_min = -2.4
        terminate_max = 2.4
        array = np.linspace(
            start=terminate_min, stop=terminate_max, num=self._discretization_config.num_buckets_for_position
        )
        return list(zip(array[:-1], array[1:]))

    def discretize_angles(self) -> list[tuple[float]]:
        # https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space
        terminate_min = -0.418
        terminate_max = 0.418
        array = np.linspace(
            start=terminate_min, stop=terminate_max, num=self._discretization_config.num_buckets_for_angle
        )
        return list(zip(array[:-1], array[1:]))

    def build_state(self) -> State:
        position_buckets = self.discretize_positions()
        angle_buckets = self.discretize_angles()

        return State(position_buckets=position_buckets, angle_buckets=angle_buckets)
