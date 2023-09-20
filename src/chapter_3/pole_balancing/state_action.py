import numpy as np

from src.chapter_3.pole_balancing.discretization_config import DiscretizationConfig
from src.chapter_3.pole_balancing.state import IndexedState


class StateActionValues:
    def __init__(self, discretization_config: DiscretizationConfig) -> None:
        num_actions = 2
        self.Q = np.zeros(
            shape=(
                discretization_config.num_buckets_for_position,
                discretization_config.num_buckets_for_angle,
                num_actions,
            )
        )

    def get_action_value(self, indexed_state: IndexedState, action: int) -> float:
        return self.Q[indexed_state.position_index][indexed_state.angle_index][action]

    def get_max_reward_action(self, indexed_state: IndexedState) -> int:
        return np.argmax(self.Q[indexed_state.position_index][indexed_state.angle_index])

    def update_Q(self, indexed_state: IndexedState, action: int, new_value: float):
        self.Q[indexed_state.position_index][indexed_state.angle_index][action] = new_value
