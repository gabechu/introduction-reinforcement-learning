import logging
from enum import Enum

import numpy as np

from src.chapter_3.gridworld.action import Action
from src.chapter_3.gridworld.reward import evaluate_reward
from src.chapter_3.gridworld.state import State, get_next_state, regularize_state
from src.chapter_3.gridworld.state_value_function import (
    get_state_value,
    update_state_value_matrix,
)

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Policy(Enum):
    RANDOM_POLICY = 1
    OPTIMAL_POLICY = 2


def calculate_state_value_function(
    state: State, state_value_matrix: np.ndarray, discount_factor: float, policy: Enum
) -> float:
    q_values = []
    for action in Action:
        next_state = regularize_state(get_next_state(action=action, current_state=state))
        next_reward = evaluate_reward(current_state=state, action=action)
        # The probabilities of both p(r | s, a, s') and p(s' | s, a) are one. So they are omitted. The reason that
        # those probabilities are zero is that given the current state and an action, the outcomes for both reward
        # and the next state are deterministic
        q_value = next_reward + discount_factor * get_state_value(
            state=next_state, state_value_matrix=state_value_matrix
        )
        q_values.append(q_value)

    match policy:
        case Policy.RANDOM_POLICY:
            action_probability = 1 / len(Action)
            return action_probability * sum(q_values)
        case Policy.OPTIMAL_POLICY:
            return max(q_values)
        case _:
            raise NotImplementedError()


def run_iterative_calculation(policy: Policy):
    # a 5 x 5 grid
    state_value_matrix = np.zeros(shape=(5, 5))
    states = [State(x=x, y=y) for x in range(5) for y in range(5)]

    for i in range(1000):
        diff = 0
        # sweep states and update
        for state in states:
            old_state_value = get_state_value(state=state, state_value_matrix=state_value_matrix)
            new_state_value = calculate_state_value_function(
                state=state, state_value_matrix=state_value_matrix, discount_factor=0.9, policy=policy
            )

            update_state_value_matrix(
                state=state,
                new_state_value=new_state_value,
                state_value_matrix=state_value_matrix,
            )

            diff += np.absolute(new_state_value - old_state_value)

        logger.debug("Before and after the update difference %s", diff)

        if np.isclose(diff, 0):
            logger.info("Converged in %s steps...", i + 1)
            logger.info(state_value_matrix)
            break


if __name__ == "__main__":
    run_iterative_calculation(Policy.RANDOM_POLICY)
    run_iterative_calculation(Policy.OPTIMAL_POLICY)
