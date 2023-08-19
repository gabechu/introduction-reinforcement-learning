import logging

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


def calculate_state_value_function_for_random_policy(
    state: State,
    state_value_matrix: np.ndarray,
    discount_factor: float,
) -> float:
    action_probability = 1 / len(Action)

    v_value = 0
    for action in Action:
        next_state = regularize_state(get_next_state(action=action, current_state=state))
        next_reward = evaluate_reward(current_state=state, action=action)
        # The probabilities of both p(r | s, a, s') and p(s' | s, a) are one. So they are omitted. The reason that
        # those probabilities are zero is that given the current state and an action, the outcomes for both reward
        # and the next state are deterministic
        q_value = next_reward + discount_factor * get_state_value(next_state, state_value_matrix)
        v_value += action_probability * q_value

    return v_value


def calculate_state_value_function_for_optimal_polic(
    state: State, state_value_matrix: np.ndarray, discount_factor: float
) -> float:
    q_values = []

    for action in Action:
        next_state = regularize_state(get_next_state(action=action, current_state=state))
        next_reward = evaluate_reward(current_state=state, action=action)
        q_value = next_reward + discount_factor * get_state_value(next_state, state_value_matrix)
        q_values.append(q_value)

    return max(q_values)


def run_calculation_for_random_policy():
    # a 5 x 5 grid
    state_value_matrix = np.zeros(shape=(5, 5))
    states = [State(x=x, y=y) for x in range(5) for y in range(5)]

    for i in range(1000):
        diff = 0
        # sweep states and update
        for state in states:
            old_state_value = get_state_value(state, state_value_matrix)
            new_state_value = calculate_state_value_function_for_random_policy(state, state_value_matrix, 0.9)

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


def run_calculation_for_optimal_policy():
    # a 5 x 5 grid
    state_value_matrix = np.zeros(shape=(5, 5))
    states = [State(x=x, y=y) for x in range(5) for y in range(5)]

    for i in range(1000):
        diff = 0
        # sweep states and update
        for state in states:
            old_state_value = get_state_value(state, state_value_matrix)
            new_state_value = calculate_state_value_function_for_optimal_polic(state, state_value_matrix, 0.9)

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
    # run_calculation_for_random_policy()
    run_calculation_for_optimal_policy()
