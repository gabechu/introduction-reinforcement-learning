import logging

import numpy as np

from src.chapter_4.gridworld.action import Action
from src.chapter_4.gridworld.policy import Policy
from src.chapter_4.gridworld.reward import calculate_reward
from src.chapter_4.gridworld.state import (
    State,
    build_state_value_matrix,
    build_states,
    get_next_state,
    get_state_value,
    update_state_value_matrix,
)

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_policy(state: State, state_value_matrix: np.ndarray, discount_factor: float, policy: Policy) -> float:
    q_values = []
    for action in Action:
        next_state = get_next_state(action=action, current_state=state)
        reward = calculate_reward(current_state=state, action=action)
        # The probabilities of both p(r | s, a, s') and p(s' | s, a) are one. So they are omitted. The reason that
        # those probabilities are zero is that given the current state and an action, the outcomes for both reward
        # and the next state are deterministic
        q_value = reward + discount_factor * get_state_value(state=next_state, state_value_matrix=state_value_matrix)
        q_values.append(q_value)

    match policy:
        case Policy.RANDOM_POLICY:
            action_probability = 1 / len(Action)
            return action_probability * sum(q_values)
        case Policy.OPTIMAL_POLICY:
            return max(q_values)
        case _:
            raise NotImplementedError()


def run_iterative_policy_evaluation(policy: Policy, iterations: int = 5000):
    states = build_states()
    state_value_matrix = build_state_value_matrix()

    for i in range(iterations):
        diff = 0
        # sweep states and update
        for state in states:
            old_state_value = get_state_value(state=state, state_value_matrix=state_value_matrix)
            new_state_value = evaluate_policy(
                state=state, state_value_matrix=state_value_matrix, discount_factor=1.0, policy=policy
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
    run_iterative_policy_evaluation(Policy.RANDOM_POLICY)
    # run_iterative_calculation(Policy.OPTIMAL_POLICY)
