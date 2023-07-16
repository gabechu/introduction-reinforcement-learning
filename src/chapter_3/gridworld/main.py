import numpy as np

from src.chapter_3.gridworld.action import Action
from src.chapter_3.gridworld.state import State, get_next_state, regularize_state
from src.chapter_3.gridworld.state_value_function import (
    get_state_value,
    update_state_value_matrix,
)
from src.chapter_3.gridworld.reward import evaluate_reward


def calculate_new_state_value(
    state: State,
    state_value_matrix: np.ndarray,
    discount_factor: float = 0.8,
) -> float:
    # evaluate the policy where the agent randomly selects an action
    actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
    action_probability = 1 / len(actions)

    new_state_value = 0
    for action in actions:
        next_state = regularize_state(get_next_state(action, state))
        next_reward = evaluate_reward(current_state=state, action=action)
        next_state_value = action_probability * (
            next_reward
            + discount_factor * get_state_value(next_state, state_value_matrix)
        )
        new_state_value += next_state_value

    return new_state_value


if __name__ == "__main__":
    # a 5 x 5 grid
    state_value_matrix = np.zeros(shape=(5, 5))
    states = [State(x=x, y=y) for x in range(5) for y in range(5)]

    for i in range(1000):
        diff = 0
        for state in states:
            old_state_value = get_state_value(state, state_value_matrix)
            new_state_value = calculate_new_state_value(state, state_value_matrix, 0.9)
            update_state_value_matrix(
                state=state,
                new_state_value=new_state_value,
                state_value_matrix=state_value_matrix,
            )
            diff += np.absolute(new_state_value - old_state_value)
        
        if np.isclose(diff, 0):
            print(f"Converged in {i + 1} steps")
            break
            

    print(state_value_matrix)
