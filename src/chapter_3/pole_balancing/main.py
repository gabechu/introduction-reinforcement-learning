import gymnasium as gym
import numpy as np
from collections import defaultdict


env = gym.make("CartPole-v1", render_mode="human")


class State:
    def __init__(self, num_buckets: int) -> None:
        self._num_buckets = num_buckets
        self.states = self.build_states()

    def discretize_position(self) -> np.ndarray:
        return np.linspace(start=-1.2, stop=1.2, num=self._num_buckets)

    def build_states(self) -> list[tuple[float]]:
        sliced_positions = self.discretize_position()
        return list(zip(sliced_positions[:-1], sliced_positions[1:]))

    def get_state_index(self, observation: list[float]) -> int:
        position = observation[0]
        for index, state in enumerate(self.states):
            if state[0] <= position < state[1]:
                return index
        raise Exception("Invalid position; no matching state found.")


class ActionValueFunction:
    def __init__(self, num_buckets: int) -> None:
        num_actions = 2
        self.Q = np.zeros(shape=(num_buckets, num_actions))

    def get_action_value(self, state_index: int, action: int) -> float:
        return self.Q[state_index][action]

    def get_max_reward_action(self, state_index: int) -> int:
        return np.argmax(self.Q[state_index])

    def update_Q(self, state_index: int, action: int, new_value: float):
        # print(f"{state_index}, {action}, {new_value}")
        self.Q[state_index][action] = new_value


def run_episode(
    discount_factor: float,
    state: State,
    action_value_func: ActionValueFunction,
    exploration_rate: float = 0.1,
    limit: int = 500,
):
    observation, info = env.reset(seed=42)
    # step 0
    current_action = env.action_space.sample()
    current_state_index = state.get_state_index(observation)

    for step in range(limit):
        observation, reward, terminated, truncated, info = env.step(current_action)

        next_state_index = state.get_state_index(observation)
        next_action = action_value_func.get_max_reward_action(next_state_index)

        if terminated or truncated:
            print(f"Died after {step} steps")
            action_value_func.update_Q(state_index=current_state_index, action=current_action, new_value=reward)
            break
        else:
            new_q_value = reward + discount_factor * action_value_func.get_action_value(next_state_index, next_action)
            action_value_func.update_Q(state_index=current_state_index, action=current_action, new_value=new_q_value)

        current_state_index = next_state_index
        if np.random.uniform(0, 1) < exploration_rate:
            current_action = env.action_space.sample()
        else:
            current_action = next_action


state = State(num_buckets=150)
action_value_func = ActionValueFunction(num_buckets=150)

for num_episode in range(100):
    run_episode(discount_factor=0.8, state=state, action_value_func=action_value_func)
env.close()
