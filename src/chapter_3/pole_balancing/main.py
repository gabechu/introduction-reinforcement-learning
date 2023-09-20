import gymnasium as gym
import numpy as np

from src.chapter_3.pole_balancing.discretization_config import DiscretizationConfig
from src.chapter_3.pole_balancing.state import State, StateBuilder
from src.chapter_3.pole_balancing.state_action import StateActionValues

env = gym.make("CartPole-v1", render_mode="human")


def run_episode(
    discount_factor: float,
    state: State,
    state_action_values: StateActionValues,
    exploration_rate: float = 0.1,
    limit: int = 500,
):
    observation, info = env.reset(seed=42)
    # step 0
    current_action = env.action_space.sample()
    current_indexed_state = state.get_indexed_state(observation)

    for step in range(limit):
        observation, reward, terminated, truncated, info = env.step(current_action)

        next_indexed_state = state.get_indexed_state(observation)
        next_action = state_action_values.get_max_reward_action(next_indexed_state)

        if terminated or truncated:
            print(f"Died after {step} steps")
            state_action_values.update_Q(indexed_state=current_indexed_state, action=current_action, new_value=reward)
            break
        else:
            new_q_value = reward + discount_factor * state_action_values.get_action_value(
                next_indexed_state, next_action
            )
            state_action_values.update_Q(
                indexed_state=current_indexed_state, action=current_action, new_value=new_q_value
            )

        current_indexed_state = next_indexed_state
        if np.random.uniform(0, 1) < exploration_rate:
            current_action = env.action_space.sample()
        else:
            current_action = next_action


discretization_config = DiscretizationConfig(num_buckets_for_position=150, num_buckets_for_angle=150)

state = StateBuilder(discretization_config).build_state()
state_action_values = StateActionValues(discretization_config)

for num_episode in range(100):
    run_episode(discount_factor=0.8, state=state, state_action_values=state_action_values)
env.close()
