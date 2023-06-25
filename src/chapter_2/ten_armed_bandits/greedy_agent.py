import numpy as np
import numpy.typing as npt

from src.chapter_2.ten_armed_bandits.action_reward_simulator import ActionRewardSimulator
from src.chapter_2.ten_armed_bandits.action_tracker import ActionTracker
from src.chapter_2.ten_armed_bandits.reward_tracker import RewardTracker
from src.chapter_2.ten_armed_bandits.utils import weighted_incremental_average_update


def run_epsilon_greedy_agent(
    action_reward_simulator: ActionRewardSimulator,
    action_tracker: ActionTracker,
    reward_tracker: RewardTracker,
    initial_action_rewards: npt.NDArray,
    steps: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Estimate rewards for a epsilon greedy agent."""

    estimated_action_rewards = initial_action_rewards.copy()
    for _ in range(steps):
        # an agent navigates using greedy strategy
        action = choose_action_with_greedy_epsilon(estimated_action_rewards, epsilon)
        action_tracker.update_action_count(action)
        current_reward = action_reward_simulator.generate_reward(action)

        # An alternative weight is: 1 / action_tracker.get_action_count(action)
        updated_mean_reward = weighted_incremental_average_update(
            old_average=estimated_action_rewards[action],
            new_value=current_reward,
            weight=0.8,
        )

        estimated_action_rewards[action] = updated_mean_reward

        # collect stats
        action_tracker.add_action(action)
        reward_tracker.add_current_reward(current_reward)
        reward_tracker.add_mean_reward(updated_mean_reward)

    return estimated_action_rewards


def choose_action_with_greedy_epsilon(
    estimated_action_rewards: npt.NDArray,
    epsilon: float,
) -> int:
    """Choose an action for a greedy epsilon agent."""
    if np.random.uniform(size=1) <= epsilon:
        action = np.random.randint(low=0, high=len(estimated_action_rewards))
    else:
        # action = np.argmax(estimated_action_rewards).item()
        # uncomment below to use for random tie breaking
        action = np.random.choice(np.flatnonzero(estimated_action_rewards == estimated_action_rewards.max()))
    return action
