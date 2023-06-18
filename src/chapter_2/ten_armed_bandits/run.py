"""Implement an epsilon greedy algorithm simulated for a 10 armed bandit problem with actions sampled from
normal distributions."""

import matplotlib.pyplot as plt

from src.chapter_2.ten_armed_bandits.action_reward_simulator import RandomWalkRewardSimulator
from src.chapter_2.ten_armed_bandits.action_tracker import ActionTrackerPool
from src.chapter_2.ten_armed_bandits.greedy_agent import run_epsilon_greedy_agent
from src.chapter_2.ten_armed_bandits.reward_tracker import RewardTrackerPool


def main():
    """Main function."""
    num_actions = 10
    num_bandits = 2000
    steps_per_bandit = 1000
    epsilons = [0.0, 0.01, 0.1]
    action_reward_simulator = RandomWalkRewardSimulator(num_actions, seed=None)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    for epsilon in epsilons:
        reward_tracker_pool = RewardTrackerPool(num_bandits)
        action_tracker_pool = ActionTrackerPool(num_bandits=num_bandits, num_actions=num_actions)

        for bandit in range(num_bandits):
            reward_tracker = reward_tracker_pool.get_reward_tracker(bandit)
            action_tracker = action_tracker_pool.get_action_tracker(bandit)
            run_epsilon_greedy_agent(
                action_reward_simulator=action_reward_simulator,
                action_tracker=action_tracker,
                reward_tracker=reward_tracker,
                steps=steps_per_bandit,
                epsilon=epsilon,
            )

        # plot average reward
        reward_tracker_pool.plot_mean_of_mean_rewards(ax1, f"epsilon={epsilon}")
        # plot % optimal acton
        action_tracker_pool.plot_percentage_of_optimal_action(
            ax2, action_reward_simulator.get_optimal_action(), label=f"epsilon={epsilon}"
        )
    plt.show()


if __name__ == "__main__":
    main()
