"""Implement an epsilon greedy algorithm simulated for a 10 armed bandit problem with actions sampled from
normal distributions."""
import matplotlib.pyplot as plt
import numpy as np
from pyinstrument import Profiler

from src.chapter_2.ten_armed_bandits.action_reward_simulator import NormalActionRewardSimulator
from src.chapter_2.ten_armed_bandits.action_tracker import ActionTrackerPool
from src.chapter_2.ten_armed_bandits.greedy_agent import run_epsilon_greedy_agent
from src.chapter_2.ten_armed_bandits.reward_tracker import RewardTrackerPool


def main():
    """Main function."""
    # configuration
    num_actions = 10
    num_bandits = 2000
    steps_per_bandit = 1000
    epsilons = [0.0]
    action_reward_simulator = NormalActionRewardSimulator(num_actions, seed=None)
    initial_action_rewards = np.full(shape=num_actions, fill_value=5)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))

    # run program
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
                initial_action_rewards=initial_action_rewards,
                steps=steps_per_bandit,
                epsilon=epsilon,
            )

        # generate plots
        plot_label = f"epsilon={epsilon}"
        reward_tracker_pool.plot_mean_of_mean_rewards(ax1, plot_label)
        action_tracker_pool.plot_percentage_of_optimal_action(
            ax2, action_reward_simulator.get_optimal_action(), plot_label
        )
        action_tracker_pool.plot_percentage_of_exploit(ax3, plot_label)

    ax1.title.set_text("Average Reward")
    ax2.title.set_text("% Optimal Action")
    ax3.title.set_text("% Exploit")
    plt.show()


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    main()
    profiler.stop()
    profiler.print()
