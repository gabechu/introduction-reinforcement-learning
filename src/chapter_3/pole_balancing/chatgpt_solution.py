import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Set the number of episodes and maximum number of steps per episode
num_episodes = 1000
max_steps = 500

# Set the learning rate, discount factor, and exploration rate
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 0.1

# Initialize the Q-table with zeros
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Run the Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    for step in range(max_steps):
        # Choose an action using the epsilon-greedy strategy
        if np.random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Perform the selected action
        next_state, reward, done, _ = env.step(action)

        # Update the Q-value using the Q-learning update rule
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])

        state = next_state

        if done:
            break

# Evaluate the learned policy
total_reward = 0
num_episodes_eval = 100

for episode in range(num_episodes_eval):
    state = env.reset()

    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

average_reward = total_reward / num_episodes_eval
print("Average reward:", average_reward)
