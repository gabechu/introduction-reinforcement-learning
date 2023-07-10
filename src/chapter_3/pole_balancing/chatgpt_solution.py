import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Set the number of episodes and maximum number of steps per episode
num_episodes = 5000
max_steps = 1000

# Set the learning rate, discount factor, and exploration rate
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 0.1

# Initialize the Q-table with zeros
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
Q = {}
state, info = env.reset(seed=42)
state = tuple(state)

# Run the Q-learning algorithm
for episode in range(num_episodes):
    for step in range(max_steps):
        # Choose an action using the epsilon-greedy strategy
        if np.random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()
        else:
            if state in Q:
                action = np.argmax(Q[state])
            else:
                Q[state] = np.zeros(num_actions)
                action = env.action_space.sample()

        # Perform the selected action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(next_state)

        if next_state not in Q:
            Q[next_state] = np.zeros(num_actions)

        # Update the Q-value using the Q-learning update rule
        Q[state][action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state][action])

        state = next_state

        if terminated or truncated:
            observation, info = env.reset()

# Evaluate the learned policy
total_reward = 0
num_episodes_eval = 100

for episode in range(num_episodes_eval):
    state, info = env.reset()
    state = tuple(state)

    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

average_reward = total_reward / num_episodes_eval
print("Average reward:", average_reward)
env.close()
