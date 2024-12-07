from ale_py import ALEInterface
ale = ALEInterface()

import gymnasium as gym

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()

# # Initialise the environment
# env = gym.make("LunarLander-v3", render_mode="human")

# # Reset the environment to generate the first observation
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     # this is where you would insert your policy
#     action = env.action_space.sample()

#     # step (transition) through the environment with the action
#     # receiving the next observation, reward and if the episode has terminated or truncated
#     observation, reward, terminated, truncated, info = env.step(action)

#     # If the episode has ended then we can reset to start a new episode
#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()