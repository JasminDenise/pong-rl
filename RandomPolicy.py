from ale_py import ALEInterface
ale = ALEInterface()

import gymnasium as gym
import matplotlib.pyplot as plt
import json

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode="None") # None or human; Remove rendering during training or evaluation

# Number of episodes to run
num_episodes = 10
cumulative_rewards = []  # Store cumulative rewards for each episode


for episode in range(num_episodes):
    observation, info = env.reset(seed=42) # Reset the environment to generate the first observations
    episode_reward = 0
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # Random action policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward # Accumulate rewards for each step in the environment

        episode_over = terminated or truncated # Check if episode is over (timeout or game over)

    cumulative_rewards.append(episode_reward)  # Save the cumulative reward after episode is over
    print(f"Episode {episode + 1}: Cumulative Reward = {episode_reward}")

env.close()

#Average cumulative reward to summarize the random policy's overall performance
average_cumulative_reward = sum(cumulative_rewards) / num_episodes
print("\n--- Performance of random policy ---")
print(f"Average Cumulative Reward over {num_episodes} episodes: {average_cumulative_reward:.2f}")

# Visualize the rewards
plt.plot(cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Performance of random policy")
plt.legend()
plt.show()


#with open("random_policy_rewards.json", "w") as f:
  #  json.dump({"cumulative_rewards": cumulative_rewards}, f)


# Save policy results
random_policy_results = {
    "policy": "random",
    "num_episodes": len(cumulative_rewards),
    "cumulative_rewards": cumulative_rewards,
    "average_reward": sum(cumulative_rewards) / len(cumulative_rewards),
}

with open("logs/random_policy_results.json", "w") as f:
    json.dump(random_policy_results, f, indent=4)

print("Random policy results saved to logs/random_policy_results.json")