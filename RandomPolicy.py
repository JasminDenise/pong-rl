import gymnasium as gym
import matplotlib.pyplot as plt
import json
import cv2  # OpenCV for image preprocessing
import numpy as np

from ale_py import ALEInterface
ale = ALEInterface()

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode=None)  # Remove rendering during training for efficiency
env.reset(seed=42)  # Initialize seed for reproducibility

# Preprocessing function; source: https://www.findingtheta.com/blog/mastering-ataris-pong-with-reinforcement-learning-overcoming-sparse-rewards-and-optimizing-performance 
"""
    Preprocess a frame of the Pong game into a suitable format for the DQN.

    The preprocessing consists of three steps:
    1. Convert the frame from RGB to grayscale.
    2. Resize the frame from 160x210 to 84x84.
    3. Normalize the pixels values from [0, 255] to [0, 1].
"""
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (84, 84))  # Resize for faster processing
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame

num_episodes = 200  # Number of episodes to run
max_iter = 500  # Maximum steps per episode
nStepsRandom = []  # Store the number of steps per episode
cumulative_rewards = []  # Store cumulative rewards for each episode

for episode in range(num_episodes):
    observation, info = env.reset()  # Reset the environment to generate the first observations
    episode_reward = 0
    episode_steps = 0  # Counter for the number of steps in each episode

    for _ in range(max_iter):
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(observation)

        action = env.action_space.sample()  # Random action policy
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward  # Accumulate rewards for each step in the environment
        episode_steps += 1  # Count steps in the episode

        # Check if the episode is over
        # terminated: Episode ends due to terminal state (e.g., agent loses or wins)
        # truncated: Episode ends due to external constraints (e.g., step limit exceeded)
        if terminated or truncated: 
            break

    # Append the results to the lists after the episode ends
    nStepsRandom.append(episode_steps)  # Store the number of steps in this episode
    cumulative_rewards.append(episode_reward)  # Save the cumulative reward after episode is over
    print(f"Episode {episode + 1}: Cumulative reward = {episode_reward}")

env.close()

# Calculate moving average of cumulative rewards to identify trends in the cumulative rewards and smooth out fluctuations
window_size = 10 # Every 10 espiodes
moving_avg = np.convolve(cumulative_rewards, np.ones(window_size) / window_size, mode='valid') 

# Calculate win rate for random policy, should be 50 %
wins = sum(1 for reward in cumulative_rewards if reward > 0)
win_rate = wins / num_episodes * 100

# Plot cumulative rewards and moving average
plt.plot(cumulative_rewards, label="Cumulative Rewards", alpha=0.5)
plt.plot(range(len(moving_avg)), moving_avg, label=f"{window_size}-Episode Moving Average")
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward and Moving Average - Random Policy')
plt.legend()
plt.savefig("cumulative_rewards.jpg", format="jpg", dpi=300)
plt.show()

# Average cumulative reward
average_cumulative_reward = sum(cumulative_rewards) / num_episodes

# Save results as JSON
random_policy_results = {
    "policy": "random",
    "num_episodes": num_episodes,
    "cumulative_rewards": cumulative_rewards,
    "average_reward": average_cumulative_reward,
    "win_rate": win_rate,
    "moving_average": moving_avg.tolist()
}

save_path = 'random_policy_results.json'

# Save the data to a JSON file for later analysis
with open(save_path, "w") as f:
    json.dump(random_policy_results, f, indent=4)
print(f"Results saved to {save_path}")