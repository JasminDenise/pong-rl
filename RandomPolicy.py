from ale_py import ALEInterface
ale = ALEInterface()

import gymnasium as gym
import matplotlib.pyplot as plt
import json
import cv2 #OpenCV for image preprocessing

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode="None") # None or human; Remove rendering during training or evaluation

# Preprocessing; source: https://www.findingtheta.com/blog/mastering-ataris-pong-with-reinforcement-learning-overcoming-sparse-rewards-and-optimizing-performance
def preprocess_frame(frame):    
     
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to grayscale  (Reduces inputspace to 1 channel)
    resized_frame = cv2.resize(gray_frame, (84, 84))   # Resize the frame  to smaller resolution for faster processing, typical size for DQN
    normalized_frame = resized_frame / 255.0    # Normalize pixel values helps neural network train more efficiently by using smaller and more consistent input values

    return normalized_frame


# Number of episodes to run
num_episodes = 10
nStepsRandom = []  # Speichern der Schritte pro Episode
cumulative_rewards = []  # Store cumulative rewards for each episode


for episode in range(num_episodes):
    observation, info = env.reset(seed=42) # Reset the environment to generate the first observations
    episode_reward = 0
    episode_steps = 0  # Counter for the number of steps in each episode
    episode_over = False

    while not episode_over:
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(observation['image'])  # Use the image from the observation, prepocressed frames

        action = env.action_space.sample()  # Random action policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward # Accumulate rewards for each step in the environment
        episode_steps += 1  # Count steps in the episode
        episode_over = terminated or truncated # Check if episode is over (timeout or game over)

    nStepsRandom.append(episode_steps)  # Store the number of steps in this episode
    cumulative_rewards.append(episode_reward)  # Save the cumulative reward after episode is over
    print(f"Episode {episode + 1}: Cumulative reward = {episode_reward}")

env.close()

#Average cumulative reward to summarize the random policy's overall performance
average_cumulative_reward = sum(cumulative_rewards) / num_episodes
average_steps = sum(nStepsRandom) / num_episodes
print("\n--- Performance of random policy ---")
print(f"Average cumulative reward over {num_episodes} episodes: {average_cumulative_reward:.2f}")
print(f"Average steps per episode: {average_steps:.2f}")

# Plot histogram the cumulative reward
plt.plot(cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Performance of random policy")
plt.legend()
plt.savefig("historgrams/cumulative_rewards.jpg", format="jpg") # Save the plot as a JPEG file
plt.show()

# Plot histogram of steps per episode
plt.hist(nStepsRandom, bins=20, edgecolor="black")
plt.xlim(0, max(nStepsRandom))
plt.xlabel("Number of steps per episode")
plt.ylabel("Frequency")
plt.title("Distribution of steps per episode for random policy")
plt.savefig("historgrams/steps_per_episode.jpg", format="jpg")  # Save the plot as a JPEG file
plt.show()

# Save results as JSON
random_policy_results = {
    "policy": "random",
    "num_episodes": len(cumulative_rewards),
    "cumulative_rewards": cumulative_rewards,
    "average_reward": average_cumulative_reward,
    "steps_per_episode": nStepsRandom,
    "average_steps": average_steps
}

# Save the data to a JSON file for later analysis
with open("logs/random_policy_results.json", "w") as f:
    json.dump(random_policy_results, f, indent=4)

print("Random policy results saved to logs/random_policy_results.json")