from ale_py import ALEInterface
ale = ALEInterface()

import gymnasium as gym
import matplotlib.pyplot as plt
import json
import cv2 #OpenCV for image preprocessing

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array") # rgb_array or human; Remove rendering during training or evaluation

# Preprocessing; source: https://www.findingtheta.com/blog/mastering-ataris-pong-with-reinforcement-learning-overcoming-sparse-rewards-and-optimizing-performance
def preprocess_frame(frame):    
     
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to grayscale  (Reduces inputspace to 1 channel)
    resized_frame = cv2.resize(gray_frame, (84, 84))   # Resize the frame  to smaller resolution for faster processing, typical size for DQN
    normalized_frame = resized_frame / 255.0    # Normalize pixel values helps neural network train more efficiently by using smaller and more consistent input values

    return normalized_frame


num_episodes = 200 # Number of episodes to run
max_iter = 500 # Maximum steps per episode
nStepsRandom = []  # Store the number of steps per episode
cumulative_rewards = []  # Store cumulative rewards for each episode


for episode in range(num_episodes):
    observation, info = env.reset(seed=42) # Reset the environment to generate the first observations
    episode_reward = 0
    episode_steps = 0  # Counter for the number of steps in each episode
    episode_over = False

    for i in range(max_iter):
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(observation)  # Use the image from the observation, prepocressed frames

        action = env.action_space.sample()  # Random action policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward # Accumulate rewards for each step in the environment
        episode_steps += 1  # Count steps in the episode

        # Check if the episode is over (either terminated or truncated)
        if terminated or truncated:
            episode_over = True
            break  # End the loop once the episode is over

    # Append the results to the lists after the episode ends
    nStepsRandom.append(episode_steps)  # Store the number of steps in this episode
    cumulative_rewards.append(episode_reward)  # Save the cumulative reward after episode is over
    print(f"Episode {episode + 1}: Cumulative reward = {episode_reward}")

env.close()


# Cumulative Reward Plot
plt.plot(cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative reward')
plt.title('Cumulative reward over time - Random Policy')
plt.savefig("cumulative_rewards.jpg", format="jpg", dpi=300) # Save the plot as a JPEG file
plt.show()

# #Average cumulative reward
average_cumulative_reward = sum(cumulative_rewards) / num_episodes

# Save results as JSON
random_policy_results = {
    "policy": "random",
    "num_episodes": num_episodes,
    "cumulative_rewards": cumulative_rewards,
    "average_reward": average_cumulative_reward
}
# Save the results to the correct path
save_path = '/kaggle/working/random_policy_results.json'

# Save the data to a JSON file for later analysis
with open(save_path, "w") as f:
    json.dump(random_policy_results, f, indent=4)
print(f"File saved to {save_path}")

print("Random policy results saved to logs/random_policy_results.json")