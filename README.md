# pong-rl

This project demonstrates the use of Deep Q-Networks (DQN) to train an agent to play the classic Atari game Pong. The agent uses reinforcement learning techniques to improve its performance over time, outperforming the baseline performance of a random policy.

## Features
- **Deep Q-Network (DQN)**: Implements a neural network to approximate Q-values for discrete actions.
- **Replay Buffer**: Stores past experiences for stable training through random sampling.
- **Target Network**: Stabilizes learning by periodically updating Q-value targets.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation during training.
- **Performance Monitoring**: Tracks cumulative rewards, win rate, and learning progress.

## How It Works
1. **Preprocessing**: The raw Pong game frames are converted to grayscale, resized to 84x84, and stacked to form a temporal representation of the environment.
2. **DQN Architecture**: A neural network with convolutional layers extracts features, followed by fully connected layers to compute Q-values.
3. **Training Loop**:
   - The agent interacts with the environment using an epsilon-greedy policy.
   - Experiences are stored in a replay buffer.
   - The policy network is updated using the Bellman equation and Mean Squared Error (MSE) loss.
   - The target network is updated periodically for stability.
4. **Evaluation**: The performance of the DQN agent is compared to a random policy using metrics like average reward and win rate.

## Requirements
- Python 3.10.4 
- PyTorch
- OpenAI Gymnasium
- ALE-Py
- NumPy
- Matplotlib
- OpenCV

## Reference
Based on Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.

## Results
The DQN agent achieves significantly better performance than a random policy with an average reward of -4.794 compared to -10.045 for the random policy.
The agent shows a clear upward trend in cumulative rewards, indicating effective learning over time.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7345bf5f-43b2-41d7-9f37-82118d49111f" alt="random policy - cumulative rewards" width="300">
   <img src="https://github.com/user-attachments/assets/af14be79-09a9-4522-8433-6b1b6dcc1583" alt="dqn - cumulative rewards" width="300">





