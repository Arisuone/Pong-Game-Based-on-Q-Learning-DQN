import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from pong_env import PongEnv
from dqn import DQN
from utils import save_model, plot_rewards

# Use GPU (if available), otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment: 32*32 discrete grid
env = PongEnv(width=400, height=300, bins=32, return_scorer=False)

# State dimension = ball position (x,y) + paddle position (agent_y), action dimension = up/down/stay
state_dim, action_dim = len(env.get_state()), 3

# Policy network (policy_net) and target network (target_net)
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Initially both have the same parameters

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters(), lr=2e-4)   # Small learning rate to avoid oscillation
loss_fn = nn.SmoothL1Loss()                               # Huber Loss, more robust

# Replay Buffer
memory = deque(maxlen=100000)  # Large buffer to avoid overfitting recent experiences

# Hyperparameters
epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.997  # Exploration rate gradually decays
gamma = 0.98                                           # Discount factor
batch_size = 128
episodes = 1500                                        # Total training episodes
reward_list = []
epsilon_list = []                                      # Record epsilon per episode

# Soft update parameter (slight update of target_net each iteration)
tau = 0.005

def replay():
    """Sample from replay buffer and update policy network"""
    if len(memory) < batch_size: 
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensor
    states = torch.FloatTensor(np.array(states)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Current Q values
    q_values = policy_net(states).gather(1, actions).squeeze()
    # Max Q value of next state (from target_net)
    next_q_values = target_net(next_states).max(1)[0]
    # Target values
    targets = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss and backpropagation
    loss = loss_fn(q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update target_net
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * policy_param.data)

# Main training loop
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0   # Step counter

    while not done and step_count < 500:  # Max step limit to prevent deadlock
        # ε-greedy strategy
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state).to(device)).argmax().item()

        # Execute action
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        step_count += 1

        # Update network
        replay()

    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay exploration rate
    epsilon_list.append(epsilon)

    # Print log every 10 episodes
    if ep % 10 == 0:
        print(f"Episode {ep}, Reward {total_reward:.2f}, Steps {step_count}, Epsilon {epsilon:.3f}")

# -------------------------
# Save model and logs after training
# -------------------------
# Save model
save_model(policy_net, "models/pong.pth")

# Save reward and epsilon data
np.save("logs/reward_list.npy", reward_list)
np.save("logs/epsilon_list.npy", epsilon_list)
print("Training logs saved to logs/")

# Plot reward curve
plot_rewards(reward_list, epsilons=epsilon_list, path="logs/reward_curve.png", window=50)
print("Reward curve saved to logs/reward_curve.png")