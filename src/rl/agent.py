import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Neural network architecture
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class DQNAgent:
    """Deep Q-Network agent implementing epsilon-greedy policy and experience replay."""
    def __init__(self, state_size=7, action_size=2, learning_rate=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = 64
        self.train_step_count = 0
        self.target_update_freq = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy and target networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def act(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        return int(torch.argmax(q_values[0]).item())

    def memorize(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.append((state.tolist(), action, reward, next_state.tolist(), done))

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def train_step(self):
        """Update Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch], dtype=np.float32)
        actions = np.array([experience[1] for experience in batch], dtype=np.int64)
        rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in batch], dtype=np.float32)
        dones = np.array([experience[4] for experience in batch], dtype=np.bool_)

        state_batch = torch.from_numpy(states).to(self.device)
        next_state_batch = torch.from_numpy(next_states).to(self.device)
        action_batch = torch.from_numpy(actions).to(self.device)
        reward_batch = torch.from_numpy(rewards).to(self.device)
        done_batch = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        q_values = self.policy_net(state_batch)
        q_action = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        loss = self.criterion(q_action, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict()) 