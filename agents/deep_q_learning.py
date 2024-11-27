import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=32, learning_rate=0.001, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).float()
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        target_q_values = q_values.clone()

        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000):
        for e in range(episodes):
            state, _ = self.env.reset()
            state = np.flatten(state)  # Flatten state to match network input shape
            done = False
            total_reward = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = np.flatten(next_state)
                self.remember(state, action, reward, next_state, terminated)
                self.replay()

                state = next_state
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

            # Update the target model every few episodes
            if e % 10 == 0:
                self.update_target_model()
