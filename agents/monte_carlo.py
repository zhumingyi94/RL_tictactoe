import numpy as np

class MonteCarloAgent:
    def __init__(self, env, gamma, epsilon):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_value = np.zeros(self.num_states)
        self.state_action_counts = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def generate_episode(self):
        episode = []
        state = self.env.reset()[0]
        done = False
        while not done:
            action = self.epsilon_greedy(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def update_state_values(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            self.state_action_counts[state] += 1
            self.state_value[state] += (G - self.state_value[state]) / self.state_action_counts[state]

    def train(self, episodes):
        for episode in range(episodes):
            episode_data = self.generate_episode()
            self.update_state_values(episode_data)
            self.update_policy()

    def update_policy(self):
        for state in range(self.num_states):
            self.policy[state] = np.argmax(self.state_value[state])

    def decode_state(self, state):
        return np.unravel_index(state, (self.env.board_size, self.env.board_size))

    def encode_state(self, board):
        return np.ravel(board).argmax()
