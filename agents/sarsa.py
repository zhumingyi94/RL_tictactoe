import numpy as np

class SARSAAgent:
    def __init__(self, env, gamma, alpha, epsilon):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # Number of states = number of unique board configurations
        self.num_states = 3**(env.board_size * env.board_size)  # Each cell has 3 states (-1, 0, 1)
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy = np.zeros(self.num_states, dtype=int)

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def train(self, episodes):
        for episode in range(episodes):
            state = self.encode_state(self.env.reset()[0])  # Encode the initial state
            action = self.epsilon_greedy(state)
            done = False
            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.encode_state(next_state)  # Encode the next state
                next_action = self.epsilon_greedy(next_state)
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
                state, action = next_state, next_action
            self.update_policy()

    def update_policy(self):
        for state in range(self.num_states):
            self.policy[state] = np.argmax(self.Q[state])

    def decode_state(self, state):
        return np.unravel_index(state, (self.env.board_size, self.env.board_size))

    def encode_state(self, board):
        # Encode the board as a single integer index based on board values
        # The values of board can be -1, 0, 1. We treat them as base-3 digits.
        return np.sum(board * (3 ** np.arange(board.size).reshape(board.shape[::-1]).flatten()))
