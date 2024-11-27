import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = np.zeros(self.num_states, dtype=int)
        self.state_value = np.zeros(self.num_states)

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.num_states):
                v = self.state_value[state]
                action = self.policy[state]
                next_state, reward, done, _, _ = self.env.step(action)
                self.state_value[state] = reward + self.gamma * self.state_value[next_state]
                delta = max(delta, abs(v - self.state_value[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.num_states):
            old_action = self.policy[state]
            action_values = []
            for action in range(self.num_actions):
                next_state, reward, done, _, _ = self.env.step(action)
                action_values.append(reward + self.gamma * self.state_value[next_state])
            self.policy[state] = np.argmax(action_values)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def train(self, episodes):
        for episode in range(episodes):
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def decode_state(self, state):
        return np.unravel_index(state, (self.env.board_size, self.env.board_size))

    def encode_state(self, board):
        return np.ravel(board).argmax()
