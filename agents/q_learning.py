import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.min_epsilon = min_epsilon  # Minimum value for epsilon
        self.num_states = env.observation_space.n  # Total number of states
        self.num_actions = env.action_space.n  # Total number of actions
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((self.num_states, self.num_actions))

    def epsilon_greedy(self, state):
        """Select action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: Choose the action with max Q-value
            return np.argmax(self.Q[state])

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using the Q-learning update rule."""
        if done:
            # If the game is over, there is no future state
            target = reward
        else:
            # Otherwise, use the Q-value of the next state (Bellman equation)
            target = reward + self.gamma * np.max(self.Q[next_state])

        # Update Q-table (Q(s, a) ← Q(s, a) + α * (target - Q(s, a)))
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, episodes):
        """Train the agent using Q-learning."""
        for episode in range(episodes):
            state = self.env.reset()[0]  # Start a new game
            done = False
            total_reward = 0

            while not done:
                action = self.epsilon_greedy(state)  # Choose an action
                next_state, reward, done, _, info = self.env.step(action)  # Take action and get next state
                next_state = self.encode_state(next_state)  # Encode the next state to 1D index

                self.update_q_value(state, action, reward, next_state, done)  # Update Q-value
                state = next_state  # Transition to the next state

                total_reward += reward

            # Decay epsilon after each episode for less exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

    def encode_state(self, board):
        """Encode the board state to a 1D index."""
        return np.ravel(board).argmax()

    def decode_state(self, state):
        """Decode the 1D state index back to a board."""
        return np.unravel_index(state, (self.env.board_size, self.env.board_size))
