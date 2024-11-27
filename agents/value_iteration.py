import numpy as np

class ValueIterAgent:
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.num_states = env.observation_space.n  # Use the total number of states directly
        self.num_actions = env.action_space.n  # Total number of actions

        # Initialize value function and policy
        self.state_value = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def value_iteration(self):
        """Perform value iteration."""
        while True:
            delta = 0
            for state in range(self.num_states):
                v = self.state_value[state]
                action_values = []

                # Try all possible actions (moves)
                for action in range(self.num_actions):
                    # Simulate the environment step for this action
                    board = self.decode_state(state)
                    next_board, reward, done, _, _ = self.env.step(action)

                    # Flatten the next state (board) and calculate its value
                    next_state = self.encode_state(next_board)
                    action_values.append(reward + self.gamma * self.state_value[next_state])

                # Update state value to the maximum action value
                self.state_value[state] = max(action_values)

                # Track the largest change in value function
                delta = max(delta, abs(v - self.state_value[state]))

            # Check for convergence
            if delta < self.theta:
                break

        # After value iteration is complete, update the policy
        self.update_policy()

    def update_policy(self):
        """Update the policy to choose the best action."""
        for state in range(self.num_states):
            action_values = []
            for action in range(self.num_actions):
                board = self.decode_state(state)
                next_board, reward, done, _, _ = self.env.step(action)
                next_state = self.encode_state(next_board)
                action_values.append(reward + self.gamma * self.state_value[next_state])
            self.policy[state] = np.argmax(action_values)

    def encode_state(self, board):
        """Encode the board state to a 1D index."""
        return np.ravel(board).argmax()

    def decode_state(self, state):
        """Decode the 1D state index back to a board."""
        return np.unravel_index(state, (self.env.board_size, self.env.board_size))
