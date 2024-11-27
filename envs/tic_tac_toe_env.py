import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import product

class TicTacToeEnv(gym.Env):
    def __init__(self, board_size=3):
        self.board_size = board_size

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int32
        )
        self.action_space = spaces.Discrete(board_size * board_size)

        self.board = None
        self.done = False

    def _get_obs(self):
        return self.board

    def _get_info(self):
        return {"empty_cells": len(self._get_empty_cells())}

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        return self.board, self._get_info()

    def _check_winner(self, player):
        # Check rows and columns
        for i in range(self.board_size):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        # Check diagonals
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def _get_empty_cells(self):
        return [(r, c) for r, c in product(range(self.board_size), repeat=2) if self.board[r, c] == 0]

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() the environment before taking further steps.")

        row, col = divmod(action, self.board_size)
        reward = 0
        terminated = False
        truncated = False

        # Check if action is valid
        if self.board[row, col] != 0:
            reward = -10
            terminated = True
            return self.board, reward, terminated, truncated, {"reason": "Invalid action"}

        # Player's move
        self.board[row, col] = 1
        if self._check_winner(1):
            reward = 1
            terminated = True
            return self.board, reward, terminated, truncated, {"winner": "Agents"}

        # Check for draw
        if np.all(self.board != 0):
            reward = 0
            terminated = True
            return self.board, reward, terminated, truncated, {"winner": "Hòa"}

        # Opponent's move
        empty_cells = self._get_empty_cells()
        opp_row, opp_col = empty_cells[np.random.choice(len(empty_cells))]
        self.board[opp_row, opp_col] = -1
        if self._check_winner(-1):
            reward = -1
            terminated = True
            return self.board, reward, terminated, truncated, {"winner": "Opponents"}

        return self.board, reward, terminated, truncated, {}
