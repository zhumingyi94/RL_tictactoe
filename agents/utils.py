import numpy as np

def encode_state(state):
    return int("".join(map(str, (state.flatten() + 1).tolist())), 3)

def decode_state(state_int, board_size):
    base3_str = np.base_repr(state_int, base=3).zfill(board_size * board_size)
    return np.array(list(map(int, base3_str)), dtype=int).reshape(board_size, board_size) - 1