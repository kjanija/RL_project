"""
src/env.py

This file will contain the custom Gymnasium environment for the Order and Chaos game.

Expected Implementation:
- A class `OrderAndChaosEnv` inheriting from `gymnasium.Env`.
- `__init__`: with `action_space` (e.g., Discrete(72)) and `observation_space` (e.g., 6x6 Box), and initialize the board.
- `reset`: Clear the board and reset the state for a new episode.
- `step`: Apply the given action, calculate the intermediate/final rewards, check for win/loss, and generate the opponent's move.
- Helper methods for calculating rewards and checking board states (lines of 5).
"""

import gymnasium
import numpy as np

class OrderAndChaosEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        # configurable rewards
        self.illegal_move_reward = config.get("illegal_move_reward", -100)
        self.win_reward = config.get("win_reward", 100)
        self.loss_reward = config.get("loss_reward", -100)
        self.step_reward = config.get("step_reward", 0)

        # 6x6 board with either X or O
        self.action_space = gymnasium.spaces.Discrete(72)

        # 0: empty
        # 1: X
        # 2: O
        self.observation_space = gymnasium.spaces.Box(low=0, high=2, shape=(6, 6), dtype=np.int8)
        self.board = np.zeros((6,6), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((6,6), dtype=np.int8)
        return self.board.copy(), {}

    def step(self, action):

        # action decoding
        position = action // 2  # i.e. 0-35
        row = position // 6     # i.e. 0-5
        col = position % 6      # i.e. 0-5
        symbol = (action % 2) +1 # 1 for X, 2 for O

        # check move legality
        if self.board[row, col] != 0:
            # what we return is: the board, the reward, whether the game is done, whether it's truncated, and an info dictionary
            return self.board.copy(), self.illegal_move_reward, True, False, {"error": "Illegal move: position already occupied."}

        self.board[row, col] = symbol

        # check win condition for order
        if self._check_five_in_a_row():
            return self.board.copy(), self.win_reward, True, False, {"result": "Order wins!"}

        # check win condition for chaos
        if self._is_board_full():
            return self.board.copy(), self.loss_reward, True, False, {"result": "Chaos wins!"}

        # intermediate reward
        reward = self.step_reward + self._compute_intermediate_rewards(row, col, symbol)

        # Chaos' turn
        chaos_row, chaos_col, chaos_symbol = self._get_chaos_move()
        self.board[chaos_row, chaos_col] = chaos_symbol

        # check if chaos' move filled board
        if self._is_board_full():
            return self.board.copy(), self.loss_reward, True, False, {"result": "Chaos wins!"}

        return self.board.copy(), reward, False, False, {}

    ##########
    # Helper methods

    def _is_board_full(self):
        return np.all(self.board != 0)

    def _check_five_in_a_row(self):
        # Optimized the previous implementation with the help of LLMs

        # assumption: 6x6 board, indices in mask don't scale with board size
        for sym in (1, 2):
            mask = (self.board == sym)

            #horizontal
            if (mask[:, :-4] & mask[:, 1:-3] & mask[:, 2:-2] & mask[:, 3:-1] & mask[:, 4:]).any():
                return True
            # vertical
            if (mask[:-4, :] & mask[1:-3, :] & mask[2:-2, :] & mask[3:-1, :] & mask[4:, :]).any():
                return True
            # diagonal \
            if (mask[:-4, :-4] & mask[1:-3, 1:-3] & mask[2:-2, 2:-2] & mask[3:-1, 3:-1] & mask[4:, 4:]).any():
                return True
            # diagonal /
            if(mask[:-4, 4:] & mask[1:-3, 3:-1] & mask[2:-2, 2:-2] & mask[3:-1, 1:-3] & mask[4:, :-4]).any():
                return True
        
        return False

    def _compute_intermediate_rewards(self, row, col, symbol):
        pass

    def _get_chaos_move(self):
        """
        Temporarily a random move, in the future it will be derived in a more advanced way
        """

        empty_cells = np.argwhere(self.board == 0)
        idx = np.random.choice(len(empty_cells))
        r, c = empty_cells[idx]
        symbol = np.random.choice([1, 2])  # Randomly choose X or O
        return r, c, symbol