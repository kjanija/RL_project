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
        self.config = config

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
        """
        We give rewards for extending lines and following teacher's strategy
        """
        reward = 0

        # line extension rewards
        def count_consecutive(dr, dc):
            count = 0
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 6 and self.board[r, c] == symbol:
                count += 1
                r += dr
                c += dc
            return count

        # check all 4 axes:
        axes = [
            ((0, 1), (0, -1)),  # horizontal
            ((1, 0), (-1, 0)),  # vertical
            ((1, 1), (-1, -1)), # diagonal \
            ((1, -1), (-1, 1))  # diagonal /
        ]

        max_line_length = 0
        for (dr1, dc1), (dr2, dc2) in axes:
            # length = 1 (the current piece) + count in one direction + count in the opposite direction
            length = 1 + count_consecutive(dr1, dc1) + count_consecutive(dr2, dc2)
            if length > max_line_length:
                max_line_length = length

        # award points based on longest line
        if max_line_length == 3:
            reward += self.config.get("line_of_3_reward", 1)
        elif max_line_length == 4:
            reward += self.config.get("line_of_4_reward", 5)

        # teacher strategy
        # will put in a try catch because at the moment of writing I haven't
        # implemented the teacher strategy
        try:
            opt_r, opt_c, opt_symbol = self._get_teacher_move()
            if r == opt_r and c == opt_c and symbol == opt_symbol:
                reward += self.config.get("teacher_move_reward", 2)
        except NotImplementedError:
            pass

        return reward

    def _get_teacher_move(self):
        # TODO: when src/strategy.py is done, uncomment the following lines
        # from src.strategy import get_teacher_move
        # return get_order_winning_move(self.board)

        raise NotImplementedError("Teacher strategy not implemented yet.")

    def _get_chaos_move(self):
        """
        Temporarily a random move, in the future it will be derived in a more advanced way
        """

        empty_cells = np.argwhere(self.board == 0)
        idx = np.random.choice(len(empty_cells))
        r, c = empty_cells[idx]
        symbol = np.random.choice([1, 2])  # Randomly choose X or O
        return r, c, symbol