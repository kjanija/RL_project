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

from src.strategy import get_order_winning_move


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
        self.reward_3_in_a_row = config.get("reward_3_in_a_row", 1)
        self.reward_4_in_a_row = config.get("reward_4_in_a_row", 5)
        self.teacher_reward = config.get("teacher_reward", 2)

        # 6x6 board with either X or O (36*2)
        self.action_space = gymnasium.spaces.Discrete(72)

        # 0: empty
        # 1: X
        # 2: O
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(6, 6), dtype=np.int8
        )
        self.board = np.zeros((6, 6), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((6, 6), dtype=np.int8)
        return self.board.copy(), {}

    def action_masks(self):
        """
        Returns a boolean array of 72 elements indincating legal moves.
        True for empty squares, False for occupied
        """
        mask = np.zeros(72, dtype=bool)
        empty_mask = (self.board == 0).flatten()
        mask[0::2] = empty_mask  # X moves (even indices)
        mask[1::2] = empty_mask  # O moves (odd indices)
        return mask

    def step(self, action):

        # action decoding
        position = action // 2  # i.e. 0-35
        row = position // 6  # i.e. 0-5
        col = position % 6  # i.e. 0-5
        symbol = (action % 2) + 1  # 1 for X, 2 for O

        # 1. check move legality
        if self.board[row, col] != 0:
            # what we return is:
            #   the board,
            #   the reward,
            #   whether the game is done,
            #   whether it's truncated,
            #   and an info dictionary
            return (
                self.board.copy(),
                self.illegal_move_reward,
                True,
                False,
                {"error": "Illegal move: position already occupied."},
            )

        # 2. Ask teacher for staretegy before placing symbol
        teacher_move = self._get_teacher_move()

        # 3. apply order's move
        self.board[row, col] = symbol

        # 4.check win condition for order
        if self._check_five_in_a_row():
            return (
                self.board.copy(),
                self.win_reward,
                True,
                False,
                {"result": "Order wins!"},
            )

        # 5. check win condition for chaos (i.e. board full)
        if self._is_board_full():
            return (
                self.board.copy(),
                self.loss_reward,
                True,
                False,
                {"result": "Chaos wins!"},
            )

        # 6. intermediate reward (line extensions and matching teacher's strategy)
        reward = self.step_reward + self._compute_intermediate_rewards(
            row, col, symbol, teacher_move=teacher_move
        )

        # 7. Chaos' turn
        chaos_row, chaos_col, chaos_symbol = self._get_chaos_move()
        self.board[chaos_row, chaos_col] = chaos_symbol

        # 8. Check if Chaos accidentaly created a 5-in-a-row (win for Order)
        if self._check_five_in_a_row():
            return (
                self.board.copy(),
                self.win_reward,
                True,
                False,
                {"result": "Order wins (Chaos completed 5-in-a-row)!"},
            )

        # 9. check if chaos' move filled board
        if self._is_board_full():
            return (
                self.board.copy(),
                self.loss_reward,
                True,
                False,
                {"result": "Chaos wins!"},
            )

        return self.board.copy(), reward, False, False, {}

    ##########
    # Helper methods

    def _is_board_full(self):
        return np.all(self.board != 0)

    def _check_five_in_a_row(self):
        # Optimized the previous implementation with the help of LLMs

        # assumption: 6x6 board, indices in mask don't scale with board size
        for sym in (1, 2):
            mask = self.board == sym

            # horizontal
            if (
                mask[:, :-4]
                & mask[:, 1:-3]
                & mask[:, 2:-2]
                & mask[:, 3:-1]
                & mask[:, 4:]
            ).any():
                return True
            # vertical
            if (
                mask[:-4, :]
                & mask[1:-3, :]
                & mask[2:-2, :]
                & mask[3:-1, :]
                & mask[4:, :]
            ).any():
                return True
            # diagonal \
            if (
                mask[:-4, :-4]
                & mask[1:-3, 1:-3]
                & mask[2:-2, 2:-2]
                & mask[3:-1, 3:-1]
                & mask[4:, 4:]
            ).any():
                return True
            # diagonal /
            if (
                mask[:-4, 4:]
                & mask[1:-3, 3:-1]
                & mask[2:-2, 2:-2]
                & mask[3:-1, 1:-3]
                & mask[4:, :-4]
            ).any():
                return True

        return False

    def _compute_intermediate_rewards(self, row, col, symbol, teacher_move=None):
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
            ((1, 1), (-1, -1)),  # diagonal \
            ((1, -1), (-1, 1)),  # diagonal /
        ]

        max_line_length = 0
        for (dr1, dc1), (dr2, dc2) in axes:
            # length = 1 (the current piece) + count in one direction + count in the opposite direction
            length = 1 + count_consecutive(dr1, dc1) + count_consecutive(dr2, dc2)
            if length > max_line_length:
                max_line_length = length

        # award points based on longest line
        if max_line_length == 3:
            reward += self.reward_3_in_a_row
        elif max_line_length == 4:
            reward += self.reward_4_in_a_row

        # teacher strategy
        if teacher_move is not None:
            opt_r, opt_c, opt_symbol = teacher_move
            if row == opt_r and col == opt_c and symbol == opt_symbol:
                reward += self.teacher_reward

        return reward

    def _get_teacher_move(self):
        try:
            return get_order_winning_move(self.board)
        except Exception:
            return None

    def _get_chaos_move(self):
        """
        generates Chaos' move based on conf "chaos_mode"
            - 'random': uniform random move across empty cells
            - 'heuristic': defensive blocker prioritizing 4-in-a-row blocks and fork
        """

        mode = self.config.get("chaos_mode", "random")
        if mode == "heuristic":
            return self._get_heuristic_chaos_move()

        empty_cells = np.argwhere(self.board == 0)
        idx = self.np_random.integers(0, len(empty_cells))
        r, c = empty_cells[idx]
        symbol = self.np_random.integers(1, 3)
        return int(r), int(c), int(symbol)

    def _get_heuristic_chaos_move(self):
        """
        Defensive chaos opponent:
        - blocks immediate order 4-in-a-row wins
        - spoils 3-in-a-row open lines
        - random fallback
        """
        from src.strategy import ALL_6X6_LINES

        # 1.
        for line in ALL_6X6_LINES:
            vals = [self.board[r, c] for r, c in line]
            for sym in (1, 2):
                if vals.count(sym) == 4 and vals.count(0) == 1:
                    empty_idx = vals.index(0)
                    r, c = line[empty_idx]
                    return int(r), int(c), int(3 - sym)

        # 2.
        threat_candidates = []
        for line in ALL_6X6_LINES:
            vals = [self.board[r, c] for r, c in line]
            for sym in (1, 2):
                if vals.count(sym) == 3 and vals.count(0) == 2:
                    for idx, (r, c) in enumerate(line):
                        if vals[idx] == 0:
                            threat_candidates.append((int(r), int(c), int(3 - sym)))

        if threat_candidates:
            ifx = self.np_random.integers(0, len(threat_candidates))
            return threat_candidates[idx]

        # 3.
        empty_cells = np.argwhere(self.board == 0)
        idx = self.np_random.integers(0, len(empty_cells))
        r, c = empty_cells[idx]
        symbol = self.np_random.integers(1, 3)
        return int(r), int(c), int(symbol)
