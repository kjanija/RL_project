import numpy as np
import pytest

from src.env import OrderAndChaosEnv


def test_empty_board():
    env = OrderAndChaosEnv()
    env.reset()
    assert not env._check_five_in_a_row(), "Empty board should not trigger a win."


def test_horizontal_win():
    env = OrderAndChaosEnv()
    env.reset()
    # 5 'X's (1) in a row on the top row, starting at column 0
    env.board[0, 0:5] = 1
    assert (
        env._check_five_in_a_row()
    ), "Failed to detect horizontal X win starting at column 0."

    env.reset()
    # 5 'O's (2) in a row on the bottom row, starting at column 1
    env.board[5, 1:6] = 2
    assert (
        env._check_five_in_a_row()
    ), "Failed to detect horizontal O win starting at column 1."


def test_vertical_win():
    env = OrderAndChaosEnv()
    env.reset()
    # 5 'X's in column 2, starting at row 1
    env.board[1:6, 2] = 1
    assert env._check_five_in_a_row(), "Failed to detect vertical win."


def test_diagonal_win():
    env = OrderAndChaosEnv()
    env.reset()
    # Top-left to bottom-right (starting at 0,0)
    for i in range(5):
        env.board[i, i] = 1
    assert (
        env._check_five_in_a_row()
    ), "Failed to detect main diagonal win starting at (0,0)."

    env.reset()
    # Top-left to bottom-right (starting at 1,1)
    for i in range(1, 6):
        env.board[i, i] = 2
    assert (
        env._check_five_in_a_row()
    ), "Failed to detect main diagonal win starting at (1,1)."


def test_anti_diagonal_win():
    env = OrderAndChaosEnv()
    env.reset()
    # Top-right to bottom-left (starting at 0,5)
    for i in range(5):
        env.board[i, 5 - i] = 1
    assert env._check_five_in_a_row(), "Failed to detect anti-diagonal win."


def test_four_in_a_row_fails():
    env = OrderAndChaosEnv()
    env.reset()
    # Exactly 4 in a row, should be False
    env.board[0, 0:4] = 1
    assert not env._check_five_in_a_row(), "Incorrectly flagged 4-in-a-row as a win."


def test_mixed_symbols_fails():
    env = OrderAndChaosEnv()
    env.reset()
    # 5 pieces in a row but mixed X and O
    env.board[0, 0:3] = 1
    env.board[0, 3:5] = 2
    assert (
        not env._check_five_in_a_row()
    ), "Incorrectly flagged a mixed line of Xs and Os as a win."


def test_intermediate_reward_three_in_a_row():
    # Setup env with specific config to guarantee exact expected outputs
    config = {"reward_3_in_a_row": 1, "reward_4_in_a_row": 5}
    env = OrderAndChaosEnv(config=config)
    env.reset()

    # Pre-place two 'X's (symbol 1) horizontally
    env.board[0, 0] = 1
    env.board[0, 1] = 1

    # Simulate placing the third 'X' at (0, 2)
    # Note: we MUST physically place the piece on the board BEFORE calling the method,
    # because count_consecutive actively scans the board state.
    env.board[0, 2] = 1
    reward = env._compute_intermediate_rewards(0, 2, 1)

    assert reward == 1, f"Expected reward 1 for 3-in-a-row, got {reward}"


def test_intermediate_reward_four_in_a_row_diagonal():
    config = {"reward_3_in_a_row": 1, "reward_4_in_a_row": 5}
    env = OrderAndChaosEnv(config=config)
    env.reset()

    # Pre-place three 'O's (symbol 2) on the main diagonal
    env.board[1, 1] = 2
    env.board[2, 2] = 2
    env.board[3, 3] = 2

    # Simulate placing the fourth 'O' at (4, 4)
    env.board[4, 4] = 2
    reward = env._compute_intermediate_rewards(4, 4, 2)

    assert reward == 5, f"Expected reward 5 for 4-in-a-row on diagonal, got {reward}"


def test_intermediate_reward_mixed_symbols_blocked():
    config = {"reward_3_in_a_row": 1, "reward_4_in_a_row": 5}
    env = OrderAndChaosEnv(config=config)
    env.reset()

    # The agent plays 'X' at (0,0), then Chaos plays 'O' at (0,1).
    env.board[0, 0] = 1
    env.board[0, 1] = 2

    # The agent attempts to play 'X' at (0,2).
    # The sequence is [X, O, X]. The unbroken line of 'X's extending from (0,2) is only length 1.
    env.board[0, 2] = 1
    reward = env._compute_intermediate_rewards(0, 2, 1)

    assert (
        reward == 0
    ), "Expected 0 reward. The line was broken by an opponent symbol, but the env rewarded it anyway."
