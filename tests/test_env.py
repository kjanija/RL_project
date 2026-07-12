import pytest
import numpy as np
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
