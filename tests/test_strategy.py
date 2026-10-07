import numpy as np
import pytest
from src.env import OrderAndChaosEnv
from src.strategy import (
    get_order_winning_move,
    find_immediate_5_in_a_row_win,
    BORDER_PAIRS,
)

try:
    from src.strategy import check_figure4_exception
except ImportError:
    from src.strategy import check_figure_4_exception as check_figure4_exception


def test_first_move_is_center():
    board = np.zeros((6, 6), dtype=int)
    r, c, sym = get_order_winning_move(board)
    assert (r, c) == (2, 2)
    assert sym == 1


def test_immediate_5_in_a_row_completion():
    board = np.zeros((6, 6), dtype=int)
    board[0, 0:4] = 1  # 4 X's on top row
    r, c, sym = get_order_winning_move(board)
    assert (r, c, sym) == (0, 4, 1)


def test_border_mirror_row_pairs():
    # Test that row pair (1, 0) <-> (1, 5) pairs horizontally, NOT across center
    board = np.zeros((6, 6), dtype=int)
    board[2, 2] = 1
    board[1, 0] = 2  # Chaos plays O at (1, 0)
    r, c, sym = get_order_winning_move(board)
    assert (r, c) == (1, 5), f"Expected (1, 5), got ({r}, {c})"
    assert sym == 1, f"Expected opposite symbol 1, got {sym}"


def test_border_mirror_col_pairs():
    # Test that col pair (0, 2) <-> (5, 2) pairs vertically, NOT across center
    board = np.zeros((6, 6), dtype=int)
    board[2, 2] = 1
    board[0, 2] = 1  # Chaos plays X at (0, 2)
    r, c, sym = get_order_winning_move(board)
    assert (r, c) == (5, 2), f"Expected (5, 2), got ({r}, {c})"
    assert sym == 2, f"Expected opposite symbol 2, got {sym}"


def test_figure4_board1_exception():
    # Canonical Board 1: O at (0,0), X at (1,0),(1,1),(1,3),(2,2), O at (1,2),(2,0),(3,0)
    b = np.zeros((4, 4), dtype=int)
    b[0, 0] = 2
    b[1, 0] = 1
    b[1, 1] = 1
    b[1, 2] = 2
    b[1, 3] = 1
    b[2, 0] = 2
    b[2, 2] = 1
    b[3, 0] = 2
    exc = check_figure4_exception(b)
    assert exc == (0, 3, 2), f"Expected play O at (0, 3), got {exc}"


def _has_five(board):
    env = OrderAndChaosEnv()
    env.board = board.astype(np.int8)
    return env._check_five_in_a_row()


def test_simulation_100_percent_win_against_random():
    rng = np.random.default_rng(123)
    for _ in range(50):
        board = np.zeros((6, 6), dtype=int)
        while True:
            r, c, s = get_order_winning_move(board)
            board[r, c] = s
            if _has_five(board):
                break
            if np.all(board != 0):
                pytest.fail("Order failed to win against random Chaos!")
            empty = np.argwhere(board == 0)
            idx = rng.integers(0, len(empty))
            cr, cc = empty[idx]
            board[cr, cc] = rng.integers(1, 3)
            if _has_five(board):
                break  # Chaos completed a line: Order wins by the rules
