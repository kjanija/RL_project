"""
src/strategy.py

This file will contain the implementation of the theoretical winning strategy for Order.

Expected Implementation:
- Functions or a class to evaluate a given 6x6 board state.
- Logic to map the board to a 4x4 inner center and an outer border.
- The 6 heuristics for playing in the 4x4 center.
- The mirroring strategy for the outer border.
- These functions will be used by the environment to generate opponent moves or to compute "teacher" rewards.
"""

import numpy as np

def get_order_winning_move(board):
    """
    Returns optimal move (row, col, symbol) for the Order player based on the
    theoretical winning strategy from the 'Order versus chaos' paper

    board: 6x6 np array (0:empty, 1:X, 2:O)
    """

    #1. Border Mirror Response
    # If chaos plays in outer border, order must play opposite symbol in the
    # opposite position, where opposite of (r, c) is (5-r,5-c)
    border_indices = [
        (r, c) for r in range(6) for c in range(6)
        if r==0 or r==5 or c==0 or c==5
    ]

    for r, c in border_indices:
        if board[r, c] != 0:
            opp_r, opp_c = 5-r, 5-c
            if board[opp_r, opp_c] == 0:
                # chaos played here, mirror cell empty
                # mirror by playing opposite sym
                opp_sym = 3 - board[r, c] # 1-->2, 2-->1
                return (opp_r, opp_c, opp_sym)

    # 2. Center
    inner_board = board[1:5, 1:5]

    if np.all(inner_board != 0):
        # If the inner board is full, Rule 3: order is guaranteed to have a
        # 4-in-a-row inside, just extend it to the border.
        return _find_extension_move(board)

    # Rule 2: map inner coords to outer coords
    inner_r, inner_c, symbol = _inner_stretegy(inner_board)
    return (inner_r + 1, inner_c + 1, symbol)

def _find_extension_move(board):
    """
    finds a 4-in-a-row in the inner board and completes it
    """
    # TODO
    pass

def _inner_stretegy(inner_board):
    """
    Implements the 6 heuristics for playing in the 4x4 center
    Returns (row, col, symbol) in inner board coordinates
    """
    # TODO
    pass