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
    inner_board = board[1:5, 1:5]
    for coords, vals in _get_4x4_lines(inner_board):
        if np.all(vals == 1) or np.all(vals == 2):
            symbol = vals[0]

            # map start and end coords to outer board
            r1, c1 = coords[0][0] + 1, coords[0][1] + 1
            r2, c2 = coords[3][0] + 1, coords[3][1] + 1

            # direction
            dr = np.sign(r2 - r1)
            dc = np.sign(c2 - c1)

            # border squares extending this line are:
            border1 = (r1 - dr, c1 - dc)
            border2 = (r2 + dr, c2 + dc)

            # play in which one is empty
            if board[border1] == 0:
                return (border1[0], border1[1], symbol)
            elif board[border2] == 0:
                return (border2[0], border2[1], symbol)
    
    # fallback (technically never reached)
    print("Warning: no 4-in-a-row found in inner board, but inner board is full")
    empty_spots = np.argwhere(board == 0)
    return (empty_spots[0][0], empty_spots[0][1], 1)

def _inner_stretegy(inner_board):
    """
    Implements the 6 heuristics for playing in the 4x4 center
    Returns (row, col, symbol) in inner board coordinates
    """
    # Rule 1
    if np.all(inner_board == 0):
        return (1, 1, 1) # play X at (1,1) of inner, which is (2,2) of whole board

    # Rule 2
    move = _find_completion_move(inner_board, target_weight=3)
    if move is not None:
        return move

    lines = _get_4x4_lines(inner_board)
    empty_spots = np.argwhere(inner_board == 0)

    # Rule 3
    # line intersections: find empty square that intersects TWO unbroken lines
    # of weight 2 with SAME symbol
    for r, c in empty_spots:
        lines_x = 0
        lines_o = 0
        for coords, vals in lines:
            if (r,c) in coords:
                has_X = (1 in vals)
                has_O = (2 in vals)

                # line of X weight 2
                if has_X and not has_O and np.sum(vals == 1) == 2:
                    lines_x += 1

                # line of O weight 2
                elif has_O and not has_X and np.sum(vals == 2) == 2:
                    lines_o += 1

        if lines_x >= 2: return (r, c, 1)
        if lines_o >= 2: return (r, c, 2)

    # Rule 4
    # 1. find global max_weight `w_star` among all unbroken lines
    w_star = 0
    unbroken_lines = []
    for coords, vals in lines:
        has_X = (1 in vals)
        has_O = (2 in vals)
        if not (has_X and has_O):
            weight = np.sum(vals != 0)
            unbroken_lines.append((coords, vals, has_X, has_O, weight))
            if weight > w_star:
                w_star = weight

    # 2. score empty spots for both symbols
    best_move = None
    best_score = -1

    for r,c in empty_spots:
        for sym in (1,2):
            breaks_max_opponent_line = False

            for coords, vals, has_X, has_O, weight in unbroken_lines:
                if (r,c) in coords:
                    opp_sym = 3-sym # if we play 1, opponent is 2

                    # if this is the opponent's line and has max weight, cannot
                    # break it
                    if (opp_sym == 1 and has_X) or (opp_sym == 2 and has_O):
                        if weight == w_star:
                            breaks_max_opponent_line = True
                            break
            if breaks_max_opponent_line:
                continue # skip playing sym at (r, c)

            # otherwise, calc max weight of OUR lines passing thorugh (r,c)
            max_resulting_weight = 0
            for coords, vals, has_X, has_O, weight in unbroken_lines:
                if (r,c) in coords:
                    # if its our line or a completely empty one
                    if (sym == 1 and not has_O) or (sym == 2 and not has_X):
                        if weight + 1 > max_resulting_weight:
                            max_resulting_weight = weight + 1

            if max_resulting_weight > best_score:
                best_score = max_resulting_weight
                best_move = (r, c, sym)

    if best_move is not None:
        return best_move

    # Fallback: first available empty space, technically never reached
    r, c = empty_spots[0]
    return (r, c, 1)

def _find_completion_move(inner_board, target_weight):
    """
    scan inner board for unbroken line of length `target_weight`

    returns (row, col, sym) to complete it
    """
    
    for coords, vals in _get_4x4_lines(inner_board):
        has_X = (1 in vals)
        has_O = (2 in vals)

        if has_X and has_O:
            continue

        weight = np.sum(vals != 0)
        if weight == target_weight:
            symbol = 1 if not (has_X or has_O) else (1 if has_X else 2)

            empty_idx = np.argwhere(vals == 0)[0][0]
            r, c = coords[empty_idx]

            return (r, c, symbol)

    return None

def _get_4x4_lines(inner_board):
    """
    Helper that returns 10 lines of length 4 of th einner board

    Returns a list of tuples like (list of coords, np array of values)
    """

    lines = []
    # rows
    for r in range(4):
        lines.append(([(r, c) for c in range(4)], inner_board[r, :]))
    # cols
    for c in range(4):
        lines.append(([(r, c) for r in range(4)], inner_board[:, c]))
    # diagonals
    lines.append(([(i, i) for i in range(4)], np.diag(inner_board)))
    lines.append(([(i, 3-i) for i in range(4)], np.diag(np.fliplr(inner_board))))

    return lines