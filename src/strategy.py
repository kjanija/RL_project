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

from symtable import Symbol

import numpy as np

# 10 border pairs from Lemma 4.4, Figure 5
# each pair connects two opposing endpoints of the lines of length 6 passing through B1
BORDER_PAIRS = {
    # 4 row pairs (r,0)<->(r,5) for r in [1,4]
    (1, 0): (1, 5),
    (1, 5): (1, 0),
    (2, 0): (2, 5),
    (2, 5): (2, 0),
    (3, 0): (3, 5),
    (3, 5): (3, 0),
    (4, 0): (4, 5),
    (4, 5): (4, 0),
    # 4 col pairs (0,c)<->(5,c) for c in [1,4]
    (0, 1): (5, 1),
    (5, 1): (0, 1),
    (0, 2): (5, 2),
    (5, 2): (0, 2),
    (0, 3): (5, 3),
    (5, 3): (0, 3),
    (0, 4): (5, 4),
    (5, 4): (0, 4),
    # 2 diagonal pairs
    (0, 0): (5, 5),
    (5, 5): (0, 0),
    (0, 5): (5, 0),
    (5, 0): (0, 5),
}

# 6 canonical exception boards from Figura 4
# they are represented as 4x4 arrays 0=empty, 1=X, 2=O with target move (r,c,sym)
FIG4_EXCEPTIONS = []


def _init_exceptions():
    # top-left board from Figure 4:
    b1 = np.zeros((4, 4), dtype=int)
    b1[0, 0] = 2
    b1[1, 0] = 1
    b1[1, 1] = 1
    b1[1, 2] = 2
    b1[1, 3] = 1
    b1[2, 0] = 2
    b1[2, 2] = 1
    b1[3, 0] = 2
    FIG4_EXCEPTIONS.append((b1, 0, 3, 2))  # i.e. play O at (0,3)

    # top-right board from figure 4:
    b2 = np.zeros((4, 4), dtype=int)
    b2[0, 1] = 2
    b2[1, 0] = 1
    b2[1, 1] = 1
    b2[1, 2] = 1
    b2[1, 3] = 2
    b2[3, 3] = 2
    FIG4_EXCEPTIONS.append((b2, 3, 1, 2))  # i.e. play O at (3,1)

    # middle-left board of Figure 4:
    b3 = np.zeros((4, 4), dtype=int)
    b3[1, 0] = 2
    b3[1, 1] = 1
    b3[1, 2] = 1
    b3[2, 0] = 2
    FIG4_EXCEPTIONS.append((b3, 0, 3, 1))  # Play X at (0,3)

    # middle-right of Figure 4:
    b4 = np.zeros((4, 4), dtype=int)
    b4[0, 0] = 2
    b4[0, 1] = 2
    b4[1, 0] = 1
    b4[1, 1] = 1
    FIG4_EXCEPTIONS.append((b4, 0, 3, 2))  # Play O at (0,3)

    # bottom-left of Figure 4:
    b5 = np.zeros((4, 4), dtype=int)
    b5[0, 0] = 1
    b5[0, 2] = 2
    b5[1, 0] = 2
    b5[1, 1] = 1
    b5[2, 0] = 2
    b5[3, 0] = 2
    FIG4_EXCEPTIONS.append((b5, 3, 3, 1))  # Play X at (3,3)

    # bottom-right of Figure 4:
    b6 = np.zeros((4, 4), dtype=int)
    b6[1, 0] = 2
    b6[1, 1] = 1
    b6[2, 0] = 2
    b6[3, 0] = 1
    b6[3, 1] = 1
    b6[3, 3] = 2
    FIG4_EXCEPTIONS.append((b6, 3, 2, 2))  # Play O at (3,2)


_init_exceptions()


def _d4_transforms():
    """
    Returns 8 coordinate mapping functions for D4 symmetry on 4x4 grid
    """
    transforms = []
    for rot in range(4):
        for reflect in [False, True]:

            def make_tf(k, ref):
                def tf(r, c):
                    curr_r, curr_c = r, c
                    for _ in range(k):
                        curr_r, curr_c = curr_c, 3 - curr_r
                    if ref:
                        curr_c = 3 - curr_c
                    return curr_r, curr_c

                return tf

            transforms.append(make_tf(rot, reflect))
    return transforms


TRANSFORMS = _d4_transforms()


def check_figure_4_exception(inner_board):
    """
    Checks if inner_board matches any Figure 4 exception (under d4 symmetry or swap of symbol X/O)
    Returns (r,c,sym) in 4x4 inner coordinates, or None
    """

    for base_b, target_r, target_c, target_sym in FIG4_EXCEPTIONS:
        for tf in TRANSFORMS:
            transformed_b = np.zeros((4, 4), dtype=int)
            for r in range(4):
                for c in range(4):
                    if base_b(r, c) != 0:
                        tr, tc = tf(r, c)
                        transformed_b[tr, tc] = base_b[r, c]

            # direct match
            if np.array_equal(inner_board, transformed_b):
                tr, tc = tf(target_r, target_c)
                return tr, tc, target_sym

            # inverted symbols match
            inv_transformed = np.zeros_like(transformed_b)
            inv_transformed[transformed_b == 1] = 2
            inv_transformed[transformed_b == 2] = 1
            if np.array_equal(inner_board, inv_transformed):
                tr, tc = tf(target_r, target_c)
                return tr, tc, 3 - target_sym
    return None


def get_all_6x6_win_lines():
    """
    Returns all 28 winning lines of length 5 on the 6x6 board
    """
    lines = []
    for r in range(6):
        for c in range(2):
            lines.append([(r, c + i) for i in range(5)])
    for c in range(6):
        for r in range(2):
            lines.append([(r + i, c) for i in range(5)])
    for r in range(2):
        for c in range(2):
            lines.append([(r + i, c + i) for i in range(5)])
    for r in range(6):
        for c in range(4, 6):
            lines.append([(r + i, c - i) for i in range(5)])
    return lines


ALL_6X6_LINES = get_all_6x6_win_lines()


def find_immediate_5_in_a_row_win(board):
    """
    Returns (r,c,sym) if order can complete a 5-in-a-row anywhere on the 6x6 board
    """
    for line in ALL_6X6_LINES:
        vals = [board[r, c] for r, c in line]
        for sym in (1, 2):
            if vals.count(sym) == 4 and vals.count(0) == 1:
                empty_idx = vals.index(0)
                r, c = line[empty_idx]
                return r, c, sym
    return None


def _find_inner_4_in_a_row_completion(board):
    """
    Checks if a 4-in-a-row exists inside B1
    If so, extends it into the empty border square across its line
    """
    inner_board = board[1:5, 1:5]
    for coords, vals in _get_4x4_lines(inner_board):
        if (np.all(vals == 1) or np.all(vals == 2)) and vals[0] != 0:
            sym = vals[0]
            r1, c1 = coords[0][0] + 1, coords[0][1] + 1
            r2, c2 = coords[3][0] + 1, coords[3][1] + 1
            dr = np.sign(r2 - r1)
            dc = np.sign(c2 - c1)
            border1 = (r1 - dr, c1 - dc)
            border2 = (r2 + dr, c2 + dc)
            if board[border1] == 0:
                return (border1[0], border1[1], sym)
            if board[border1] == 0:
                return (border2[0], border2[1], sym)
    return None


def get_order_winning_move(board):
    """
    Returns optimal move (row, col, symbol) for the Order player based on the
    theoretical winning strategy from the 'Order versus chaos' paper

    board: 6x6 np array (0:empty, 1:X, 2:O)
    """

    # 0. immediate 5-in-a-row
    win_move = find_immediate_5_in_a_row_win(board)
    if win_move is not None:
        return win_move

    # 1. 4-in-a-row extension from inner board
    ext_move = _find_inner_4_in_a_row_completion(board)
    if ext_move is not None:
        return ext_move

    # 2. Border Mirror Response (lemma 4.4 and figure 5)
    for u, v in BORDER_PAIRS.items():
        if board[u] != 0 and board[v] == 0:
            opp_sym = 3 - board[u]
            return (v[0], v[1], opp_sym)

    # 3. Center 4x4 (strategy 4.2)
    inner_board = board[1:5, 1:5]
    if np.any(inner_board == 0):
        ir, ic, sym = _inner_stretegy(inner_board)
        return (ir + 1, ic + 1, sym)

    # fallback first available cell
    empty = np.argwhere(board == 0)
    if len(empty) > 0:
        return (int(empty[0][0]), int(empty[0][1]), 1)

    return (0, 0, 1)


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
    Implements the 6 heuristics strategy from 4.2 for playing in the 4x4 center
    Returns (row, col, symbol) in inner board coordinates
    """
    # Rule 1: exeptions from figure 4
    exc = check_figure_4_exception(inner_board)
    if exc is not None:
        return exc

    # Rule 2: first move
    if np.all(inner_board == 0):
        return (1, 1, 1)  # play X at (1,1), which is (2,2) on 6x6

    lines = _get_4x4_lines(inner_board)
    empty_spots = [(r, c) for r in range(4) for c in range(4) if inner_board[r, c] == 0]

    # Rule 3: immediate win in 4x4 (i.e. unbroken line of weigth 3)
    candidate_wins = []
    for coords, vals in lines:
        for sym in (1, 2):
            if np.sum(vals == sym) == 3 and np.sum(vals == 0) == 1:
                idx = np.where(vals == 0)[0][0]
                candidate_wins.append((coords[idx][0], coords[idx][1], sym))
    if candidate_wins:
        candidate_wins.sort(key=lambda m: (m[0], m[1]))
        return candidate_wins[0]

    # Rule 4: fork/double threat
    candidate_forks = []
    for r, c in empty_spots:
        for sym in (1, 2):
            count = 0
            for coords, vals in lines:
                if (r, c) in coords:
                    opp = 3 - sym
                    if np.all(vals != opp) and np.sum(vals == sym) == 2:
                        count += 1
            if count >= 2:
                candidate_forks.append((r, c, sym))
    if candidate_forks:
        candidate_forks.sort(key=lambda m: (m[0], m[1]))
        return candidate_forks[0]

    # Rule 5: safe extensions
    safe_moves_by_line_weight = []
    for r, c in empty_spots:
        for sym in (1, 2):
            opp = 3 - sym
            breaks_any = False
            for coords, vals in lines:
                if (r, c) in coords:
                    if np.all(vals != sym) and np.any(vals == opp):
                        breaks_any = True
                        break
            if not breaks_any:
                max_w = 0
                for coords, vals in lines:
                    if (r, c) in coords and np.all(vals != opp):
                        w = np.sum(vals == sym)
                        if w > max_w:
                            max_w = w
                safe_moves_by_line_weight.append((max_w, r, c, sym))
    if safe_moves_by_line_weight:
        safe_moves_by_line_weight.sort(key=lambda m: (-m[0], m[1], m[2]))
        best_w = safe_moves_by_line_weight[0][0]
        best_candidates = [m for m in safe_moves_by_line_weight if m[0] == best_w]
        best_candidates.sort(key=lambda m: (m[1], m[2]))
        _, r, c, sym = best_candidates[0]
        return (r, c, sym)

    # Rule 6: maximal weight extension
    w_star = 0
    unbroken_lies = []

    for coords, vals in lines:
        has_X = np.any(vals == 1)
        has_O = np.any(vals == 2)
        if not (has_X and has_O):
            w = np.sum(vals != 0)
            unbroken_lies.append((coords, vals, has_X, has_O, w))
            if w > w_star:
                w_star = w

    best_score = -1
    best_candidates_vi = []
    for r, c in empty_spots:
        for sym in (1, 2):
            opp = 3 - sym
            breaks_w_star = False
            for coords, vals, has_X, has_O, w in unbroken_lies:
                if (r, c) in coords and w == w_star:
                    if (opp == 1 and has_X) or (opp == 2 and has_O):
                        breaks_w_star = True
                        break
            if breaks_w_star:
                continue

            max_resulting_w = 0
            for coords, vals, has_X, has_O, w in unbroken_lies:
                if (r, c) in coords:
                    if (sym == 1 and not has_O) or (sym == 2 and not has_X):
                        if w + 1 > max_resulting_w:
                            max_resulting_w = w + 1
            if max_resulting_w > best_score:
                best_score = max_resulting_w
                best_candidates_vi = [(r, c, sym)]
            elif max_resulting_w == best_score:
                best_candidates_vi.append((r, c, sym))

    if best_candidates_vi:
        best_candidates_vi.sort(key=lambda m: (m[0], m[1]))
        return best_candidates_vi[0]

    # fallback to empty spot
    r, c = empty_spots[0]
    return (r, c, 1)


def _find_completion_move(inner_board, target_weight):
    """
    scan inner board for unbroken line of length `target_weight`

    returns (row, col, sym) to complete it
    """

    for coords, vals in _get_4x4_lines(inner_board):
        has_X = 1 in vals
        has_O = 2 in vals

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
    lines.append(
        ([(i, i) for i in range(4)], np.array([inner_board[i, i] for i in range(4)]))
    )
    lines.append(
        (
            [(i, 3 - i) for i in range(4)],
            np.array([inner_board[i, 3 - i] for i in range(4)]),
        )
    )
    return line

