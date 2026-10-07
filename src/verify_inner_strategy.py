"""
src/verify_inner_strategy.py

Exhaustive check of the 4x4 inner-board strategy (`_inner_stretegy`).

Model of the inner game: Order moves first with the teacher's inner rules,
Chaos replies with ANY (cell, symbol) in the 4x4. Chaos 'escapes' if the inner
board fills up without any uniform line of 4 ever appearing (the teacher then
has no 4-line to extend into the border and must open a border pair itself).
Chaos moves to the border are 'passes' (Order mirrors them), so they do not
change the inner board and can be ignored.

Usage:
    uv run python src/verify_inner_strategy.py              # teacher (solved)
    uv run python src/verify_inner_strategy.py --heuristic  # Strategy IV.2 rules
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.inner_solver import canonical
from src.strategy import _get_4x4_lines, _inner_heuristic_strategy, _inner_stretegy


def has_uniform_4_line(inner_board) -> bool:
    return any(
        vals[0] != 0 and np.all(vals == vals[0])
        for _, vals in _get_4x4_lines(inner_board)
    )


def find_chaos_escape(strategy=None, dedupe_symmetry=False):
    """
    `strategy` maps an inner board (Order to move) to (row, col, symbol); it
    defaults to the teacher's strategy (_inner_stretegy).

    `dedupe_symmetry` visits one board per symmetry class (8 board symmetries x X/O
    swap). Only valid when the strategy's winning-ness is symmetric (e.g. the solved
    teacher), and ~16x faster.

    Returns a list of (player, row, col, symbol) moves leading to a full inner
    board without a uniform 4-line, or None if the strategy always forces one.
    """
    strategy = strategy or _inner_stretegy
    seen = set()

    def chaos_turn(board, path):
        key = (
            canonical(tuple(int(x) for x in board.flatten()))
            if dedupe_symmetry
            else board.tobytes()
        )
        if key in seen:
            return None
        seen.add(key)

        for r in range(4):
            for c in range(4):
                if board[r, c] != 0:
                    continue
                for sym in (1, 2):
                    nb = board.copy()
                    nb[r, c] = sym
                    if has_uniform_4_line(nb):
                        continue
                    chaos_move = ("C", r, c, sym)
                    if not np.any(nb == 0):
                        return path + [chaos_move]
                    orow, ocol, osym = strategy(nb)
                    nb[orow, ocol] = osym
                    if has_uniform_4_line(nb):
                        continue
                    res = chaos_turn(nb, path + [chaos_move, ("O", orow, ocol, osym)])
                    if res is not None:
                        return res
        return None

    board = np.zeros((4, 4), dtype=int)
    r, c, s = strategy(board)
    board[r, c] = s
    return chaos_turn(board, [("O", r, c, s)])


def main():
    parser = argparse.ArgumentParser(description="Exhaustively verify the inner strategy")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Check the rule-based Strategy IV.2 transcription instead of the solved teacher",
    )
    args = parser.parse_args()
    escape = find_chaos_escape(_inner_heuristic_strategy if args.heuristic else None)
    if escape is None:
        print("OK: inner strategy always forces a uniform 4-line.")
        return
    print("FAIL: Chaos can fill the inner 4x4 without any uniform 4-line.")
    grid = np.zeros((4, 4), dtype=int)
    for player, r, c, s in escape:
        grid[r, c] = s
        print(f"  {player}: {'XO'[s - 1]} at ({r}, {c})")
    print("Final inner board (1=X, 2=O):")
    print(grid)


if __name__ == "__main__":
    main()
