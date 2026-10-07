"""
src/inner_solver.py

Exact solver for the inner 4x4 game used by the Order teacher.

Background: Order wins the 6x6 game by (a) mirroring Chaos' border moves and
(b) winning the 4x4 inner game ovc'([4]^2, 4, Chaos), i.e. forcing a
homogeneous line of 4 in the inner board (van den Bergh et al., "Order versus
Chaos", Lemma IV.3 / IV.4). The rule-based Strategy IV.2 printed in the paper
does not verify when implemented from the text (see src/verify_inner_strategy.py),
so the teacher uses an exact solution of the 4x4 game instead.

The solution is stored as the set of *Chaos-to-move* inner positions (canonical
under the 8 board symmetries and the X/O swap) from which Order forces a
homogeneous 4-line. The set is closed: for every position in it, every Chaos
reply leads to a position where some Order move lands again in the set (or
completes a line). Looking up a move is therefore sound and always succeeds
while Order stays inside the set.

Build the table once (takes ~2 minutes):
    uv run python src/inner_solver.py --build
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

TABLE_PATH = Path(__file__).resolve().parent / "data" / "inner_win_table.npy"

_ROWS = [[r * 4 + c for c in range(4)] for r in range(4)]
_COLS = [[r * 4 + c for r in range(4)] for c in range(4)]
_DIAGS = [[i * 4 + i for i in range(4)], [i * 4 + (3 - i) for i in range(4)]]
LINE_INDICES = _ROWS + _COLS + _DIAGS


def _rot(i):
    r, c = divmod(i, 4)
    return c * 4 + (3 - r)


def _refl(i):
    r, c = divmod(i, 4)
    return r * 4 + (3 - c)


def _build_perms():
    perms = []
    for k in range(4):
        for f in (0, 1):
            p = []
            for i in range(16):
                j = i
                for _ in range(k):
                    j = _rot(j)
                if f:
                    j = _refl(j)
                p.append(j)
            perms.append(p)
    return perms


PERMS = _build_perms()
_SWAP = (0, 2, 1)


def canonical(board):
    """Smallest tuple over the 8 symmetries and the X/O swap."""
    best = None
    for p in PERMS:
        t = [0] * 16
        for i in range(16):
            t[p[i]] = board[i]
        t = tuple(t)
        if best is None or t < best:
            best = t
        t2 = tuple(_SWAP[x] for x in t)
        if t2 < best:
            best = t2
    return best


def encode(canon_board) -> int:
    value = 0
    for x in canon_board:
        value = value * 3 + x
    return value


def has_line(board) -> bool:
    for L in LINE_INDICES:
        v = board[L[0]]
        if v and v == board[L[1]] == board[L[2]] == board[L[3]]:
            return True
    return False


# ---------------------------------------------------------------------------
# Solver (used only to build the table)
# ---------------------------------------------------------------------------
class _Solver:
    def __init__(self):
        self.order_memo = {}
        self.chaos_memo = {}

    def order_wins(self, b) -> bool:
        """Order to move."""
        if has_line(b):
            return True
        k = canonical(b)
        if k in self.order_memo:
            return self.order_memo[k]
        res = False
        for i in range(16):
            if b[i]:
                continue
            for s in (1, 2):
                nb = list(b)
                nb[i] = s
                if self.chaos_loses(tuple(nb)):
                    res = True
                    break
            if res:
                break
        self.order_memo[k] = res
        return res

    def chaos_loses(self, b) -> bool:
        """Chaos to move; True if Order forces a line."""
        if has_line(b):
            return True
        if all(b):
            return False
        k = canonical(b)
        if k in self.chaos_memo:
            return self.chaos_memo[k]
        res = True
        for i in range(16):
            if b[i]:
                continue
            for s in (1, 2):
                nb = list(b)
                nb[i] = s
                if not self.order_wins(tuple(nb)):
                    res = False
                    break
            if not res:
                break
        self.chaos_memo[k] = res
        return res


def build_table():
    solver = _Solver()
    empty = tuple([0] * 16)
    if not solver.order_wins(empty):
        raise RuntimeError("Order does not win the inner 4x4 game (unexpected).")
    # The paper's opening move (X at (2, 2) in 1-indexed coordinates) must be a winning move
    # so the teacher keeps it. Evaluating it also stores it (and its closure) in the table.
    paper_opening = [0] * 16
    paper_opening[1 * 4 + 1] = 1
    if not solver.chaos_loses(tuple(paper_opening)):
        raise RuntimeError("The paper's opening move is not winning (unexpected).")
    keys = sorted(encode(k) for k, v in solver.chaos_memo.items() if v)
    return np.array(keys, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Runtime lookup
# ---------------------------------------------------------------------------
_WIN_SET = None


def table_available() -> bool:
    return _load_table() is not None


def _load_table():
    global _WIN_SET
    if _WIN_SET is None:
        if not TABLE_PATH.exists():
            return None
        _WIN_SET = frozenset(int(x) for x in np.load(TABLE_PATH))
    return _WIN_SET


def is_winning_after(board_tuple) -> bool:
    """True if the Chaos-to-move position `board_tuple` is a forced Order win."""
    table = _load_table()
    if table is None:
        raise RuntimeError(f"Missing {TABLE_PATH}; run: uv run python src/inner_solver.py --build")
    if has_line(board_tuple):
        return True
    return encode(canonical(board_tuple)) in table


def winning_move(inner_board, preferred=None):
    """
    Returns (row, col, symbol) in inner coordinates for a move that keeps a
    forced win, or None if the table is missing or no such move exists.
    `preferred` (a move tuple) is returned if it is winning.
    """
    if _load_table() is None:
        return None
    flat = tuple(int(x) for x in np.asarray(inner_board).flatten())

    def after(move):
        r, c, s = move
        nb = list(flat)
        nb[r * 4 + c] = s
        return tuple(nb)

    if preferred is not None and flat[preferred[0] * 4 + preferred[1]] == 0:
        if is_winning_after(after(preferred)):
            return preferred

    for i in range(16):
        if flat[i]:
            continue
        for s in (1, 2):
            move = (i // 4, i % 4, s)
            if is_winning_after(after(move)):
                return move
    return None


def main():
    parser = argparse.ArgumentParser(description="Build the inner 4x4 win table")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return
    start = time.time()
    table = build_table()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TABLE_PATH, table)
    print(
        f"Saved {len(table)} winning positions to {TABLE_PATH} "
        f"({TABLE_PATH.stat().st_size / 1e6:.2f} MB) in {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
