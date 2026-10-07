import pytest

from src.inner_solver import table_available, winning_move
from src.strategy import _inner_heuristic_strategy
from src.verify_inner_strategy import find_chaos_escape


@pytest.mark.skipif(not table_available(), reason="inner win table not built")
def test_teacher_inner_strategy_always_forces_a_4_line():
    # Solved moves only (no heuristic preference) with symmetry dedupe keeps this fast.
    assert find_chaos_escape(lambda board: winning_move(board), dedupe_symmetry=True) is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Strategy IV.2 as transcribed from the paper text lets Chaos fill the 4x4 "
        "without a uniform 4-line (exact solver: Order's winning moves differ). "
        "The teacher uses src/inner_solver.py instead."
    ),
)
def test_paper_rule_heuristic_is_not_winning():
    assert find_chaos_escape(_inner_heuristic_strategy) is None
