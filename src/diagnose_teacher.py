"""
src/diagnose_teacher.py

Diagnostic for the theoretical Teacher (src/strategy.py).

The 'Order versus Chaos' paper proves Order wins on the 6x6 board against ANY
Chaos play. This script plays the Teacher against the project's Chaos
opponents and prints every game it fails to win, so that a strategy bug (or a
mismatch between the env and the paper's rules) can be located.

Usage:
    uv run python src/diagnose_teacher.py --episodes 1000 --chaos_mode heuristic
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import OrderAndChaosEnv
from src.strategy import get_order_winning_move

SYMBOLS = {0: ".", 1: "X", 2: "O"}


def board_to_str(board) -> str:
    return "\n".join(" ".join(SYMBOLS[int(v)] for v in row) for row in board)


def play_game(env: OrderAndChaosEnv, seed: int):
    """Plays one Teacher game. Returns (won, history, final_info)."""
    env.reset(seed=seed)
    history = []
    done = False
    info = {}
    while not done:
        board_before = env.board.copy()
        r, c, sym = get_order_winning_move(env.board)
        action = (r * 6 + c) * 2 + (sym - 1)
        if board_before[r, c] != 0:
            return False, history, {"error": f"Teacher chose occupied cell {(r, c)}"}
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append((board_before, (r, c, sym), env.board.copy()))
    return "Order wins" in info.get("result", ""), history, info


def main():
    parser = argparse.ArgumentParser(description="Find games the Teacher fails to win")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--chaos_mode", type=str, default="heuristic", choices=["random", "heuristic"]
    )
    parser.add_argument("--chaos_defense_prob", type=float, default=1.0)
    parser.add_argument("--max_print", type=int, default=3)
    args = parser.parse_args()

    env = OrderAndChaosEnv(
        config={
            "chaos_mode": args.chaos_mode,
            "chaos_defense_prob": args.chaos_defense_prob,
        }
    )

    losses = []
    for seed in range(args.episodes):
        won, history, info = play_game(env, seed)
        if not won:
            losses.append((seed, history, info))

    wins = args.episodes - len(losses)
    print(
        f"Teacher vs {args.chaos_mode} (beta={args.chaos_defense_prob}): "
        f"{wins}/{args.episodes} wins ({100.0 * wins / args.episodes:.1f}%)"
    )

    for seed, history, info in losses[: args.max_print]:
        print("\n" + "=" * 60)
        print(f"LOSS seed={seed}  info={info}")
        for i, (before, move, after) in enumerate(history):
            print(f"\n-- Move {i + 1}: Teacher plays {SYMBOLS[move[2]]} at {move[:2]}")
            print("before:")
            print(board_to_str(before))
            print("after Teacher + Chaos:")
            print(board_to_str(after))


if __name__ == "__main__":
    main()
