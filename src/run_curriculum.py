"""
src/run_curriculum.py

Unified curriculum automation runner script for Order and Chaos RL project.
Sequences the 5-stage curriculum automatically:
  1. Stage 1  (Random)              : chaos_mode="random",    beta=0.00,  500 ep, lr=3e-4,   eps: 1.00 -> 0.05 (decay 0.994)
  2. Stage 2A (Light Defense)       : chaos_mode="heuristic", beta=0.35,  600 ep, lr=2e-4,   eps: 0.35 -> 0.05 (decay 0.995)
  3. Stage 2B (Medium Defense)      : chaos_mode="heuristic", beta=0.70,  800 ep, lr=1.5e-4, eps: 0.25 -> 0.03 (decay 0.996)
  4. Stage 2C (High Defense Bridge) : chaos_mode="heuristic", beta=0.85,  800 ep, lr=1.2e-4, eps: 0.20 -> 0.02 (decay 0.996)
  5. Stage 2D (Full Blocker)        : chaos_mode="heuristic", beta=1.00, 1500 ep, lr=1e-4,   eps: 0.15 -> 0.01 (decay 0.998)

Key Architectural & Training Features:
  - Stage handoff: each stage starts from the previous stage's *_best.pt weights (best = highest eval
    win rate, ties broken by fewer moves) with a trimmed replay buffer.
  - Increased Batch Size: Defaults to 128 for CPU vectorization and variance reduction.
  - Comprehensive Logging: Outputs metrics CSV and JSON files for each stage.
  - Automated Final Holdout Benchmark: 100-game greedy holdout vs Random Chaos and 100% Heuristic Chaos,
    benchmarked against the optimal theoretical Teacher.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.dqn import MaskedDQNAgent, encode_board
from src.env import OrderAndChaosEnv
from src.strategy import get_order_winning_move


CURRICULUM_STAGES = [
    {
        "id": "stage1_random",
        "name": "Stage 1: Random Opponent Baseline",
        "chaos_mode": "random",
        "chaos_defense_prob": 0.0,
        "episodes": 8000,
        "lr": 3e-4,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.99950,
    },
    {
        "id": "stage2a_light",
        "name": "Stage 2A: Light Defense (beta=0.35)",
        "chaos_mode": "heuristic",
        "chaos_defense_prob": 0.35,
        "episodes": 10000,
        "lr": 2e-4,
        "epsilon_start": 0.40,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.99970,
    },
    {
        "id": "stage2b_medium",
        "name": "Stage 2B: Medium Defense (beta=0.70)",
        "chaos_mode": "heuristic",
        "chaos_defense_prob": 0.70,
        "episodes": 12000,
        "lr": 1.5e-4,
        "epsilon_start": 0.30,
        "epsilon_end": 0.03,
        "epsilon_decay": 0.99973,
    },
    {
        "id": "stage2c_bridge",
        "name": "Stage 2C: High Defense Bridge (beta=0.85)",
        "chaos_mode": "heuristic",
        "chaos_defense_prob": 0.85,
        "episodes": 12000,
        "lr": 1.2e-4,
        "epsilon_start": 0.25,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.99972,
    },
    {
        "id": "stage2d_full",
        "name": "Stage 2D: Full Blocker (beta=1.00)",
        "chaos_mode": "heuristic",
        "chaos_defense_prob": 1.00,
        "episodes": 18000,
        "lr": 1e-4,
        "epsilon_start": 0.15,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99980,
    },
]


def set_global_seed(seed: int) -> None:
    """Seeds python, numpy and torch RNGs (epsilon-greedy, buffer sampling, init)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_greedy(
    agent: MaskedDQNAgent,
    env: OrderAndChaosEnv,
    num_episodes: int = 100,
    seed_base: int = 0,
) -> Dict[str, float]:
    """
    Evaluates agent greedily (epsilon=0) over num_episodes.
    Episode i resets the env with seed seed_base + i, so every checkpoint is
    evaluated on the same Chaos randomness (lower-variance checkpoint comparison).
    """
    wins = 0
    total_rewards = []
    total_lengths = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        ep_reward = 0.0
        moves = 0

        while not done:
            mask = env.action_masks()
            state = encode_board(obs)
            action = agent.select_action(state, mask, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            moves += 1
            if done and "Order wins" in info.get("result", ""):
                wins += 1

        total_rewards.append(ep_reward)
        total_lengths.append(moves)

    return {
        "win_rate": float(wins / num_episodes),
        "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_moves": float(np.mean(total_lengths)) if total_lengths else 0.0,
    }


def run_agent_holdout_benchmark(
    agent: MaskedDQNAgent,
    chaos_mode: str,
    chaos_defense_prob: float = 1.0,
    num_episodes: int = 100,
    seed_base: int = 1_000_000,
) -> Dict[str, float]:
    """
    Runs greedy evaluation for final holdout testing (fixed per-episode seeds).
    """
    env = OrderAndChaosEnv(
        config={"chaos_mode": chaos_mode, "chaos_defense_prob": chaos_defense_prob}
    )
    wins = 0
    moves_list = []
    rewards_list = []
    illegal_moves = 0

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        ep_moves = 0
        ep_reward = 0.0

        while not done:
            mask = env.action_masks()
            state = encode_board(obs)
            action = agent.select_action(state, mask, epsilon=0.0)

            if not mask[action]:
                illegal_moves += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_moves += 1
            ep_reward += reward

            if done and "Order wins" in info.get("result", ""):
                wins += 1

        moves_list.append(ep_moves)
        rewards_list.append(ep_reward)

    return {
        "chaos_mode": chaos_mode,
        "episodes": num_episodes,
        "win_rate": float((wins / num_episodes) * 100.0),
        "mean_moves": float(np.mean(moves_list)) if moves_list else 0.0,
        "std_moves": float(np.std(moves_list)) if moves_list else 0.0,
        "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "illegal_moves": illegal_moves,
    }


def run_teacher_holdout_benchmark(
    chaos_mode: str,
    chaos_defense_prob: float = 1.0,
    num_episodes: int = 100,
    seed_base: int = 1_000_000,
) -> Dict[str, float]:
    """
    Runs holdout benchmark for optimal theoretical Teacher policy.
    """
    env = OrderAndChaosEnv(
        config={"chaos_mode": chaos_mode, "chaos_defense_prob": chaos_defense_prob}
    )
    wins = 0
    moves_list = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        moves = 0
        while not done:
            move = get_order_winning_move(env.board)
            if move is None:
                break
            r, c, sym = move
            action = (r * 6 + c) * 2 + (sym - 1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            moves += 1
            if done and "Order wins" in info.get("result", ""):
                wins += 1

        moves_list.append(moves)

    return {
        "chaos_mode": chaos_mode,
        "episodes": num_episodes,
        "win_rate": float((wins / num_episodes) * 100.0),
        "mean_moves": float(np.mean(moves_list)) if moves_list else 0.0,
        "std_moves": float(np.std(moves_list)) if moves_list else 0.0,
    }


def print_and_save_benchmark_summary(
    agent: MaskedDQNAgent,
    holdout_episodes: int,
    output_dir: Path,
    seed: int = 0,
) -> Dict[str, dict]:
    """
    Runs holdout benchmarks and prints comparison table with Teacher.
    """
    print("\n" + "=" * 76)
    print("                FINAL HOLDOUT BENCHMARK EVALUATION                  ")
    print("=" * 76)
    print(
        f"{'Benchmark Opponent':<26} | {'Win Rate':<10} | {'Avg Moves (± std)':<18} | {'Illegal Moves':<13}"
    )
    print("-" * 76)

    # 1. DQN vs Random
    dqn_random = run_agent_holdout_benchmark(
        agent, chaos_mode="random", num_episodes=holdout_episodes
    )
    print(
        f"DQN vs Random              | "
        f"{dqn_random['win_rate']:>6.1f}%   | "
        f"{dqn_random['mean_moves']:>5.1f} ± {dqn_random['std_moves']:<5.1f}        | "
        f"{dqn_random['illegal_moves']:>13d}"
    )

    # 2. DQN vs Heuristic (100% Defense Blocker)
    dqn_heuristic = run_agent_holdout_benchmark(
        agent, chaos_mode="heuristic", chaos_defense_prob=1.0, num_episodes=holdout_episodes
    )
    print(
        f"DQN vs Heuristic (100%)    | "
        f"{dqn_heuristic['win_rate']:>6.1f}%   | "
        f"{dqn_heuristic['mean_moves']:>5.1f} ± {dqn_heuristic['std_moves']:<5.1f}        | "
        f"{dqn_heuristic['illegal_moves']:>13d}"
    )

    print("-" * 76)

    # 3. Teacher vs Random
    teacher_random = run_teacher_holdout_benchmark(
        chaos_mode="random", num_episodes=holdout_episodes
    )
    print(
        f"Teacher vs Random          | "
        f"{teacher_random['win_rate']:>6.1f}%   | "
        f"{teacher_random['mean_moves']:>5.1f} ± {teacher_random['std_moves']:<5.1f}        | "
        f"{'0 (Optimal)':>13}"
    )

    # 4. Teacher vs Heuristic (100%)
    teacher_heuristic = run_teacher_holdout_benchmark(
        chaos_mode="heuristic", chaos_defense_prob=1.0, num_episodes=holdout_episodes
    )
    print(
        f"Teacher vs Heuristic (100%)| "
        f"{teacher_heuristic['win_rate']:>6.1f}%   | "
        f"{teacher_heuristic['mean_moves']:>5.1f} ± {teacher_heuristic['std_moves']:<5.1f}        | "
        f"{'0 (Optimal)':>13}"
    )
    print("=" * 76 + "\n")

    benchmark_results = {
        "seed": seed,
        "holdout_episodes": holdout_episodes,
        "dqn_vs_random": dqn_random,
        "dqn_vs_heuristic_100": dqn_heuristic,
        "teacher_vs_random": teacher_random,
        "teacher_vs_heuristic_100": teacher_heuristic,
    }

    benchmark_json_path = output_dir / "final_holdout_benchmark.json"
    with open(benchmark_json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Benchmark results serialized to: {benchmark_json_path}")

    return benchmark_results


def run_curriculum(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick test configuration adjustments
    episodes_per_stage_override = 50 if args.quick_test else None
    eval_interval = 10 if args.quick_test else args.eval_interval
    eval_episodes = 20 if args.quick_test else args.eval_episodes
    holdout_episodes = 20 if args.quick_test else args.holdout_episodes

    print("=" * 76)
    print("           STARTING AUTOMATED ORDER & CHAOS CURRICULUM RUNNER       ")
    print("=" * 76)
    print(f"Device               : {args.device}")
    print(f"Algorithm            : {'Double DQN (DDQN)' if args.double_dqn else 'Standard DQN (Default)'}")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Buffer Capacity      : {args.buffer_capacity}")
    print(f"Buffer Trim Size     : {args.buffer_trim_size}")
    print(f"Discount Factor Gamma: {args.gamma}")
    print(f"Seed                 : {args.seed}")
    print(f"Output Directory     : {output_dir.resolve()}")
    print(f"Quick Test Mode      : {args.quick_test}")
    print(f"Eval Interval        : {eval_interval} eps (eval count: {eval_episodes})")
    print(f"Final Holdout Games  : {holdout_episodes}")
    print("=" * 76 + "\n")

    set_global_seed(args.seed)

    # 1. Instantiate shared MaskedDQNAgent
    agent = MaskedDQNAgent(
        lr=CURRICULUM_STAGES[0]["lr"],
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=args.device,
        double_dqn=args.double_dqn,
    )

    if args.initial_checkpoint and Path(args.initial_checkpoint).exists():
        print(f"Loading initial model weights from checkpoint: {args.initial_checkpoint}")
        agent.load(args.initial_checkpoint)

    # 2. Iterate through all curriculum stages sequentially
    for stage_idx, stage_info in enumerate(CURRICULUM_STAGES):
        stage_id = stage_info["id"]
        stage_name = stage_info["name"]
        chaos_mode = stage_info["chaos_mode"]
        chaos_defense_prob = stage_info["chaos_defense_prob"]
        episodes = (
            episodes_per_stage_override
            if episodes_per_stage_override is not None
            else stage_info["episodes"]
        )
        stage_lr = stage_info["lr"]
        eps_start = stage_info["epsilon_start"]
        eps_end = stage_info["epsilon_end"]
        eps_decay = stage_info["epsilon_decay"]

        stage_header = (
            f" [Stage {stage_idx + 1}/5] {stage_name} "
            f"(Mode: {chaos_mode}, Beta: {chaos_defense_prob:.2f}, Episodes: {episodes}) "
        )
        print("\n" + "#" * 76)
        print(f"{stage_header:^76}")
        print("#" * 76)

        # Replay Buffer Inheritance & Weights Loading
        if stage_idx > 0:
            prev_stage_id = CURRICULUM_STAGES[stage_idx - 1]["id"]
            prev_best_path = output_dir / f"order_dqn_{prev_stage_id}_best.pt"
            if prev_best_path.exists():
                print(
                    f"[Inheritance] Resuming weights from previous best checkpoint: {prev_best_path}"
                )
                agent.load(str(prev_best_path))
            else:
                print(
                    f"[Inheritance Warning] Previous checkpoint {prev_best_path} not found. Continuing with current network."
                )

            # Stage-aware replay buffer cleansing
            if args.buffer_trim_size > 0:
                prev_len = len(agent.replay_buffer)
                agent.replay_buffer.trim_to_size(max_retained=args.buffer_trim_size)
                print(
                    f"[Stage-Aware Cleansing] Trimmed replay buffer from {prev_len} "
                    f"to {len(agent.replay_buffer)} recent transitions to prevent distribution lag."
                )
            else:
                print(
                    f"[Crucial Replay Buffer Inheritance] Preserved active replay buffer "
                    f"with {len(agent.replay_buffer)} transitions."
                )

        # Set stage-specific learning rate
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = stage_lr
        print(f"[Optimizer] Learning rate configured to: {stage_lr:.2e}")

        # Configure environment for current stage with calibrated rewards
        env_config = {
            "chaos_mode": chaos_mode,
            "chaos_defense_prob": chaos_defense_prob,
            "win_reward": 1.0,
            "loss_reward": -1.0,
            "illegal_move_reward": -1.0,
            "step_reward": -0.005,
            "reward_3_in_a_row": 0.02,
            "reward_4_in_a_row": 0.05,
            "teacher_reward": 0.02,
        }
        env = OrderAndChaosEnv(config=env_config)
        # Separate evaluation env: evaluation never consumes the training env's RNG.
        eval_env = OrderAndChaosEnv(config=env_config)
        env.reset(seed=args.seed + stage_idx)

        # File paths for stage outputs
        best_model_path = output_dir / f"order_dqn_{stage_id}_best.pt"
        final_model_path = output_dir / f"order_dqn_{stage_id}_final.pt"
        metrics_csv_path = output_dir / f"order_dqn_{stage_id}_metrics.csv"
        metrics_json_path = output_dir / f"order_dqn_{stage_id}_metrics.json"
        history_json_path = output_dir / f"order_dqn_{stage_id}_history.json"

        with open(metrics_csv_path, "w") as f:
            f.write(
                "episode,epsilon,eval_win_rate,eval_mean_moves,eval_mean_reward,msbe_loss\n"
            )

        history = []
        epsilon = eps_start
        best_key = (-1.0, float("-inf"))  # (win_rate, -mean_moves): ties go to faster wins
        stage_start_time = time.time()

        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_losses = []

            while not done:
                mask = env.action_masks()
                state = encode_board(obs)
                action = agent.select_action(state, mask, epsilon=epsilon)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_mask = env.action_masks()
                next_state = encode_board(next_obs)

                # Push to inherited replay buffer
                agent.replay_buffer.push(
                    state, action, reward, next_state, done, next_mask
                )
                loss = agent.update()
                if loss > 0.0:
                    ep_losses.append(loss)

                obs = next_obs
                ep_reward += reward

            # Decay epsilon
            epsilon = max(eps_end, epsilon * eps_decay)

            # Evaluation check
            if ep % eval_interval == 0 or ep == episodes:
                eval_metrics = evaluate_greedy(
                    agent,
                    eval_env,
                    num_episodes=eval_episodes,
                    seed_base=args.seed * 1000,
                )
                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

                log_entry = {
                    "episode": ep,
                    "epsilon": float(epsilon),
                    "eval_win_rate": float(eval_metrics["win_rate"]),
                    "eval_mean_moves": float(eval_metrics["mean_moves"]),
                    "eval_mean_reward": float(eval_metrics["mean_reward"]),
                    "msbe_loss": mean_loss,
                }
                history.append(log_entry)

                with open(metrics_csv_path, "a") as f:
                    f.write(
                        f"{ep},{epsilon:.4f},{eval_metrics['win_rate']:.4f},"
                        f"{eval_metrics['mean_moves']:.2f},{eval_metrics['mean_reward']:.2f},{mean_loss:.4f}\n"
                    )

                with open(metrics_json_path, "w") as f:
                    json.dump(history, f, indent=2)

                # Also save _history.json for backwards compatibility
                with open(history_json_path, "w") as f:
                    json.dump(history, f, indent=2)

                print(
                    f"[{stage_id}] Ep {ep:4d}/{episodes:4d} | "
                    f"Eps: {epsilon:.3f} | "
                    f"Eval Win Rate: {eval_metrics['win_rate'] * 100:5.1f}% | "
                    f"Moves: {eval_metrics['mean_moves']:4.1f} | "
                    f"Reward: {eval_metrics['mean_reward']:6.1f} | "
                    f"MSBE Loss: {mean_loss:.4f} | "
                    f"Buf: {len(agent.replay_buffer)}"
                )

                key = (eval_metrics["win_rate"], -eval_metrics["mean_moves"])
                if key > best_key:
                    best_key = key
                    agent.save(str(best_model_path))
                    print(
                        f"  ==> New stage best model saved to {best_model_path.name} "
                        f"(Win Rate: {key[0] * 100:.1f}%, Moves: {-key[1]:.1f})"
                    )

        # Fallback: ensure best_model_path exists even if best_win_rate was 0.0
        if not best_model_path.exists():
            agent.save(str(best_model_path))

        agent.save(str(final_model_path))
        elapsed = time.time() - stage_start_time
        print(
            f"Stage completed in {elapsed:.1f}s. Saved: {final_model_path.name} and {best_model_path.name}"
        )

    # 3. Automated Final Holdout Benchmark after Stage 2D
    final_stage_id = CURRICULUM_STAGES[-1]["id"]
    final_best_path = output_dir / f"order_dqn_{final_stage_id}_best.pt"
    if final_best_path.exists():
        print(f"\nLoading Stage 2D best model for final holdout: {final_best_path}")
        agent.load(str(final_best_path))

    print_and_save_benchmark_summary(
        agent=agent,
        holdout_episodes=holdout_episodes,
        output_dir=output_dir,
        seed=args.seed,
    )
    print("Curriculum pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Curriculum Automation Runner for Order & Chaos DQN"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch computation device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick debugging test (50 episodes per stage, 20 benchmark games)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints and metric logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for experience replay updates (default: 256)",
    )
    parser.add_argument(
        "--double_dqn",
        action="store_true",
        default=False,
        help="Enable Double DQN target calculation (decouples action selection and evaluation). Default: False (Standard DQN).",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=100000,
        help="Maximum capacity of inherited replay buffer (default: 100000)",
    )
    parser.add_argument(
        "--buffer_trim_size",
        type=int,
        default=20000,
        help="Maximum transitions to retain across stage boundaries (default: 20000, set <=0 to disable)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        help="Discount factor for Bellman optimality backup (default: 0.90)",
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=1000,
        help="Frequency (in gradient steps) for target network synchronization",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Episodes between greedy evaluations during regular training",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Number of episodes per evaluation run (default: 100; 40 gives ~+-6%% noise)",
    )
    parser.add_argument(
        "--holdout_episodes",
        type=int,
        default=100,
        help="Number of holdout games for final evaluation vs Random/Heuristic/Teacher",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for python/numpy/torch RNGs and the environments (default: 0)",
    )
    parser.add_argument(
        "--initial_checkpoint",
        type=str,
        default=None,
        help="Optional initial checkpoint to bootstrap Stage 1",
    )

    args = parser.parse_args()
    run_curriculum(args)


if __name__ == "__main__":
    main()
