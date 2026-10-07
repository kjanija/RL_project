"""
time_test.py

Standalone profiling and resource calibration script for Order & Chaos DQN.
Measures:
  - Environment step latency (CPU)
  - Action selection / inference latency (CPU or GPU)
  - Mini-batch gradient update latency (CPU or GPU)
  - Full episode throughput and steps-per-second (SPS)
  - Exact maximum safe training episodes fitting within a 2-hour cluster budget
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.dqn import MaskedDQNAgent, encode_board
from src.env import OrderAndChaosEnv


def run_time_test(device: str = "cpu", batch_size: int = 256, test_episodes: int = 150):
    print("=" * 76)
    print("       ORDER & CHAOS RL: CLUSTER BENCHMARK & TIME CALIBRATION        ")
    print("=" * 76)
    print(f"Target Device      : {device.upper()}")
    print(f"Mini-batch Size    : {batch_size}")
    print(f"Calibration Games  : {test_episodes}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU Device Name    : {torch.cuda.get_device_name(0)}")
        print(f"Allocated Memory   : {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print("=" * 76 + "\n")

    # 1. Initialize environment (Heuristic blocker beta=1.0 for worst-case step latency)
    env = OrderAndChaosEnv(
        config={
            "chaos_mode": "heuristic",
            "chaos_defense_prob": 1.0,
            "win_reward": 100.0,
            "loss_reward": -100.0,
            "illegal_move_reward": -100.0,
            "step_reward": -0.5,
            "reward_3_in_a_row": 1.0,
            "reward_4_in_a_row": 5.0,
            "teacher_reward": 2.0,
        }
    )

    agent = MaskedDQNAgent(
        batch_size=batch_size,
        buffer_capacity=50000,
        device=device,
    )

    # Pre-fill replay buffer so update() executes on every step
    print("[Warmup] Pre-filling replay buffer with warmup transitions...")
    obs, _ = env.reset()
    for _ in range(batch_size + 10):
        mask = env.action_masks()
        state = encode_board(obs)
        legal_actions = np.where(mask)[0]
        act = int(np.random.choice(legal_actions))
        n_obs, r, term, trunc, _ = env.step(act)
        n_mask = env.action_masks()
        n_state = encode_board(n_obs)
        agent.replay_buffer.push(state, act, r, n_state, term or trunc, n_mask)
        obs = n_obs if not (term or trunc) else env.reset()[0]

    # Benchmark profiling loop
    env_step_times = []
    action_select_times = []
    update_times = []
    total_steps = 0
    start_total_time = time.perf_counter()

    print(f"[Profiling] Executing {test_episodes} complete episodes...")
    for ep in range(1, test_episodes + 1):
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.action_masks()
            state = encode_board(obs)

            # Measure action selection
            t0 = time.perf_counter()
            action = agent.select_action(state, mask, epsilon=0.1)
            if device == "cuda":
                torch.cuda.synchronize()
            action_select_times.append(time.perf_counter() - t0)

            # Measure environment transition
            t1 = time.perf_counter()
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_mask = env.action_masks()
            next_state = encode_board(next_obs)
            env_step_times.append(time.perf_counter() - t1)

            agent.replay_buffer.push(state, action, reward, next_state, done, next_mask)

            # Measure gradient backpropagation update
            t2 = time.perf_counter()
            agent.update()
            if device == "cuda":
                torch.cuda.synchronize()
            update_times.append(time.perf_counter() - t2)

            obs = next_obs
            total_steps += 1

    total_elapsed = time.perf_counter() - start_total_time

    # Derived performance metrics
    sps = total_steps / total_elapsed
    eps_per_sec = test_episodes / total_elapsed
    eps_per_min = eps_per_sec * 60.0
    sec_per_ep = total_elapsed / test_episodes
    mean_moves_per_ep = total_steps / test_episodes

    mean_env_ms = float(np.mean(env_step_times) * 1000.0)
    mean_act_ms = float(np.mean(action_select_times) * 1000.0)
    mean_upd_ms = float(np.mean(update_times) * 1000.0)

    # 2-Hour Time Budget Projections (105 min training + 15 min evaluation/checkpoint safety buffer)
    usable_seconds = 105 * 60
    max_episodes_2h = int(usable_seconds * eps_per_sec)

    print("\n" + "=" * 76)
    print("                         BENCHMARK RESULTS                           ")
    print("=" * 76)
    print(f"Total Test Episodes : {test_episodes}")
    print(f"Total Steps Sampled : {total_steps}")
    print(f"Total Wall Time     : {total_elapsed:.2f} s")
    print("-" * 76)
    print(f"Throughput          : {sps:6.1f} steps/second (SPS)")
    print(f"Episode Speed       : {eps_per_sec:6.2f} games/sec ({eps_per_min:6.1f} games/minute)")
    print(f"Mean Latency per Ep : {sec_per_ep * 1000:6.1f} ms (avg {mean_moves_per_ep:.1f} moves/ep)")
    print("-" * 76)
    print("Component Breakdown (per step):")
    print(f"  • Env Step + Chaos Heuristic : {mean_env_ms:6.3f} ms")
    print(f"  • Policy Action Selection    : {mean_act_ms:6.3f} ms")
    print(f"  • Q-Network Update (batch {batch_size}) : {mean_upd_ms:6.3f} ms")
    print("=" * 76)
    print("              2-HOUR ALLOCATION RECOMMENDATION (105m budget)         ")
    print("=" * 76)
    print(f"Safe Total Episodes for 2h Window : ~{max_episodes_2h:,} episodes")
    print("\nRecommended Stage Distribution based on measured speed:")

    # Proportion: 12% Stage 1, 16% Stage 2A, 20% Stage 2B, 20% Stage 2C, 32% Stage 2D
    s1 = int(round(max_episodes_2h * 0.12 / 100.0) * 100)
    s2a = int(round(max_episodes_2h * 0.16 / 100.0) * 100)
    s2b = int(round(max_episodes_2h * 0.20 / 100.0) * 100)
    s2c = int(round(max_episodes_2h * 0.20 / 100.0) * 100)
    s2d = max_episodes_2h - (s1 + s2a + s2b + s2c)

    print(f"  • Stage 1  (Random Baseline,   beta=0.00): {s1:6d} episodes")
    print(f"  • Stage 2A (Light Defense,     beta=0.35): {s2a:6d} episodes")
    print(f"  • Stage 2B (Medium Defense,    beta=0.70): {s2b:6d} episodes")
    print(f"  • Stage 2C (High Defense,      beta=0.85): {s2c:6d} episodes")
    print(f"  • Stage 2D (Full Blocker,      beta=1.00): {s2d:6d} episodes")
    print(f"  ------------------------------------------------")
    print(f"  • Total Sum                              : {s1 + s2a + s2b + s2c + s2d:6d} episodes")
    print("=" * 76 + "\n")

    results_data = {
        "device": device,
        "batch_size": batch_size,
        "test_episodes": test_episodes,
        "total_steps": total_steps,
        "total_elapsed_s": total_elapsed,
        "sps": sps,
        "eps_per_sec": eps_per_sec,
        "eps_per_min": eps_per_min,
        "sec_per_ep": sec_per_ep,
        "mean_moves_per_ep": mean_moves_per_ep,
        "component_latencies_ms": {
            "env_step_mean_ms": mean_env_ms,
            "policy_action_mean_ms": mean_act_ms,
            "q_update_mean_ms": mean_upd_ms,
        },
        "safe_total_episodes_2h": max_episodes_2h,
        "recommended_stages": {
            "stage1_random": s1,
            "stage2a_light": s2a,
            "stage2b_medium": s2b,
            "stage2c_bridge": s2c,
            "stage2d_full": s2d,
        },
    }

    import json
    out_file = Path("time_test_results.json")
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Detailed calibration metrics serialized to: {out_file.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Order & Chaos training speed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to benchmark ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for DQN updates",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=150,
        help="Number of episodes for profiling",
    )
    args = parser.parse_args()

    run_time_test(device=args.device, batch_size=args.batch_size, test_episodes=args.test_episodes)
