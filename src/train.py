"""
Curriculum training script for Order's DQN policy
1. random chaos
2. heuristic chaos
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dqn import MaskedDQNAgent, encode_board
from src.env import OrderAndChaosEnv


def evaluate(
    agent: MaskedDQNAgent, env: OrderAndChaosEnv, num_episodes: int = 50
) -> dict:
    """
    Evaluates agent greedily (eps=0) over num_episodes
    """
    wins = 0
    total_rewards = []
    total_lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
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
            if done:
                if "Order wins" in info.get("result", ""):
                    wins += 1

        total_rewards.append(ep_reward)
        total_lengths.append(moves)

    return {
        "win_rate": wins / num_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_moves": float(np.mean(total_lengths)),
    }


def train_dqn(args):
    env_config = {
        "chaos_mode": args.chaos_mode,
        "chaos_defense_prob": args.chaos_defense_prob,
        "win_reward": 1.0,
        "loss_reward": -1.0,
        "illegal_move_reward": -1.0,
        "step_reward": -0.005,
        "reward_3_in_a_row": 0.02,
        "reward_4_in_a_row": 0.05,
        "teacher_reward": 0.02,
    }

    env = OrderAndChaosEnv(config=env_config)

    agent = MaskedDQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=args.device,
        double_dqn=args.double_dqn,
    )

    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        agent.load(args.resume_checkpoint)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = args.lr
        print(f"Resumed training from checkpoint: {args.resume_checkpoint} (LR set to {args.lr})")

    model_tag = args.model_tag if args.model_tag else args.chaos_mode
    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"order_dqn_{model_tag}_best.pt"

    # Setup CSV and JSON metrics tracking
    metrics_csv_path = save_dir / f"order_dqn_{model_tag}_metrics.csv"
    history_json_path = save_dir / f"order_dqn_{model_tag}_history.json"

    with open(metrics_csv_path, "w") as f:
        f.write("episode,epsilon,eval_win_rate,eval_mean_moves,eval_mean_reward,msbe_loss\n")

    history = []
    epsilon = args.epsilon_start
    best_win_rate = -1.0

    print(
        f"--- Starting DQN training (Chaos: {args.chaos_mode}, Defense Prob: {args.chaos_defense_prob:.2f}, Tag: {model_tag}, Episodes: {args.episodes}) ---"
    )

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_loss = []

        while not done:
            mask = env.action_masks()
            state = encode_board(obs)
            action = agent.select_action(state, mask, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_masks()
            next_state = encode_board(next_obs)

            agent.replay_buffer.push(state, action, reward, next_state, done, next_mask)
            loss = agent.update()
            if loss > 0.0:
                ep_loss.append(loss)

            obs = next_obs
            ep_reward += reward

        # decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.eval_interval == 0:
            eval_metrics = evaluate(agent, env, num_episodes=args.eval_episodes)
            mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0

            log_entry = {
                "episode": episode,
                "epsilon": float(epsilon),
                "eval_win_rate": float(eval_metrics["win_rate"]),
                "eval_mean_moves": float(eval_metrics["mean_moves"]),
                "eval_mean_reward": float(eval_metrics["mean_reward"]),
                "msbe_loss": mean_loss,
            }
            history.append(log_entry)

            with open(metrics_csv_path, "a") as f:
                f.write(
                    f"{episode},{epsilon:.4f},{eval_metrics['win_rate']:.4f},"
                    f"{eval_metrics['mean_moves']:.2f},{eval_metrics['mean_reward']:.2f},{mean_loss:.4f}\n"
                )

            with open(history_json_path, "w") as f:
                json.dump(history, f, indent=2)

            print(
                f"Episode {episode:5d} | Epsilon: {epsilon:.3f} | "
                f"Eval Win Rate: {eval_metrics['win_rate'] * 100:5.1f}% | "
                f"Eval Avg Moves: {eval_metrics['mean_moves']:4.1f} | "
                f"Eval Avg Reward: {eval_metrics['mean_reward']:6.1f} | "
                f"MSBE Loss: {mean_loss:.4f}"
            )

            if eval_metrics["win_rate"] > best_win_rate:
                best_win_rate = eval_metrics["win_rate"]
                agent.save(str(best_model_path))
                print(
                    f"  ==> Saved new best model to {best_model_path} (Win Rate: {best_win_rate * 100:.1f}%)"
                )

    final_model_path = save_dir / f"order_dqn_{model_tag}_final.pt"
    agent.save(str(final_model_path))
    print(f"Training complete. Final model saved to {final_model_path}")
    print(f"Metrics saved to {metrics_csv_path} and {history_json_path}")

if __name__ == "__main__":                                                              
    parser = argparse.ArgumentParser(description="Train Order's policy using DQN with Action Masking")                                                                          
    parser.add_argument("--chaos_mode", type=str, default="random", choices=["random", "heuristic"], help="Opponent mode")
    parser.add_argument("--chaos_defense_prob", type=float, default=1.0, help="Probability of playing heuristic defensive moves (vs random fallback)")
    parser.add_argument("--model_tag", type=str, default=None, help="Optional custom identifier for saved model/metrics filenames")                                                       
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")                                                                                
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for MSBE projection")                                                                         
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor (default: 0.90)")    
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity (default: 100000)")                                                                         
    parser.add_argument("--double_dqn", action="store_true", default=False, help="Enable Double DQN target calculation")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update period")                                                                   
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")                                                                        
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Minimum exploration rate")                                                                        
    parser.add_argument("--epsilon_decay", type=float, default=0.997, help="Multiplicative epsilon decay factor")                                               
    parser.add_argument("--eval_interval", type=int, default=100, help="Episodes between evaluations")                                                                             
    parser.add_argument("--eval_episodes", type=int, default=40, help="Episodes per evaluation run")                                                                          
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")                                                               
    parser.add_argument("--device", type=str, default="cpu", help="Torch device ('cpu' or 'cuda')")                                                                              
                                                                                        
    args = parser.parse_args()                                                          
    train_dqn(args)