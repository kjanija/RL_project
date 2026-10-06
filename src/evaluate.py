"""
Evaluatoin benchmark script for trained Order DQN policies

Tests against:
    - random chaos
    - heuristic defensive chaos
    - teacher strategy benchmark
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.env import OrderAndChaosEnv
from src.dqn import MaskedDQNAgent, encode_board
from src.strategy import get_order_winning_move                                                                                                                                         
                                                                                                                                                                                        
                                                                                                                                                                                        
def run_benchmark(agent: MaskedDQNAgent, chaos_mode: str, num_episodes: int = 100) -> dict:                                                                                             
    env = OrderAndChaosEnv(config={"chaos_mode": chaos_mode})                                                                                                                           
    wins = 0                                                                                                                                                                            
    moves_list = []                                                                                                                                                                     
    rewards_list = []                                                                                                                                                                   
    illegal_moves = 0                                                                                                                                                                   
                                                                                                                                                                                        
    for _ in range(num_episodes):                                                                                                                                                       
        obs, _ = env.reset()                                                                                                                                                            
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
        "win_rate": (wins / num_episodes) * 100.0,                                                                                                                                      
        "mean_moves": float(np.mean(moves_list)),                                                                                                                                       
        "std_moves": float(np.std(moves_list)),                                                                                                                                         
        "mean_reward": float(np.mean(rewards_list)),                                                                                                                                    
        "illegal_moves": illegal_moves,                                                                                                                                                 
    }                                                                                                                                                                                   
                                                                                                                                                                                        
                                                                                                                                                                                        
def run_teacher_benchmark(chaos_mode: str, num_episodes: int = 100) -> dict:                                                                                                            
    env = OrderAndChaosEnv(config={"chaos_mode": chaos_mode})                                                                                                                           
    wins = 0                                                                                                                                                                            
    moves_list = []                                                                                                                                                                     
                                                                                                                                                                                        
    for _ in range(num_episodes):                                                                                                                                                       
        obs, _ = env.reset()                                                                                                                                                            
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
        "win_rate": (wins / num_episodes) * 100.0,                                                                                                                                      
        "mean_moves": float(np.mean(moves_list)),                                                                                                                                       
    }                                                                                                                                                                                   
                                                                                                                                                                                        
                                                                                                                                                                                        
def main():                                                                                                                                                                             
    parser = argparse.ArgumentParser(description="Benchmark trained Order DQN agent")                                                                                                   
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint (.pt)")                                                                           
    parser.add_argument("--episodes", type=int, default=100, help="Number of benchmark games per mode")                                                                                 
    parser.add_argument("--device", type=str, default="cpu")                                                                                                                            
    args = parser.parse_args()                                                                                                                                                          
                                                                                                                                                                                        
    agent = MaskedDQNAgent(device=args.device)                                                                                                                                          
    agent.load(args.model_path)                                                                                                                                                         
    print(f"Loaded trained DQN policy from: {args.model_path}\n")                                                                                                                       
                                                                                                                                                                                        
    print("=" * 65)                                                                                                                                                                     
    print(f"{'Benchmark Mode':<20} | {'Win Rate':<12} | {'Avg Moves':<12} | {'Illegal Moves':<12}")                                                                                     
    print("=" * 65)                                                                                                                                                                     
                                                                                                                                                                                        
    for mode in ["random", "heuristic"]:                                                                                                                                                
        res = run_benchmark(agent, mode, num_episodes=args.episodes)                                                                                                                    
        print(                                                                                                                                                                          
            f"DQN vs {res['chaos_mode'].capitalize():<13} | "                                                                                                                           
            f"{res['win_rate']:>6.1f}%      | "                                                                                                                                         
            f"{res['mean_moves']:>6.1f} ± {res['std_moves']:.1f} | "                                                                                                                    
            f"{res['illegal_moves']:>12d}"                                                                                                                                              
        )                                                                                                                                                                               
                                                                                                                                                                                        
    print("-" * 65)                                                                                                                                                                     
    for mode in ["random", "heuristic"]:                                                                                                                                                
        teacher_res = run_teacher_benchmark(mode, num_episodes=args.episodes)                                                                                                           
        print(                                                                                                                                                                          
            f"Teacher vs {mode.capitalize():<10} | "                                                                                                                                    
            f"{teacher_res['win_rate']:>6.1f}%      | "                                                                                                                                 
            f"{teacher_res['mean_moves']:>6.1f}        | "                                                                                                                              
            f"{'0 (Optimal)':>12}"                                                                                                                                                      
        )                                                                                                                                                                               
    print("=" * 65)                                                                                                                                                                     
                                                                                                                                                                                        
                                                                                                                                                                                        
if __name__ == "__main__":                                                                                                                                                              
    main()