import numpy as np
import torch

from src.dqn import MaskedDQNAgent
from src.env import OrderAndChaosEnv
from src.run_curriculum import evaluate_greedy, set_global_seed


def test_set_global_seed_makes_agent_init_reproducible():
    set_global_seed(7)
    a = MaskedDQNAgent(device="cpu")
    set_global_seed(7)
    b = MaskedDQNAgent(device="cpu")
    for pa, pb in zip(a.q_net.parameters(), b.q_net.parameters()):
        assert torch.equal(pa, pb)


def test_evaluate_greedy_is_deterministic_for_fixed_seed_base():
    set_global_seed(0)
    agent = MaskedDQNAgent(device="cpu")
    env = OrderAndChaosEnv(config={"chaos_mode": "heuristic", "chaos_defense_prob": 0.5})
    first = evaluate_greedy(agent, env, num_episodes=5, seed_base=123)
    second = evaluate_greedy(agent, env, num_episodes=5, seed_base=123)
    assert first == second


def test_env_reset_seed_reproduces_chaos_moves():
    def play(seed):
        env = OrderAndChaosEnv(config={"chaos_mode": "random"})
        env.reset(seed=seed)
        boards = []
        for action in (0, 10, 20, 30):
            obs, *_ = env.step(action)
            boards.append(obs.copy())
        return boards

    for x, y in zip(play(5), play(5)):
        assert np.array_equal(x, y)
