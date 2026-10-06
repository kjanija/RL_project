"""
tests/test_dqn.py

Unit tests for DQN with Action Masking and Experience Replay.
"""

import numpy as np
import pytest
import torch

from src.dqn import OrderQNetwork, ReplayBuffer, MaskedDQNAgent, encode_board


def test_board_encoding_shape_and_values():
    board = np.zeros((6, 6), dtype=np.int8)
    board[0, 0] = 1  # X
    board[0, 1] = 2  # O

    encoded = encode_board(board)
    assert encoded.shape == (3, 6, 6)
    # Channel 0: empty
    assert encoded[0, 0, 0] == 0.0
    assert encoded[0, 0, 1] == 0.0
    assert encoded[0, 1, 1] == 1.0

    # Channel 1: X
    assert encoded[1, 0, 0] == 1.0
    assert encoded[1, 0, 1] == 0.0

    # Channel 2: O
    assert encoded[2, 0, 0] == 0.0
    assert encoded[2, 0, 1] == 1.0


def test_q_network_forward_pass():
    net = OrderQNetwork(in_channels=3, action_dim=72)
    dummy_input = torch.zeros((4, 3, 6, 6), dtype=torch.float32)
    output = net(dummy_input)

    assert output.shape == (4, 72)
    assert not torch.isnan(output).any()


def test_replay_buffer_capacity_and_sampling():
    buffer = ReplayBuffer(capacity=5)
    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)
    dummy_mask = np.ones(72, dtype=bool)

    for i in range(10):
        buffer.push(dummy_state, action=i % 72, reward=float(i), next_state=dummy_state, done=False, next_mask=dummy_mask)

    assert len(buffer) == 5

    states, actions, rewards, next_states, dones, next_masks = buffer.sample(3, device=torch.device("cpu"))
    assert states.shape == (3, 3, 6, 6)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_masks.shape == (3, 72)


def test_action_masking_prevents_illegal_exploration():
    agent = MaskedDQNAgent(device="cpu")
    board = np.zeros((6, 6), dtype=np.int8)
    # Occupy all but 1 cell (cell 0,0)
    board[:] = 1
    board[0, 0] = 0

    mask = np.zeros(72, dtype=bool)
    mask[0] = True  # X at (0, 0)
    mask[1] = True  # O at (0, 0)

    # Under random exploration (epsilon = 1.0), it must only pick legal actions {0, 1}
    for _ in range(50):
        action = agent.select_action(encode_board(board), mask, epsilon=1.0)
        assert action in [0, 1], f"Illegal action {action} chosen during exploration!"


def test_action_masking_prevents_illegal_greedy_action():
    agent = MaskedDQNAgent(device="cpu")
    board = np.zeros((6, 6), dtype=np.int8)

    # Let cell (0, 0) be occupied
    mask = np.ones(72, dtype=bool)
    mask[0] = False  # X at (0, 0) illegal
    mask[1] = False  # O at (0, 0) illegal

    # Override network weights such that illegal action 0 has a massive bias
    with torch.no_grad():
        for param in agent.q_net.parameters():
            param.zero_()
        agent.q_net.fc2.bias[0] = 1e6   # Action 0 has highest unmasked value
        agent.q_net.fc2.bias[2] = 50.0  # Action 2 is highest legal value

    action = agent.select_action(encode_board(board), mask, epsilon=0.0)
    assert action == 2, f"Expected action 2 (highest legal), got action {action}"


def test_masked_target_computation_in_update():
    agent = MaskedDQNAgent(batch_size=2, device="cpu")
    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)

    # Fill buffer with 2 transitions
    mask_all = np.ones(72, dtype=bool)
    agent.replay_buffer.push(dummy_state, 0, 1.0, dummy_state, False, mask_all)
    agent.replay_buffer.push(dummy_state, 1, 0.0, dummy_state, True, mask_all)

    initial_loss = agent.update()
    assert isinstance(initial_loss, float)
    assert initial_loss >= 0.0
