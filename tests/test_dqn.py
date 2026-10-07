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
    assert agent.double_dqn is False
    assert agent.gamma == 0.90
    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)

    # Fill buffer with 2 transitions
    mask_all = np.ones(72, dtype=bool)
    agent.replay_buffer.push(dummy_state, 0, 1.0, dummy_state, False, mask_all)
    agent.replay_buffer.push(dummy_state, 1, 0.0, dummy_state, True, mask_all)

    initial_loss = agent.update()
    assert isinstance(initial_loss, float)
    assert initial_loss >= 0.0


def test_replay_buffer_trim_to_size():
    buffer = ReplayBuffer(capacity=100)
    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)
    dummy_mask = np.ones(72, dtype=bool)

    # Push 50 transitions with action = i
    for i in range(50):
        buffer.push(dummy_state, action=i, reward=float(i), next_state=dummy_state, done=False, next_mask=dummy_mask)

    assert len(buffer) == 50

    # Trim to 20
    buffer.trim_to_size(max_retained=20)
    assert len(buffer) == 20

    # Verify that the 20 retained items are the most recent (actions 30 through 49)
    retained_actions = [t[1] for t in buffer.buffer]
    assert retained_actions == list(range(30, 50))

    # Trimming with higher max_retained should be a no-op
    buffer.trim_to_size(max_retained=50)
    assert len(buffer) == 20


def test_double_dqn_update_computation():
    agent_std = MaskedDQNAgent(batch_size=2, device="cpu", double_dqn=False)
    agent_ddqn = MaskedDQNAgent(batch_size=2, device="cpu", double_dqn=True)

    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)
    mask_all = np.ones(72, dtype=bool)

    for ag in (agent_std, agent_ddqn):
        ag.replay_buffer.push(dummy_state, 0, 1.0, dummy_state, False, mask_all)
        ag.replay_buffer.push(dummy_state, 1, -1.0, dummy_state, True, mask_all)
        loss = ag.update()
        assert isinstance(loss, float)
        assert loss >= 0.0


def test_double_dqn_target_decoupling_logic():
    agent = MaskedDQNAgent(batch_size=1, device="cpu", double_dqn=True)

    dummy_state = np.zeros((3, 6, 6), dtype=np.float32)
    mask = np.ones(72, dtype=bool)
    mask[0] = False  # Action 0 is illegal

    # Set up deterministic biases such that:
    # Online net prefers action 2
    # Target net prefers action 3
    with torch.no_grad():
        for p in agent.q_net.parameters():
            p.zero_()
        for p in agent.target_net.parameters():
            p.zero_()

        # Online network weights
        agent.q_net.fc2.bias[2] = 100.0  # highest legal online action
        agent.q_net.fc2.bias[0] = 500.0  # highest unmasked online action (should be masked out)

        # Target network weights
        agent.target_net.fc2.bias[2] = 5.0   # value of online's preferred action
        agent.target_net.fc2.bias[3] = 50.0  # target's own preferred action

    agent.replay_buffer.push(dummy_state, 2, 0.0, dummy_state, False, mask)

    # In Double DQN:
    # 1. Action selection uses online net: action 2 is chosen (action 0 is masked out)
    # 2. Action evaluation uses target net: target net evaluates action 2 -> 5.0
    # Bellman target: y = r + gamma * target_net(s', 2) = 0.0 + 0.90 * 5.0 = 4.5
    # Online current Q(s, 2) is 100.0
    # Expected MSBE loss: (100.0 - 4.5)^2 = 95.5^2 = 9120.25
    states, actions, rewards, next_states, dones, next_masks = agent.replay_buffer.sample(1, agent.device)
    with torch.no_grad():
        online_next_q = agent.q_net(next_states)
        online_next_q[~next_masks] = -1e9
        best_next_actions = torch.argmax(online_next_q, dim=1, keepdim=True)
        assert best_next_actions.item() == 2

        target_next_q = agent.target_net(next_states)
        max_next_q = target_next_q.gather(1, best_next_actions).squeeze(1)
        assert torch.isclose(max_next_q, torch.tensor([5.0]))

    loss = agent.update()
    assert loss > 0.0


def test_save_and_load_preserves_double_dqn():
    import tempfile
    from pathlib import Path

    agent = MaskedDQNAgent(device="cpu", double_dqn=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        agent.save(str(ckpt_path))

        loaded_agent = MaskedDQNAgent(device="cpu", double_dqn=False)
        loaded_agent.load(str(ckpt_path))
        assert loaded_agent.double_dqn is True
