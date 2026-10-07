"""
Deep Q-Network implementation with invalid action maskinx, experience replay, and Mean Squared Bellman Error minimization for Order's policy

Framed for Sample-Based Approximate Value Iteration:
    - Onlne Q-network Q_theta and target Q-network Q_theta-
    - Experience Replay Buffer providing empirical distribution for L2 metric projection
    - Masked action selection for illegal moves
    - Masked Bellman optimality target computation
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def encode_board(board: np.ndarray) -> np.ndarray:
    """
    Converts 6x6 board into a 3-channel binary tensor of shape (3,6,6)
        - channel 0 for empty or occupied
        - channel 1 for X or otherwise
        - channel 2 for O or otherwise
    """
    board_np = np.asarray(board, dtype=np.int8)
    encoded = np.zeros((3, 6, 6), dtype=np.float32)
    encoded[0] = (board_np == 0).astype(np.float32)
    encoded[1] = (board_np == 1).astype(np.float32)
    encoded[2] = (board_np == 2).astype(np.float32)
    return encoded


class OrderQNetwork(nn.Module):
    """
    Convolutional Q-network for Order
    Preserves spatial locality to identify lines of 3, 4, and 5.
    3 conv layers (3 -> 64 -> 128 -> 128) expand receptive field to 7x7,
    spanning the entire 6x6 board in a single pass.
    Input shape: (b, 3, 6, 6)
    Output shape: (b, 72)
    """

    def __init__(
        self, in_channels: int = 3, action_dim: int = 72, hidden_dim: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # start_dim=1 preserves batch dimension 0 and flattens (128, 6, 6) -> (4608,) per sample
        h = torch.flatten(h, start_dim=1)
        h = F.relu(self.fc1(h))
        q_values = self.fc2(h)
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer providing empirical transition distribution
    Stores transitions: (state, action, reward, next_state, done, next_mask)
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        next_masks_t = torch.tensor(
            np.array(next_masks), dtype=torch.bool, device=device
        )

        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t

    def trim_to_size(self, max_retained: int = 20000) -> None:
        """
        Cleanses obsolete transitions across curriculum stages by keeping
        only the max_retained most recent transitions.
        """
        if max_retained > 0 and len(self.buffer) > max_retained:
            recent = list(self.buffer)[-max_retained:]
            self.buffer.clear()
            self.buffer.extend(recent)

    def __len__(self) -> int:
        return len(self.buffer)


class MaskedDQNAgent:
    """
    DQN Agentwith invalidaction masking

    Implements sample-based approximate value iteration
        - theta: online network parameters
        - theta^-: target network parameters
        - action masking
        - loss: mean squared bellman error
    """

    def __init__(
        self,
        in_channels: int = 3,
        action_dim: int = 72,
        lr: float = 3e-4,
        gamma: float = 0.90,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        target_update_freq: int = 1000,
        device: str = "cpu",
        double_dqn: bool = False,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.double_dqn = double_dqn
        self.step_count = 0

        # networks
        self.q_net = OrderQNetwork(in_channels=in_channels, action_dim=action_dim).to(
            self.device
        )
        self.target_net = OrderQNetwork(
            in_channels=in_channels, action_dim=action_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # MSBE
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(
        self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0.0
    ) -> int:
        """
        Epsilon-greedy action selection restricted to legal actions
            - random exploration: uniform sampling from legal actions
            - greedy exploitation: argmax_{a legal} Q(s,a)
        """
        legal_actions = np.where(mask)[0]
        if len(legal_actions) == 0:
            return 0

        if random.random() < epsilon:
            return int(np.random.choice(legal_actions))

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0)

        # mask invalid actions with large negative value (-1e9 avoids 0 * inf = NaN)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        masked_q = q_values.clone()
        masked_q[~mask_tensor] = -1e9

        return int(torch.argmax(masked_q).item())

    def update(self) -> float:
        """
        Performs one gradient step solving empirical L2 metric projection.
        Supports both Standard DQN (default) and Double DQN (if double_dqn=True).
        """

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks = (
            self.replay_buffer.sample(self.batch_size, self.device)
        )

        # Q_theta(s,a)
        # Q_theta(s, a): actions.unsqueeze(1) creates (B, 1) indices for gather; squeeze(1) restores (B,)
        q_values = self.q_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman optimality target computation
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN:
                # 1. Action selection using ONLINE network Q_theta (with legal action masking)
                online_next_q = self.q_net(next_states)
                online_next_q[~next_masks] = -1e9
                best_next_actions = torch.argmax(online_next_q, dim=1, keepdim=True)

                # 2. Action evaluation using TARGET network Q_theta-
                target_next_q = self.target_net(next_states)
                max_next_q = target_next_q.gather(1, best_next_actions).squeeze(1)
            else:
                # Standard DQN (default):
                next_q = self.target_net(next_states)
                # Mask invalid next-state actions with -1e9 (avoids 0.0 * inf = NaN)
                next_q[~next_masks] = -1e9
                max_next_q, _ = torch.max(next_q, dim=1)

            # Guard against edge-case where next state was full
            max_next_q = torch.clamp(max_next_q, min=-1e8)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "double_dqn": self.double_dqn,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", 0)
        if "double_dqn" in checkpoint:
            self.double_dqn = bool(checkpoint["double_dqn"])
