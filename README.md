# Order and Chaos

## Model

We will model the problem as a Markov Decision Problem since both players can see the complete state at all times.

## States

The state represents the current board configuration. Since we have 36 cells, each of which can be either *empty*, *X* or *O*, we have the following state space:

$$
S = \left\lbrace(\text{row}, \text{col}, \text{symbol}) \;\middle|\; \text{row},\text{col}\in[0,5], \text{symbol}\in\lbrace\emptyset, X, O\rbrace\right\rbrace
$$

## Actions

Since each player chooses not only the cell, but also the symbol, we define the set of actions as the product between the board coordinates and the symbol set.

Let $\mathcal C= \left\lbrace (r, c)\in\mathbb Z^2 \;\middle|\; 0\leq r, c \leq 5 \right\rbrace$ and $\Sigma = \left\lbrace X,O\right\rbrace$

Then the action state is

$$
A = \mathcal C \times \Sigma = \left\lbrace(r, c, \sigma)\;\middle|\;(r,c)\in\mathcal C,\;\sigma\in\Sigma\right\rbrace
$$

that has a size of $|A| = |\mathcal C|\times|\Sigma| = 36\times 2 = 72$

A problem with this action space is that is is valid only on the first round. In the following rounds the set of possible actions reduces. One way to accound for this is to make the action space dependant on the time step. This approach has some other problems for the way we aim to solve the MDP, which is by using some deep reinforcement learning technique.

1. The output layer of a neural network is fixed in a standard neural network;
2. If we simply re-index the actions, "taking action 0" might have a different meaning at every turn.

### The "trick"

To allow standard Deep Reinforcement Learning architectures (e.g., Actor-Critic in PPO or DQN) to operate effectively, two structural requirements must be met:
1. **Fixed Output Dimension:** The output layer of the policy network must have a constant dimension $|A| = 72$ regardless of how many squares are currently occupied.
2. **Spatial & Semantic Consistency:** Action index $a$ must consistently refer to the exact same square and symbol across all timesteps. If we were to dynamically re-index only the remaining empty squares, "action 0" would refer to a different cell on every turn, preventing neural layers from learning stable spatial representations.

#### Action Bijective Mapping
Every integer action $a \in \{0, 1, \dots, 71\}$ is deterministically mapped to and from a board move $(r, c, \sigma)$:

* **Decoding ($a \to (r, c, \sigma)$)**:
  $$
  p = \lfloor a / 2 \rfloor \in [0, 35], \quad r = \lfloor p / 6 \rfloor, \quad c = p \pmod 6
  $$
  $$
  \sigma = (a \pmod 2) + 1 \in \{1, 2\} \quad (\text{where } 1 = X, \; 2 = O)
  $$

* **Encoding ($(r, c, \sigma) \to a$)**:
  $$
  a = (6 \cdot r + c) \cdot 2 + (\sigma - 1)
  $$

In Python (`src/env.py`):
```python
position = action // 2      # 0 to 35
row = position // 6         # 0 to 5
col = position % 6          # 0 to 5
symbol = (action % 2) + 1   # 1 for X, 2 for O
```

---

#### Handling Depleted / Occupied Squares

As the game progresses, occupied cells become illegal moves. We support two distinct paradigms:

#### 1. Baseline: Environment Penalization & Episode Termination
In standard Gym environments, if the agent outputs an action $a$ pointing to an already occupied cell $(r, c)$, the environment:
1. Assigns a large negative penalty (`illegal_move_reward = -100`).
2. Terminates the episode immediately (`terminated = True`).

* **Limitation:** In early training, the agent wastes up to 90% of its samples crashing into occupied squares without ever reaching mid-game tactical states.

#### 2. Advanced: Invalid Action Masking (`action_masks()`)
Instead of allowing the agent to pick an illegal move and punishing it post-hoc, **Invalid Action Masking** prevents the policy network from sampling illegal actions in the first place.

The environment exposes an `action_masks()` method returning a boolean array $M \in \{0, 1\}^{72}$:
$$
M_a = \begin{cases} 
1 & \text{if cell } (\lfloor \lfloor a/2 \rfloor / 6 \rfloor, \lfloor a/2 \rfloor \pmod 6) \text{ is empty} \\ 
0 & \text{if occupied} 
\end{cases}
$$

During training (e.g., using `MaskablePPO` from `sb3-contrib`), the policy logits $\mathbf{z} \in \mathbb{R}^{72}$ are masked **before** the softmax operation:
$$
\tilde{z}_a = \begin{cases} z_a & \text{if } M_a = 1 \\ -\infty & \text{if } M_a = 0 \end{cases}
$$
$$
\pi(a \mid s) = \frac{\exp(\tilde{z}_a)}{\sum_{a'} \exp(\tilde{z}_{a'})}
$$

Because $\exp(-\infty) = 0$, the probability of sampling any occupied cell is strictly **zero**:
$$
\pi(a_{\text{illegal}} \mid s) = 0
$$

This ensures **100% of training steps explore valid game transitions**, drastically accelerating convergence and stabilizing the value network.

## Project Setup with uv

This project uses `uv` for fast Python environment and package management.

### Initializing the environment
To create the virtual environment and install dependencies from the lockfile:
```bash
uv sync
```

### Activating the virtual environment
Before running any code, make sure your virtual environment is active:
- **Linux/macOS:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

### Adding new packages
To install a new package (e.g., `stable-baselines3` or `gymnasium`) and add it to your project's dependencies:
```bash
uv add stable-baselines3
```

### Removing packages
To remove a package from your environment and `pyproject.toml`:
```bash
uv remove package_name
```

### Running scripts directly
You can also run scripts inside the virtual environment without explicitly activating it using:
```bash
uv run main.py
```

## Environment Configuration

The OrderAndChaosEnv accepts a configuration dictionary at initialization. This allows you to dynamically tune the reward shaping and penalty values during hyperparameter sweeps without ever having to modify the core environment code.

**Configurable Parameters**:

- `illegal_move_reward`: Penalty for attempting to play on an already occupied cell (Default: -100).
- `win_reward`: Massive reward for successfully placing 5 symbols in a row (Default: 100).
- `loss_reward`: Penalty assigned when the board fills up, resulting in a win for Chaos (Default: -100).
- `reward_3_in_a_row`: Intermediate shaping reward for extending an unbroken line to 3 (Default: 1).
- `reward_4_in_a_row`: Intermediate shaping reward for extending an unbroken line to 4 (Default: 5).
- `teacher_reward`: Intermediate reward granted if the agent's move perfectly matches the theoretical winning strategy (Default: 2).