# Order and Chaos

## Model

We will model the problem as a Markov Decision Problem since both players can see the complete state at all times.

## States

The state represents the current board configuration. Since we have 36 cells, each of which can be either *empty*, *X* or *O*, we have the following state space:

$$
S = \left\{(\text{row}, \text{col}, \text{symbol}) \;\middle|\; \text{row},\text{col}\in[0,5], \text{symbol}\in\{\emptyset, X, O\}\right\}
$$

## Actions

Since each player chooses not only the cell, but also the symbol, we define the set of actions as the product between the board coordinates and the symbol set.

Let $\mathcal C= \left\{ (r, c)\in\mathbb Z^2 \;\middle|\; 0\leq r, c \leq 5 \right\}$ and $\Sigma = \left\{X,O\right\}$

Then the action state is

$$
A = \mathcal C \times \Sigma = \left\{(r, c, \sigma)\;\middle|\;(r,c)\in\mathcal C,\;\sigma\in\Sigma\right\}
$$

that has a size of $|A| = |\mathcal C|\times|\Sigma| = 36\times 2 = 72$

A problem with this action space is that is is valid only on the first round. In the following rounds the set of possible actions reduces. One way to accound for this is to make the action space dependant on the time step. This approach has some other problems for the way we aim to solve the MDP, which is by using some deep reinforcement learning technique.

1. The output layer of a neural network is fixed in a standard neural network;
2. If we simply re-index the actions, "taking action 0" might have a different meaning at every turn.

### The "trick"

placeholder

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
