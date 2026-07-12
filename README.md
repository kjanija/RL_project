# Order and Chaos

## Model

We will model the problem as a Markov Decision Problem since both players can see the complete state at all times.

## States

The state represents the current board configuration. Since we have 36 cells, each of which can be either *empty*, *X* or *O*, we have the following state space:

$$
S = \Big\{(\text{row}, \text{col}, \text{symbol}) \;|\; \text{row},\text{col}\in[0,5], \text{symbol}\in\{\emptyset, X, O\}\Big\}
$$

## Actions

Since each player chooses not only the cell, but also the symbol, we define the set of actions as the product between the board coordinates and the symbol set.

Let $\mathcal C= \Big\{ (r, c)\in\mathbb Z^2 \;|\; 0\leq r, c \leq 5 \Big\}$ and $\Sigma = \Big\{X,O\Big\}$

Then the action state is

$$
A = \mathcal C \times \Sigma = \Big\{(r, c, \sigma)\;|\;(r,c)\in\mathcal C,\;\sigma\in\Sigma\Big\}
$$

that has a size of $|A| = |\mathcal C|\times|\Sigma| = 36\times 2 = 72$

A problem with this action space is that is is valid only on the first round. In the following rounds the set of possible actions reduces. One way to accound for this is to make the action space dependant on the time step. This approach has some other problems for the way we aim to solve the MDP, which is by using some deep reinforcement learning technique.

1. The output layer of a neural network is fixed in a standard neural network;
2. If we simply re-index the actions, "taking action 0" might have a different meaning at every turn.

### The "trick"

placeholder
