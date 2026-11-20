# Policy Gradients in PyTorch

<p align="center">
  <img src="https://gymnasium.farama.org/_images/lunar_lander.gif" width="280"/>
</p>

REINFORCE, REINFORCE with baseline and A2C, written from scratch in PyTorch and compared on LunarLander-v3. Works with any Gymnasium environment that has discrete actions.

## Setup

```bash
git clone https://github.com/buuuuulat/pytorch_policy_gradients.git
cd pytorch_policy_gradients
pip install torch gymnasium numpy
```

## Usage

```bash
python3 algorithms/reinforce.py
python3 algorithms/a2c.py
```

Hyperparameters and the environment are set inside the scripts.

## Algorithms

### REINFORCE

Play a full episode, compute discounted returns, normalize them, then take a gradient step on the loss:

$$R_t = \sum_{i=0}^{T-t} \gamma^i r_{t+i}$$

$$L = -\sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t)\, R_t$$

### REINFORCE with baseline

Same as above, but the mean return is subtracted first: $A_t = R_t - b$, where $b = \mathbb{E}[R]$. This lowers the variance of the gradient.

### A2C

One shared MLP with two heads: a categorical policy and a value estimate $V(s)$.

- Advantage: $A_t = R_t - V(s_t)$
- Loss: `policy_loss + value_coef * value_loss - entropy_coef * entropy`
- Returns are bootstrapped from the critic when an episode is cut off, and from `0` when it terminates.
- `UNROLL_STEPS > 0` updates every few steps instead of waiting for the episode to end.

With a single environment A2C does not converge reliably. The vectorized version is in the [drafts branch](https://github.com/buuuuulat/pytorch_policy_gradients/tree/drafts).

### Entropy bonus

```python
entropy = -(probs * log_probs).sum(dim=1).mean()
```

It is subtracted from the loss to keep the policy from collapsing too early.

## Results

All runs are on LunarLander-v3.

Most of the time the baseline makes little difference:

![REINFORCE vs baseline](data/graph1.png)

On some runs it converges about twice as fast and reaches a higher reward:

![REINFORCE vs baseline, better run](data/graph2.png)

The entropy bonus (yellow) gives a much higher mean reward and gets there almost twice as fast over 5000 episodes:

![Entropy bonus](data/graph3.png)

A2C with 8 parallel environments does clearly better than both REINFORCE versions:

![A2C](data/graph4.png)

## Notes

The code is not optimized for speed. Buffers are Python lists, tensors are often converted to NumPy and back, and there is no CUDA support.
