# Dreamer4

A JAX implementation of [DreamerV4](https://arxiv.org/pdf/2509.24527) targeting [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) — the GPU-accelerated MuJoCo backend that runs physics entirely on-device via JAX.

By keeping both the environment and the learner in JAX, the full training loop (environment step → replay → world model update → actor-critic update) can run JIT-compiled end-to-end on GPU with no Python overhead per step.

---

## Architecture

Dreamer learns a compact latent world model from experience and trains an actor-critic entirely inside imagination — no environment interaction is needed during policy optimization.

```
Observations
     │
     ▼
┌──────────┐       ┌─────────────────────────────────────────────┐
│ Encoder  │──────▶│              World Model (RSSM)              │
└──────────┘       │                                              │
                   │  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  (GRU) │
                   │  z_t ~ q(z | h_t, x_t)              (post.) │
                   │  z_t ~ p(z | h_t)                   (prior) │
                   └───────────┬──────────────────────────────────┘
                               │ s_t = (h_t, z_t)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌─────────────┐  ┌──────────────┐
        │ Decoder  │   │   Reward    │  │   Continue   │
        └──────────┘   │  Predictor  │  │  Predictor   │
                       └─────────────┘  └──────────────┘

                  Imagination Rollouts
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐            ┌──────────────┐
        │  Actor   │            │    Critic    │
        │ (policy) │            │   (value)    │
        └──────────┘            └──────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Encoder** | MLP for low-dimensional state observations; CNN for pixel observations |
| **RSSM** | Recurrent State Space Model — GRU dynamics with straight-through categorical latents |
| **Decoder** | Reconstructs observations from the latent state (auxiliary reconstruction loss) |
| **Reward / Continue heads** | Symlog-transformed scalar reward prediction; binary episode-continuation prediction |
| **Actor** | Categorical or Gaussian policy trained on imagined trajectories |
| **Critic** | Value function with λ-return targets; slow EMA target network |

---

## Stack

| Library | Role |
|---------|------|
| [JAX](https://github.com/google/jax) | Autodiff, JIT compilation, hardware acceleration |
| [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/) | Neural network modules with mutable state |
| [Optax](https://optax.readthedocs.io/) | Optimizers and gradient transforms |
| [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) | Batched GPU-accelerated physics environments |

---

## Project Structure

```
Dreamer4/
├── main.py                  # Entry point / training script
├── pyproject.toml
├── data/
│   └── mnist.py             # MNIST loader (JAX training harness smoke-test)
└── src/
    └── dreamer4/
        └── networks/
            ├── __init__.py
            └── mlp.py       # Flax NNX MLP (shared backbone)
```

**Planned modules** (in progress):

```
src/dreamer4/
├── env/
│   └── mjx_env.py           # MJX environment wrapper (batched, JAX-native)
├── networks/
│   ├── mlp.py               # ✓ done
│   ├── cnn.py               # Encoder / decoder for pixel observations
│   └── rssm.py              # GRU-based recurrent state space model
├── agent/
│   ├── world_model.py       # World model training (KL + reconstruction + reward)
│   └── actor_critic.py      # Imagined-trajectory actor-critic
├── replay.py                # Flat replay buffer (JAX arrays, device-resident)
└── trainer.py               # Outer training loop: collect → update → log
```

---

## Installation

Requires Python ≥ 3.9 and a CUDA 12 GPU.

```bash
# Clone
git clone https://github.com/yourname/Dreamer4.git
cd Dreamer4

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install (editable, pulls JAX with CUDA 12 support)
pip install -e .
```

Verify JAX can see your GPU:

```bash
python -c "import jax; print(jax.devices())"
```

---

## Current Status

> **Early development.** The JAX training harness (optimizer, JIT-compiled train step, batch loop) is validated on MNIST. World model and MJX environment components are next.

- [x] Flax NNX MLP backbone
- [x] JAX training harness (Adam, JIT train step, batch loop)
- [ ] MJX environment wrapper
- [ ] RSSM (GRU + categorical latents)
- [ ] Encoder / decoder (MLP + CNN)
- [ ] Reward / continue predictors
- [ ] World model training (KL balancing, free nats, symlog)
- [ ] Actor-critic on imagined rollouts (λ-returns, entropy regularization)
- [ ] Replay buffer (device-resident)
- [ ] Logging (WandB / TensorBoard)

---

## Running the Smoke Test

The current `main.py` trains the MLP on MNIST to verify the JAX/Flax/Optax stack end-to-end:

```bash
python main.py
```

Expected output (10 epochs, ~98 % test accuracy):

```
epoch  1  loss 0.3421  test acc 0.9254
...
epoch 10  loss 0.0412  test acc 0.9801
```

---

## References

- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), 2023 — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Todorov et al., *MuJoCo: A physics engine for model-based control*, 2012
- MuJoCo MJX documentation — [mujoco.readthedocs.io/en/stable/mjx.html](https://mujoco.readthedocs.io/en/stable/mjx.html)
