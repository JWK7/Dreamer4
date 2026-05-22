import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from dreamer_v4.networks import MLP
from data.mnist import load_mnist_data

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3


@nnx.jit
def train_step(model: MLP, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
    def loss_fn(model):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def accuracy(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = model(x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)


def make_batches(xs, ys, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(len(xs))
    xs, ys = xs[idx], ys[idx]
    for start in range(0, len(xs) - batch_size + 1, batch_size):
        yield xs[start:start + batch_size], ys[start:start + batch_size]


def main():
    train, test = load_mnist_data()
    train_x = np.stack([x for x, _ in train])
    train_y = np.array([y for _, y in train], dtype=np.int32)
    test_x  = np.stack([x for x, _ in test])
    test_y  = np.array([y for _, y in test], dtype=np.int32)

    rngs = nnx.Rngs(0)
    model = MLP(in_dim=784, hidden_layers=[256,256], out_dim=10,
                activation=jax.nn.relu, rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt= nnx.Param)

    rng = np.random.default_rng(42)
    for epoch in range(1, EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for bx, by in make_batches(train_x, train_y, BATCH_SIZE, rng):
            loss = train_step(model, optimizer, jnp.array(bx), jnp.array(by))
            total_loss += float(loss)
            n_batches += 1

        test_acc = accuracy(model, jnp.array(test_x), jnp.array(test_y))
        print(f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  "
              f"test acc {float(test_acc):.4f}")


if __name__ == "__main__":
    main()
