from jax import Array
from flax import nnx


class MLP(nnx.Module):
    def __init__(
            self,
            in_dim: int,
            h_dim: int,
            out_dim: int,
            num_layers: int,
            activation: callable,
            rngs: nnx.Rngs):

        layers = []

        layers.append(nnx.Linear(in_dim, h_dim, rngs=rngs))
        layers.append(activation)

        for _ in range(num_layers):
            layers.append(nnx.Linear(h_dim, h_dim, rngs=rngs))
            layers.append(activation)

        layers.append(nnx.Linear(h_dim, out_dim, rngs=rngs))

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Array):
        return self.layers(x)