from flax import nnx
from typing import Any, Callable, Sequence, Optional
import jax.numpy as jnp
from jax import Array
import dataclasses 

# Type alias for any activation function mapping an array to an array.
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]

class MLP(nnx.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        in_dim: int,
        hidden_layers: Sequence[int],
        activation: ActivationFn,
        layer_norm: bool = False,
        activate_final: bool = False,
        rngs: Optional[nnx.Rngs] = None,
        out_dim: Optional[int] = None,
        ):
        if rngs is None:
            raise ValueError("rngs must be provided.")
        
        # Build full size sequence: [in_dim, h0, h1, ..., hn]
        layer_sizes = [in_dim] + list(hidden_layers)

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nnx.Linear(layer_sizes[i], layer_sizes[i+1], rngs = rngs))
            layers.append(activation)
            if layer_norm:
                layers.append(nnx.LayerNorm(layer_sizes[i+1], rngs = rngs))


        # Output projection is optional — omit to keep last hidden layer as output
        if out_dim is not None:
            layers.append(nnx.Linear(layer_sizes[-1], out_dim, rngs=rngs))

        if activate_final:
            layers.append(activation)

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Array):
        return self.layers(x)