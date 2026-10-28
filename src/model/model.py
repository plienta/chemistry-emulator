import jax
import jax.numpy as jnp
import numpy as np


def init_mlp(key, sizes_of_layers):
    layers = []
    keys = jax.random.split(key, len(sizes_of_layers) - 1)
    for i in range(len(sizes_of_layers) - 1):
        n_in, n_out = sizes_of_layers[i], sizes_of_layers[i + 1]
        W = jax.random.normal(keys[i], (n_in, n_out)) * np.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        layers.append({"W": W, "b": b})
    return layers


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jax.nn.silu(x @ layer["W"] + layer["b"])
    return x @ layers[-1]["W"] + layers[-1]["b"]


def count_params(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


class Model:
    """Base class: subclasses define init_params and forward."""
    name = "model"

    def init_params(self, key):
        raise NotImplementedError

    def forward(self, params, x):
        raise NotImplementedError

    def loss_fn(self, params, x, y):
        return jnp.mean((self.forward(params, x) - y) ** 2)

    def n_params(self, key=None):
        key = jax.random.PRNGKey(0) if key is None else key
        return count_params(self.init_params(key))