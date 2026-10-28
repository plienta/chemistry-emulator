import jax
import jax.numpy as jnp
from model.model import Model, init_mlp, mlp
import numpy as np

class MLP(Model):
  def __init__(self, hidden=128, n_hidden=9, n_in=5, n_out=2):
    self.sizes = [n_in] + [hidden] * n_hidden + [n_out]
    self.name = f"MLP_h{hidden}_d{n_hidden}"

  def init_params(self, key):
    return {"net": init_mlp(key, self.sizes)}

  def forward(self, params, x):
    return mlp(params["net"], x)