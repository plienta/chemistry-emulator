import jax
import jax.numpy as jnp
from model.model import Model, init_mlp, mlp
import numpy as np

class DeepONet(Model):
  def __init__(self, latent=128, hidden=128, n_hidden=9, n_out=2, n_branch_in=4):
    self.latent, self.n_out, self.n_branch_in = latent, n_out, n_branch_in
    self.branch_sizes = [n_branch_in] + [hidden] * n_hidden + [n_out * latent]
    self.trunk_sizes  = [1] + [hidden] * n_hidden + [latent]
    self.name = f"DeepONet_k{latent}_h{hidden}_d{n_hidden}"

  def init_params(self, key):
    kb, kt = jax.random.split(key)
    return {"branch": init_mlp(kb, self.branch_sizes),
            "trunk":  init_mlp(kt, self.trunk_sizes),
            "bias":   jnp.zeros(self.n_out)}

  def forward(self, params, x):
    p, t = x[:, :self.n_branch_in], x[:, self.n_branch_in:self.n_branch_in + 1]
    b = mlp(params["branch"], p).reshape(-1, self.n_out, self.latent)
    f = mlp(params["trunk"], t)
    return jnp.einsum("bok,bk->bo", b, f) / np.sqrt(self.latent) + params["bias"]