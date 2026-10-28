"""Data splitting / normalisation and the training loop (works for any Model)."""
import pickle
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from chemistry_emulator.model.model import count_params
from chemistry_emulator.model import Model


@dataclass
class Data:
    """Arrays + split indices + normalisation statistics (computed on the train set only)."""
    X: np.ndarray
    Y: np.ndarray
    TID: np.ndarray
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    x_mu: np.ndarray
    x_sd: np.ndarray
    y_mu: np.ndarray
    y_sd: np.ndarray
    Xn_j: jnp.ndarray      # normalised inputs, on the device
    Yn_j: jnp.ndarray      # normalised targets, on the device


def prepare(X, Y, TID, seed=0, frac_train=0.7, frac_val=0.15):
    """Split BY TRAJECTORY (rows of one trajectory never straddle train/test), then normalise."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(TID))
    n_tr, n_va = int(frac_train * len(ids)), int(frac_val * len(ids))
    tr = np.where(np.isin(TID, ids[:n_tr]))[0]
    va = np.where(np.isin(TID, ids[n_tr:n_tr + n_va]))[0]
    te = np.where(np.isin(TID, ids[n_tr + n_va:]))[0]
    print(f"rows: train {len(tr)}, val {len(va)}, test {len(te)}")

    x_mu, x_sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    y_mu, y_sd = Y[tr].mean(0), Y[tr].std(0) + 1e-8
    return Data(X, Y, TID, tr, va, te, x_mu, x_sd, y_mu, y_sd,
                jnp.asarray((X - x_mu) / x_sd), jnp.asarray((Y - y_mu) / y_sd))


def train(model: Model, 
          data: Data, 
          epochs=150, 
          batch=512, 
          lr=1e-3, 
          weight_decay=1e-4, 
          seed=0, 
          verbose=True
          ):
    """AdamW + cosine decay; returns the parameters of the epoch with the best validation loss."""
    params = model.init_params(jax.random.PRNGKey(seed))
    if verbose:
        print(f"{model.name}: {count_params(params):,} params")

    steps = max(1, len(data.tr) // batch)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs * steps), weight_decay=weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(model.loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    val_loss = jax.jit(model.loss_fn)
    rng = np.random.default_rng(seed)
    best_val, best_params, best_epoch = np.inf, params, 0
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(data.tr)
        losses = []
        for i in range(steps):
            b = perm[i * batch:(i + 1) * batch]
            params, opt_state, loss = step(params, opt_state, data.Xn_j[b], data.Yn_j[b])
            losses.append(loss)
        v = float(val_loss(params, data.Xn_j[data.va], data.Yn_j[data.va]))
        if v < best_val:
            best_val, best_params, best_epoch = v, params, epoch
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  train {float(np.mean(losses)):.5f}  val {v:.5f}  "
                  f"(best {best_val:.5f} @ {best_epoch})  {time.time() - t0:.0f}s")
    return best_params


def save_run(path, 
             params, 
             data: Data, 
             extra=None):
    """Save trained parameters + normalisation stats (Colab sessions are temporary: copy to Drive)."""
    with open(path, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params),
                     "x_mu": data.x_mu, "x_sd": data.x_sd, 
                     "y_mu": data.y_mu, "y_sd": data.y_sd,
                     "extra": extra}, 
                     f)


def load_run(path):
    with open(path, "rb") as f:
        return pickle.load(f)
