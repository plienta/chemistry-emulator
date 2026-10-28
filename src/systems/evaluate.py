"""Metrics: direct-emulator error, one-step error (with a copy-the-state baseline), rollouts."""
import jax
import jax.numpy as jnp
import numpy as np

FLOOR = 1e-20
NAMES = ("y1", "y2", "y3")


def _summary(rel, dex, names):
    return {n: {"mean": rel[:, j].mean(), "median": np.median(rel[:, j]),
                "p90": np.percentile(rel[:, j], 90), "gt100": (rel[:, j] > 1).mean(),
                "mae_dex": dex[:, j].mean()} for j, n in enumerate(names)}


def _print(title, summ):
    print(f"\n{title}")
    for n, s in summ.items():
        print(f"  {n}: mean {100 * s['mean']:8.2f}% | median {100 * s['median']:7.3f}% "
              f"| 90th {100 * s['p90']:8.3f}% | >100%: {100 * s['gt100']:.2f}% "
              f"| MAE {s['mae_dex']:.4f} dex")


def evaluate_direct(model, params, data, idx=None, names=NAMES, check_sum=True, verbose=True):
    """Direct emulator: targets are log y(t). Errors are relative (linear space) and in dex.

    Prefer median / 90th percentile: the mean is dominated by a few tiny-value rows.
    check_sum: also report |sum(y) - 1| (valid for ROBER, where the species sum to 1).
    """
    idx = data.te if idx is None else idx
    pred_log = np.asarray(model.forward(params, data.Xn_j[idx])) * data.y_sd + data.y_mu
    true_log = data.Y[idx]
    pred, true = 10 ** pred_log, 10 ** true_log
    summ = _summary(np.abs(pred - true) / np.maximum(true, FLOOR), np.abs(pred_log - true_log), names)
    if verbose:
        _print(f"{model.name} - direct, test set", summ)
    if check_sum:
        s = np.abs(pred.sum(1) - 1)
        summ["sum_err"] = {"median": np.median(s), "p90": np.percentile(s, 90)}
        if verbose:
            print(f"  |sum(y) - 1|: median {summ['sum_err']['median']:.2e}, 90th {summ['sum_err']['p90']:.2e}")
    return summ


def evaluate_step(model, params, data, idx=None, names=NAMES, verbose=True):
    """One-step emulator: targets are the change in log y. Compared with the copy-the-state baseline
    (predict y(t+dt) = y(t)); the model has to beat it clearly to have learned anything."""
    idx = data.te if idx is None else idx
    delta = np.asarray(model.forward(params, data.Xn_j[idx])) * data.y_sd + data.y_mu
    log_ya = data.X[idx, :len(names)]
    true = 10 ** (log_ya + data.Y[idx])
    out = {}
    for label, pred in [("model", 10 ** (log_ya + delta)), ("copy y(t)", 10 ** log_ya)]:
        out[label] = _summary(np.abs(pred - true) / np.maximum(true, FLOOR),
                              np.abs(np.log10(np.maximum(pred, FLOOR)) - np.log10(np.maximum(true, FLOOR))), names)
        if verbose:
            _print(f"{model.name} - one step, {label}, test set", out[label])
    return out


def rollout(model, params, data, y0, dt, n_steps, floor=FLOOR, log_max=1.0):
    """Apply the one-step model n_steps times, feeding its own output back in. Returns (n_steps+1, n_species)."""
    fwd = jax.jit(model.forward)
    y = np.array(y0, dtype=np.float64)
    out = [y.copy()]
    for _ in range(n_steps):
        log_y = np.log10(np.maximum(y, floor))
        x = np.concatenate([log_y, [np.log10(dt)]]).astype(np.float32)
        xn = jnp.asarray((x - data.x_mu) / data.x_sd)[None, :]
        delta = np.asarray(fwd(params, xn))[0] * data.y_sd + data.y_mu
        y = 10 ** np.clip(log_y + delta, np.log10(floor), log_max)   # clip: a diverged rollout stays finite
        out.append(y.copy())
    return np.array(out)


def rollout_vs_solver(model, params, data, solve_fn, rate_consts, y0, dt, n_steps):
    """Roll out and compare with the reference solver at the same times.

    solve_fn: e.g. systems.rober.solve_trajectory.  Returns times, pred, true, rel_err (all (n_steps+1, ...)).
    """
    times = dt * np.arange(n_steps + 1)
    true = solve_fn((rate_consts, np.asarray(y0, dtype=float), times))
    pred = rollout(model, params, data, y0, dt, n_steps)
    rel = np.abs(pred - true) / np.maximum(true, FLOOR)
    return times, pred, true, rel
