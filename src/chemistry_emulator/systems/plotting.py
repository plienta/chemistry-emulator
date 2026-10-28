"""Plots for looking at the data and the results (matplotlib)."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua: one colour per species
GRID = "#e1e0d9"


def sketch_direct_data(X, Y, TID, n_traj=12, names=("y1", "y2", "y3")):
    """One panel per species (log y vs log t, one thin line per trajectory) + the start states.

    Expects the direct-data layout X = [log y1(0), log y2(0), log y3(0), log t].
    Look for: y1 falling, y3 rising, trajectories merging onto one curve, nothing stuck at -20.
    """
    ids = np.unique(TID)[:n_traj]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, ax in enumerate(axes.flat[:3]):
        for i in ids:
            m = TID == i
            ax.plot(X[m, 3], Y[m, j], color=COLORS[j], lw=1.5, alpha=0.7, marker="o", ms=2.5)
        ax.set_title(f"log10 {names[j]}(t)", loc="left")
        ax.set_xlabel("log10 t")
        ax.grid(True, color=GRID)
    ax = axes.flat[3]
    first = np.unique(TID, return_index=True)[1]
    ax.scatter(X[first, 1], X[first, 2], s=18, color=COLORS[0])
    ax.set_title("start states", loc="left")
    ax.set_xlabel("log10 y2(0)")
    ax.set_ylabel("log10 y3(0)")
    ax.grid(True, color=GRID)
    fig.tight_layout()
    return fig


def plot_rollout_error(times, rel, names=("y1", "y2", "y3")):
    """Relative error of a rollout against the solver, per species, versus step number."""
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(len(times))
    for j, n in enumerate(names):
        ax.plot(steps, 100 * rel[:, j], color=COLORS[j], lw=2, label=n)
    ax.set_xlabel("step")
    ax.set_ylabel("relative error (%)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.grid(True, color=GRID)
    ax.legend()
    fig.tight_layout()
    return fig
