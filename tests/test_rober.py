"""Cheap sanity checks. Run with:  python -m pytest -q   (pip install -e ".[dev]")"""
import numpy as np

from robertson.rober import (K_REF, make_direct_data, make_step_data, rober,
                                              solve_trajectory)


def test_rober_derivative_at_known_point():
    # r1 = 0.02, r2 = 0.003, r3 = 0.05  ->  [-r1 + r3, r1 - r2 - r3, r2]
    d = rober(0.0, [0.5, 1e-5, 0.5], K_REF)
    assert np.allclose(d, [0.03, -0.033, 0.003])


def test_derivatives_sum_to_zero_and_y3_never_decreases():
    rng = np.random.default_rng(0)
    for _ in range(100):
        y = rng.dirichlet([1, 1, 1])
        d = rober(0.0, y, K_REF)
        assert abs(sum(d)) < 1e-9 * max(1.0, np.abs(d).max())
        assert d[2] >= 0.0


def test_canonical_reference_values():
    # standard Robertson reference solution (Hairer & Wanner), to ~3 digits
    sol = solve_trajectory((K_REF, np.array([1.0, 0.0, 0.0]), np.array([40.0, 4000.0])))
    assert np.allclose(sol[0], [0.7158, 9.19e-6, 0.2842], rtol=2e-3)
    assert np.allclose(sol[1], [0.1832, 8.94e-7, 0.8168], rtol=2e-3)


def test_direct_data_shapes_and_no_floor():
    X, Y, TID = make_direct_data(n_traj=6, n_target_times=8, seed=1)
    assert X.shape == (48, 4) and Y.shape == (48, 3) and TID.shape == (48,)
    assert (Y > np.log10(1e-20)).all()
    assert np.abs((10 ** Y).sum(1) - 1).max() < 1e-4


def test_step_data_shapes_and_no_floor():
    X, Y, TID = make_step_data(n_traj=6, n_pairs=8, seed=1)
    assert X.shape == (48, 4) and Y.shape == (48, 3)
    assert (X[:, :3] > np.log10(1e-20)).all()
