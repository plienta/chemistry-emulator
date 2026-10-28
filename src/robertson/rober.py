"""ROBER (Robertson 1966): three-species stiff chemical kinetics benchmark.

Reactions and what each one does to the species (net):
    1:  y1       -> y2        rate r1 = k1 * y1
    2:  y2 + y2  -> y3 + y2   rate r2 = k2 * y2**2
    3:  y2 + y3  -> y1 + y3   rate r3 = k3 * y2 * y3      (y3 is a catalyst)

    dy1/dt = -r1 + r3
    dy2/dt =  r1 - r2 - r3
    dy3/dt =  r2

Standard values: k = (0.04, 3e7, 1e4), y(0) = (1, 0, 0). y1 + y2 + y3 = 1 always.

Two kinds of training data are generated (rate constants are fixed to K_REF):
  * direct: (start state, t)        -> log y(t)
  * step:   (state at t_a, dt)      -> log y(t_a + dt) - log y(t_a)
Both use the layout  X = [log y1, log y2, log y3, log time-like]  (4 columns).
"""
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp

K_REF = np.array([0.04, 3.0e7, 1.0e4])   # standard rate constants
T_MIN, T_MAX = 1e-5, 1e5                 # time range of the direct data
T_STATE_MIN = 1e-6                       # earliest state time used for step data
DT_MIN, DT_MAX = 1e-5, 1e2               # step sizes for step data (choose from your use case)
ZERO_FRAC = 0.15                         # share of step pairs that start at t = 0
FLOOR = 1e-20                            # smallest value allowed before log10
LOGY_MIN, LOGY_MAX = -10.0, -2.0         # range of log10 y2(0), log10 y3(0); y1(0) = 1 - y2 - y3
RTOL, ATOL = 1e-8, 1e-12                 # solver tolerances (1e-10/1e-14 is ~3x slower)


def rober(t, current_states, rate_consts):
    r_const_1, r_const_2, r_const_3 = rate_consts
    y1, y2, y3 = current_states

    # reaction rates of each equation
    r_react_1 = r_const_1 * y1
    r_react_2 = r_const_2 * y2 * y2
    r_react_3 = r_const_3 * y2 * y3

    # rate of change of each species
    # decrease
    r_change_species_1 = -r_react_1 + r_react_3
    r_change_species_2 = r_react_1 - r_react_2 - r_react_3
    r_change_species_3 = r_react_2

    return [r_change_species_1, r_change_species_2, r_change_species_3]


def solve_trajectory(args):
    """Solve ONE trajectory (one set of rate constants + one start state) at all `times`.

    args = (rate_consts, start_state, times); `times` sorted, may start at 0.
    Returns an array (n_times, 3), or None if the solver failed.
    """
    rate_consts, start_state, times = args
    sol = solve_ivp(rober, 
                    (0.0, times[-1]), 
                    start_state, 
                    method="BDF",
                    t_eval=times, 
                    args=(rate_consts,), 
                    rtol=RTOL, atol=ATOL)
    if not sol.success:
        return None
    return sol.y.T


def _solve_all(jobs, n_workers):
    if n_workers > 1:
        with Pool(n_workers) as pool:
            return pool.map(solve_trajectory, jobs, chunksize=16)
    return [solve_trajectory(job) for job in jobs]


def sample_start_states(rng, n):
    """(n, 3) random start states"""
    y2 = 10 ** rng.uniform(LOGY_MIN, LOGY_MAX, n)
    y3 = 10 ** rng.uniform(LOGY_MIN, LOGY_MAX, n)
    y1 = 1 - y2 - y3
    return np.stack([y1, y2, y3], axis=1)


def make_direct_data(n_traj, n_target_times, seed=0, n_workers=1):
    """Rows: [log y(0) (3), log t] -> [log y(t) (3)].  Returns X, Y, TID."""
    rng = np.random.default_rng(seed)
    start_states = sample_start_states(rng, n_traj)
    target_times = np.sort(
        10 ** rng.uniform(np.log10(T_MIN), np.log10(T_MAX), (n_traj, n_target_times)), axis=1)

    jobs = [(K_REF, start_states[i], target_times[i]) for i in range(n_traj)]
    sols = _solve_all(jobs, n_workers)

    X, Y, TID = [], [], []
    for i, s in enumerate(sols):
        if s is None:
            continue
        p = np.log10(np.maximum(start_states[i], FLOOR))
        X.append(np.hstack([np.tile(p, (n_target_times, 1)), np.log10(target_times[i])[:, None]]))
        Y.append(np.log10(np.maximum(s, FLOOR)))
        TID.append(np.full(n_target_times, i))
    print(f"solved {len(X)}/{n_traj} trajectories")
    return (np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32),
            np.concatenate(TID))
