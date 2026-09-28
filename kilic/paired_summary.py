"""
Paired per-cell summaries for the E. coli lacZ data (used by `application_kilic.py`).

Observation i is one fixed cell with its own covariate, the time t_i at which it was imaged. A
minibatch is n indices drawn from the pool of N = 3833 cells; for each index the simulator runs
one independent cell of the telegraph model up to that cell's own t_i, and the simulated cell is
compared with the observed cell of the same index.

The per-cell summary row is the vector of cumulative count indicators 1{count <= m} for
m = 0, 1, 3, 7, 15 (bin edges 0, 1, 2, 4, 8, 16), placed in the block of the cell's fixation time,
so with 16 times a row has 80 entries and the batch mean of the rows is the batch's empirical
count distribution at each time.
"""
import atexit
import os

import numpy as np

import kilic_model as km
import kilic_data


# Step-cap accounting. A proposal in which any cell ran out of SSA steps is rejected; the counts
# are kept per process here (an imported module, so each joblib worker accumulates its own) and
# printed once per process at exit.
TRUNC = dict(model_calls=0, cells=0, truncated_calls=0, truncated_cells=0)
_WARN_FIRST = 3
_WARN_EVERY = 500


def record_truncation(n_cells, n_truncated_cells, params=None, max_steps=None, warn=True):
    """Record one `model` call. A call with any truncated cell is a rejected proposal."""
    TRUNC["model_calls"] += 1
    TRUNC["cells"] += int(n_cells)
    if n_truncated_cells:
        TRUNC["truncated_calls"] += 1
        TRUNC["truncated_cells"] += int(n_truncated_cells)
        if warn and (TRUNC["truncated_calls"] <= _WARN_FIRST
                     or TRUNC["truncated_calls"] % _WARN_EVERY == 0):
            where = "" if params is None else " at params=" + np.array2string(np.asarray(params), precision=4)
            cap = "" if max_steps is None else ", max_steps=%d" % max_steps
            print(f"[trunc] pid={os.getpid()} step cap hit: {n_truncated_cells}/{n_cells} cells"
                  f"{where}{cap} -> proposal rejected (process total {TRUNC['truncated_calls']} "
                  f"rejected of {TRUNC['model_calls']} proposals)", flush=True)


def truncation_report(where="exit", quiet_if_zero=True):
    if TRUNC["model_calls"] == 0:
        return
    if TRUNC["truncated_cells"] == 0 and quiet_if_zero:
        return
    print(f"[trunc-summary] pid={os.getpid()} {where}: {TRUNC['truncated_calls']} of "
          f"{TRUNC['model_calls']} proposals rejected for step-cap truncation "
          f"({TRUNC['truncated_cells']} of {TRUNC['cells']} simulated cells truncated)", flush=True)


atexit.register(truncation_report)


def load_pool(t_min=0.0):
    """
    The pooled per-cell observations, times t <= t_min dropped (t = 0 by default).

    Returns
    -------
    obs_t : (K,) float        the time points kept
    counts : list of K arrays the observed counts at each time point
    time_idx : (N,) int       time-point index of each pooled observation
    obs_counts : (N,) int     count of each pooled observation
    """
    ts_all, counts_all = kilic_data.load_kilic_fig4()
    keep = ts_all > max(0.0, t_min)
    obs_t = ts_all[keep]
    counts = [c for c, k in zip(counts_all, keep) if k]
    time_idx = np.concatenate([np.full(c.size, k, dtype=np.int64) for k, c in enumerate(counts)])
    obs_counts = np.concatenate(counts).astype(np.int64)
    return obs_t, counts, time_idx, obs_counts


def bin_of(c, edges):
    """Bin index of each count (counts below edges[0] are clipped into bin 0)."""
    return np.clip(np.searchsorted(edges, np.asarray(c), side="right") - 1, 0, edges.size - 1)


def cumulative_rows(time_idx, counts, edges, n_times, dtype=np.float64):
    """
    (n, n_times * (B - 1)) rows of cumulative indicators, B = len(edges).

    Within its own time block a cell contributes 1{bin_of(count) <= b} for b = 0 .. B - 2, the
    empirical CDF of its count at the bin upper edges (the always-true last indicator is dropped).
    Column of (time k, indicator b) is k * (B - 1) + b. The entries are 0 or 1, so int8 storage
    gives the same batch means and variance estimates as float64 at an eighth of the memory.
    """
    time_idx = np.asarray(time_idx, dtype=np.int64)
    B = int(np.asarray(edges).size)
    n_keep = B - 1
    b = bin_of(counts, edges)
    rows = np.zeros((time_idx.size, int(n_times) * n_keep), dtype=dtype)
    base = time_idx * n_keep
    for j in range(n_keep):
        hit = np.flatnonzero(b <= j)
        rows[hit, base[hit] + j] = 1
    return rows


def build_W_inv_rows(rows, shrink=0.1, ridge=1e-8):
    """
    Mahalanobis weight W^{-1} from the full observed row matrix (N, S): the covariance with its
    correlation shrunk toward the identity by `shrink`, a zero standard deviation set to 1, and a
    `ridge` on the diagonal.

    Returns (W_inv, diagnostics dict).
    """
    X = np.asarray(rows, dtype=float)
    cov = np.cov(X.T, ddof=1)
    d = cov.shape[0]
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    n_zero_std = int((std == 0).sum())
    std = np.where(std == 0, 1.0, std)
    corr = cov / np.outer(std, std)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    corr = (1 - shrink) * corr + shrink * np.eye(d)
    W = np.outer(std, std) * corr + ridge * np.eye(d)
    W_inv = np.linalg.inv(W)
    ev = np.linalg.eigvalsh(W)
    diag = dict(shrink=float(shrink), ridge=float(ridge), n_stats=int(d),
                n_zero_std_cols=n_zero_std, raw_cov_rank=int(np.linalg.matrix_rank(cov)),
                W_cond=float(ev.max() / ev.min()) if ev.min() > 0 else np.inf)
    return W_inv, diag


def simulate_paired(params_full, batch_time_idx, obs_t, max_steps, seed, G=2):
    """
    Simulate one independent cell per entry of `batch_time_idx`, each to that entry's own time.

    Cells sharing a time point are simulated in one `kilic_model.simulate` call, so a batch costs
    one call per distinct time in it, and the simulated horizon is sum_i t_i rather than n max_i t_i.

    Returns (counts, n_truncated): `counts` is float and holds NaN for any cell whose trajectory
    ran out of `max_steps` before its sample time.
    """
    tk = np.asarray(batch_time_idx, dtype=np.int64)
    out = np.empty(tk.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for k in np.unique(tk):                 # sorted, so the seeds are deterministic given tk
        sel = np.flatnonzero(tk == k)
        t = float(obs_t[k])
        sub_seed = int(rng.integers(0, 2 ** 31 - 1))
        _, m = km.simulate(params_full, G=G, K=int(sel.size), t_max=t,
                           sample_times=np.array([t]), max_steps=int(max_steps), seed=sub_seed)
        out[sel] = np.asarray(m)[:, 0]
    return out, int(np.isnan(out).sum())


DEFAULT_V_TOTAL_MAX_BYTES = 256 * 1024 ** 2


def make_v_total_estimator(obs_rows, max_bytes=DEFAULT_V_TOTAL_MAX_BYTES, report=False):
    """
    Estimator of v_total for `est.estimate_v_total`, for the paired rows.

    It computes the pooled within-batch covariance of the per-cell differences,

        Sigma = (1 / (P (n - 1))) sum_i sum_j (r_ij - rbar_i) (r_ij - rbar_i)^T ,
        r_ij = simulated row - observed row of the same cell,  rbar_i = batch mean of particle i,

    and v_total = est.esv(Sigma), without building the (P, n, S) float64 stack of observed rows
    that the estimator in `smc_abc` uses (2.4 GB at P = 1000, n = 3833, S = 80). When the float64
    working array fits in `max_bytes`, the operations are those of the `smc_abc` estimator, so the
    result is identical. Otherwise particles are processed in chunks and Sigma is accumulated as
    sum_i D_i^T D_i - (1/n) sum_i s_i s_i^T, with s_i the column sums of D_i; the entries of D are
    -1, 0 or 1, so the sums are exact in float64 and do not depend on the chunk size.

    Parameters
    ----------
    obs_rows : (N_pool, S) array
        The observed rows, the array passed as `data`.
    max_bytes : int
        Budget for the float64 working array.
    report : bool
        Print which path was taken, once per generation.
    """
    obs = np.asarray(obs_rows)

    def estimate_v_total(est):
        sim = est.posterior_stats
        P = int(est.num_particles)
        reps = int(est.reps)
        S = int(sim.shape[-1])
        idx = est.batch_indices[est.posterior_indices]          # (P, n) observed indices used
        n = int(est.batch_size)
        nb = P * n * reps * S * 8
        if nb <= max_bytes:
            ref = obs[idx]
            if reps > 1:
                ref = np.repeat(ref, repeats=reps, axis=1)
            delta = np.subtract(sim, ref, dtype=np.float64)      # (P, n*reps, S)
            if reps > 1:
                bm = np.mean(delta.reshape(P, n, reps, S), axis=2)
            else:
                bm = delta.reshape(P, n, S)
            bm -= bm.mean(axis=1, keepdims=True)
            Sigma = np.einsum('ijk,ijl->kl', bm, bm) / (P * (n - 1))
            path, work = "direct", nb
        else:
            per = max(1, int(max_bytes // max(1, n * reps * S * 8)))
            G = np.zeros((S, S))
            T = np.zeros((S, S))
            for a in range(0, P, per):
                b = min(a + per, P)
                D = np.subtract(sim[a:b], obs[idx[a:b]] if reps == 1
                                else np.repeat(obs[idx[a:b]], repeats=reps, axis=1),
                                dtype=np.float64)
                if reps > 1:
                    D = np.mean(D.reshape(b - a, n, reps, S), axis=2)
                D2 = D.reshape((b - a) * n, S)
                G += D2.T @ D2
                s = D.reshape(b - a, n, S).sum(axis=1)
                T += s.T @ s
                del D, D2, s
            Sigma = (G - T / n) / (P * (n - 1))
            path, work = "chunked(%d)" % per, per * n * reps * S * 8
        est.Sigma = Sigma
        est.v_total_est = est.esv(Sigma)
        if report:
            print(f"[v_total] {path}: P={P} n={n} S={S} "
                  f"stats dtype={np.asarray(sim).dtype} ({np.asarray(sim).nbytes / 2**20:.0f} MB), "
                  f"working float64 {work / 2**20:.0f} MB", flush=True)

    return estimate_v_total
