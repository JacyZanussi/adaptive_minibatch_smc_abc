"""
Apply adaptive minibatch SMC ABC and constant minibatch SMC ABC
to the Kilic gene expression model.
"""

# =========================
# Imports
# =========================

import time
import pickle as pkl
import warnings
import numpy as np
import numba as nb
import scipy.io as sio
from numba import njit
import tracemalloc
import psutil
import os
import sys
process = psutil.Process(os.getpid())


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors

import smc_abc_schemes as schemes
import kilic_model as km

satid = int(sys.argv[1])



# =========================
# Data loading
# =========================

data_Wang = sio.loadmat("datasets/Wang2020_SFig21_2_tf_1800_K_18_J_2000_5e-03_3e-02_0e+00_2e-01_5e-03.mat", simplify_cells=True)

data = data_Wang["r_ct"].T[:, 1:]
true_params = data_Wang["ground"]["rates"]
obs_times = np.asarray(data_Wang["data"]["obs_t"])

del data_Wang

# =========================
# Model definition
# =========================

sample_times = obs_times
idx_sets = [[i] for i in range(17)]
gamma = true_params[-1]


def model(params, batch_idx, _data=data,_gamma=gamma,_st=sample_times):
    """Simulator wrapper used by SMC-ABC."""
    params = np.hstack((params, _gamma))
    _, sim = km.simulate(
        params,
        G=2,
        K=batch_idx.shape[0],
        sample_times=_st,
        t_max=1800,
        max_steps=400,
    )
    return np.asarray(sim)[:, 1:], _data[batch_idx]


# =========================
# Summary statistics
# =========================


@nb.njit()
def stats_func_wang(x):
    N_batch, N_data = x.shape
    # Compute column means (over batch axis), ignoring NaNs
    col_means = np.zeros(N_data)
    col_counts = np.zeros(N_data)
    for j in range(N_data):
        for i in range(N_batch):
            if not np.isnan(x[i, j]):
                col_means[j] += x[i, j]
                col_counts[j] += 1.0
        if col_counts[j] > 0:
            col_means[j] /= col_counts[j]

    # Build output: [x_nonan | xvar_infl], shape (N_batch, 2*N_data)
    out = np.empty((N_batch, 2 * N_data))
    for i in nb.prange(N_batch):
        for j in range(N_data):
            val = 0.0 if np.isnan(x[i, j]) else x[i, j]
            out[i, j] = val
            out[i, N_data + j] = (val - col_means[j]) ** 2
    return out

def stats_func(x, y):
    """Wraps `stats_func_wang` into the (sim, ref) -> (sim_stats, ref_stats) contract expected by smc_abc_iterator."""
    return stats_func_wang(x), stats_func_wang(y)


# =========================
# Prior (log10 multivariate lognormal)
# =========================

mu = np.array([-1, -1, -1, -1])
Sigma = np.diag([0.5] * 4) ** 2
prior_domain = np.repeat([[0, np.inf]], 4, axis=0)

seed = 42
mu = np.ascontiguousarray(mu, dtype=np.float64)
Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
base_rng = np.random.default_rng(seed) if seed is not None else None
def prior_func(rng=None):
    """Draw params as 10**z where z ~ N(mu, Sigma) (i.e. a log10-multivariate-lognormal prior)."""
    if rng is not None:
        z = rng.multivariate_normal(mu, Sigma)
    elif base_rng is not None:
        z = base_rng.multivariate_normal(mu, Sigma)
    else:
        z = np.random.multivariate_normal(mu, Sigma)
    return 10.0 ** z


def density(x, mu=mu, Sigma=Sigma):
    """Density of the log10-multivariate-lognormal prior at x (0 for any non-positive component)."""
    x = np.atleast_2d(x).astype(float)
    valid = np.all(x > 0, axis=1)
    logpdf = np.full(x.shape[0], -np.inf)

    if not np.any(valid):
        return logpdf

    logx = np.log10(x[valid])
    diff = logx - mu

    L = np.linalg.cholesky(Sigma)
    y = np.linalg.solve(L, diff.T)
    quad = np.sum(y ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    log_norm = (
        -0.5 * len(mu) * np.log(2 * np.pi)
        - 0.5 * logdet
        - len(mu) * np.log(np.log(10.0))
    )
    log_jac = -np.sum(np.log(x[valid]), axis=1)

    logpdf[valid] = log_norm + log_jac - 0.5 * quad
    return np.exp(logpdf)

prior = [prior_func, prior_domain, density]

# =========================
# SMC-ABC configuration
# =========================

INIT_PARAMS = dict(
    data=data,
    model=model,
    stats_func=stats_func,
    prior=prior,
    num_particles=1000,
    cores=4,
    low_mem=True,
    seed=42
)


TIME_LIMIT = 2 * 60 * 60

def extract_tracking(est):
    """Per-generation diagnostics saved for this run (separate dict/keys from experiments.get_info's attr_list)."""
    return dict(
        generation=est.generation,
        total_time=est.total_time,
        total_sims=est.total_sims,
        acceptance=est.acceptance_rate,
        ESS=est.ESS,
        batch_size=est.batch_size,
        log_hdpr_product=est.log_hdpr_product,
        alpha_threshold=est.alpha_threshold,
        snr=est.snr,
        noise=est.alpha_threshold**2 / (est.v_total_est / est.batch_size),
        hdpr=est.hdpr_marginal(),
    )


# =========================
# Adaptive minibatch SMC-ABC
# =========================
if satid == 0:
    tracked = {}
    est = schemes.fvc_init(INIT_PARAMS, p=[2, None, 1.0])

    tracemalloc.start()
    while est.total_time < TIME_LIMIT:
        schemes.fvc_loop(est, ess_resample=False)
        tracked[est.generation] = extract_tracking(est)
        est.ESS_resample(proportion=0.5)
        current,peak = tracemalloc.get_traced_memory()
        print('current(MB): ',current/1024**2, 'peak(MB) :', peak/1024**2, ' total rss: ',process.memory_info().rss / 1e9, "GB")

    with open("kilic/ecoli_slowgrowth_fvc.pkl", "wb") as f:
        pkl.dump((tracked, est.posterior, est.weights), f)

    del est
    tracemalloc.reset_peak()


# =========================
# Constant minibatch SMC-ABC
# =========================
if satid == 1:
    tracked = {}
    est = schemes.constant_init(INIT_PARAMS, p=[100])

    while est.total_time < TIME_LIMIT:
        schemes.constant_loop(est, ess_resample=False)
        tracked[est.generation] = extract_tracking(est)
        est.ESS_resample(proportion=0.5)
        current,peak = tracemalloc.get_traced_memory()
        print('current(MB): ',current/1024**2, 'peak(MB) :', peak/1024**2, ' total rss: ',process.memory_info().rss / 1e9, "GB")

    with open("kilic/ecoli_slowgrowth.pkl", "wb") as f:
        pkl.dump((tracked, est.posterior, est.weights), f)

