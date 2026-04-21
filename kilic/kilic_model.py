"""
Kilic gene expression model (Gillespie SSA)

Parameters are ordered as:
    [K_12, K_21, ..., beta_1, ..., beta_G, delta]

For G=2:
    params = [K_12, K_21, beta_1, beta_2, delta]
"""

import numpy as np
from numba import njit


@njit
def simulate(params, G, K, t_max, sample_times, max_steps=10_000):
    """
    Stochastic gene expression simulation (Gillespie SSA).

    Parameters
    ----------
    params : array
        [K_ij..., beta_1...beta_G, delta]
    G : int
        Number of promoter states
    K : int
        Number of cells (trajectories)
    t_max : float
        Maximum simulation time
    sample_times : array
        Times at which to record observations
    max_steps : int
        Maximum SSA steps per trajectory

    Returns
    -------
    t_out : (K, n_samples) array
        Times of last reaction before each sample
    m_out : (K, n_samples) array
        mRNA counts at each sample time
    """

    # --- unpack parameters ---
    n_switch = G * (G - 1)

    K_rates = np.zeros((G, G))
    idx = 0
    for i in range(G):
        for j in range(G):
            if i != j:
                K_rates[i, j] = params[idx]
                idx += 1

    beta = params[n_switch:n_switch + G]
    delta = params[-1]

    n_samples = sample_times.shape[0]

    t_out = np.full((K, n_samples), np.nan)
    m_out = np.full((K, n_samples), np.nan)

    # --- simulate each cell ---
    for cell in range(K):

        t = 0.0
        m = 0
        s = 0  # initial promoter state

        sample_idx = 0
        t_sample = sample_times[0]

        for _ in range(max_steps):

            if t >= t_max:
                break

            # reaction rates
            rates = np.zeros(G + 2)

            for j in range(G):
                if j != s:
                    rates[j] = K_rates[s, j]

            rates[G] = beta[s]       # transcription
            rates[G + 1] = delta * m # degradation

            total_rate = rates.sum()
            if total_rate <= 0.0:
                break

            # advance time
            t_last = t
            m_last = m
            t += np.random.exponential(1.0 / total_rate)

            # choose reaction
            r = np.random.rand() * total_rate
            cum = 0.0
            for i in range(G + 2):
                cum += rates[i]
                if r < cum:
                    if i < G:
                        s = i
                    elif i == G:
                        m += 1
                    else:
                        if m > 0:
                            m -= 1
                    break

            # record samples
            while sample_idx < n_samples and t >= t_sample:
                t_out[cell, sample_idx] = t_last
                m_out[cell, sample_idx] = m_last
                sample_idx += 1
                if sample_idx < n_samples:
                    t_sample = sample_times[sample_idx]

    return t_out, m_out
