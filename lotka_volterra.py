"""
Class for simulating LV.
"""

import numpy as np
from numba import njit


@njit
def tau_leaping(X0, t_max, alpha, beta, gamma, delta, K=-1.0, dt=0.01):
    batch_size = X0.shape[0]
    times = np.arange(0, t_max + dt, dt)
    nt = times.shape[0]
    X = np.zeros((batch_size, nt, 2))
    X[:, 0, :] = X0 

    for i in range(nt - 1):
        # Explicitly loop over the batch for Numba compatibility
        for b in range(batch_size):
            R = X[b, i, 0]
            F = X[b, i, 1]

            # Propensities (Scalars)
            pb = max(beta * R * F * dt, 0.0)
            pc = max(gamma * F * dt, 0.0)
            pd = max(delta * beta * R * F * dt, 0.0)

            # Draws (Numba needs scalars here)
            qb = np.random.poisson(pb)
            qc = np.random.poisson(pc)
            qd = np.random.poisson(pd)

            if K < 0: # Using -1.0 as the 'None' sentinel
                if R < 1.0 or F < 1.0:
                    X[b, i+1, 0] = R
                    X[b, i+1, 1] = F
                    continue
                pa = max(alpha * R * dt, 0.0)
            else:
                pa = max(alpha * R * (1.0 - R/K) * dt, 0.0)
            
            qa = np.random.poisson(pa)

            # Updates
            X[b, i+1, 0] = max(R + qa - qb, 0.0)
            X[b, i+1, 1] = max(F - qc + qd, 0.0)

    return times, X

