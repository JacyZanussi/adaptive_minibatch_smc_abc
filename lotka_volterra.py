import numpy as np
from numba import njit

@njit
def tau_leaping(X0, t_max, alpha, beta, gamma, delta, dt=0.01, seed=-1, lower_bound=1.0):
    if seed >= 0:
        np.random.seed(seed)
    batch_size = X0.shape[0]
    times = np.arange(0, t_max + dt, dt)
    nt = times.shape[0]
    X = np.zeros((batch_size, nt, 2))
    X[:, 0, :] = X0
    for i in range(nt - 1):
        for b in range(batch_size):
            R = X[b, i, 0]
            F = X[b, i, 1]
            # Freeze at lower bound
            # if R <= lower_bound or F <= lower_bound:
            #     X[b, i + 1, 0] = R
            #     X[b, i + 1, 1] = F
            #     continue
            # Propensities
            pa = alpha * R * dt
            pb = beta * R * F * dt
            pc = gamma * F * dt
            pd = delta * beta * R * F * dt
            # Poisson draws
            qa = np.random.poisson(pa)
            qb = np.random.poisson(pb)
            qc = np.random.poisson(pc)
            qd = np.random.poisson(pd)
            # Cap consuming reactions so population can't go below lower_bound
            qb = min(qb, int(R - lower_bound))
            qc = min(qc, int(F - lower_bound))
            # Updates
            X[b, i + 1, 0] = R + qa - qb
            X[b, i + 1, 1] = F - qc + qd
    return times, X


# @njit
# def tau_leaping(X0, t_max, alpha, beta, gamma, delta, dt=0.01, seed=-1):
#     if seed >= 0:
#         np.random.seed(seed)

#     batch_size = X0.shape[0]
#     times = np.arange(0, t_max + dt, dt)
#     nt = times.shape[0]
#     X = np.zeros((batch_size, nt, 2))
#     X[:, 0, :] = X0

#     for i in range(nt - 1):
#         for b in range(batch_size):
#             R = X[b, i, 0]
#             F = X[b, i, 1]

#             # Check extinction first — freeze state and skip all propensities
#             if R < 1.0 or F < 1.0:
#                 X[b, i + 1, 0] = R
#                 X[b, i + 1, 1] = F
#                 continue

#             # Propensities (no carrying capacity)
#             pa = max(alpha * R * dt, 0.0)
#             pb = max(beta * R * F * dt, 0.0)
#             pc = max(gamma * F * dt, 0.0)
#             pd = max(delta * beta * R * F * dt, 0.0)

#             # Poisson draws
#             qa = np.random.poisson(pa)
#             qb = np.random.poisson(pb)
#             qc = np.random.poisson(pc)
#             qd = np.random.poisson(pd)

#             # Updates
#             X[b, i + 1, 0] = max(R + qa - qb, 0.0)
#             X[b, i + 1, 1] = max(F - qc + qd, 0.0)

#     return times, X