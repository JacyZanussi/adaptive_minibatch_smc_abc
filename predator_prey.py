"""
Class for simulating LV.
"""

import numpy as np

class lotka_volterra:
    def __init__(self):
        pass

    def tau_leaping(X0,t_max,alpha,beta,gamma,delta,K=None,dt = 0.01):
        """
        alpha : prey reproduction rate
        beta : predation rate 
        gamma : predator death rate
        delta : predator reproduction rate
        """
        batch_size = X0.shape[0] 
        times = np.arange(0,t_max + dt,dt)
        nt = times.shape[0]
        X = np.zeros(shape=(batch_size,nt,2))
        X[:,0,:] = X0 # Initial conditions

        for i in range(nt-1):
            #Populations
            R = X[:,i,0]
            F = X[:,i,1]

            #Propensities
            pb = beta * R * F * dt
            pc = gamma * F * dt
            pd = delta * beta * R * F * dt

            #Draws
            qb = np.random.poisson(pb)
            qc = np.random.poisson(pc)
            qd = np.random.poisson(pd)

            if K == None:
                #If extinct, we freeze the values
                pa = alpha * R * dt
                qa = np.random.poisson(pa)
                extinct = (R < 1) | (F < 1)
                R_update = R + qa - qb
                F_update = F - qc + qd
                R_update[extinct] = R[extinct]
                F_update[extinct] = F[extinct]
            else:
                # Capacity caps prey growth in extinction events
                pa = np.maximum(alpha * R*(1 - R/K) * dt,0)
                qa = np.random.poisson(pa)
                R_update = R + qa - qb
                F_update = F - qc + qd
            X[:,i+1,0] = np.maximum(R_update,0)
            X[:,i+1,1] = np.maximum(F_update,0)
        return times,X

    def euler_maruyama(X0,t_max,alpha,beta,gamma,delta,sigma,dt = 0.01):
        nt = int(t_max/dt) + 1
        times = np.arange(0,t_max + dt,dt)
        batch_size = X0.shape[0]
        X = np.zeros((batch_size,nt,2))
        sqrtdt = np.sqrt(dt)

        #Initial conditions
        X[:,0,:] = X0
        for i in range(nt - 1):
            X_curr = X[:,i,:]
            vf = dX_dt(X_curr, alpha, beta, gamma, delta)
            W = np.random.normal(0,sqrtdt,size = (batch_size,2))
            X_next = np.maximum(X_curr  +  (vf * dt)  +  (sigma * X_curr * W),0)
            X[:,i+1,:] = X_next
        return times,X


def dX_dt(X,alpha,beta,gamma,delta):
    x = X[:,0]
    y = X[:,1]
    vf = np.stack([
        alpha * x - beta * x * y,
        -gamma * y + delta * beta * x * y
    ], axis=1)
    return vf
