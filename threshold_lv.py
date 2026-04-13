import numpy as np
import pickle as pkl
from smc_abc import smc_abc_iterator as abc_iter
import adaptive_smc_abc_schemes as schemes
import smc_abc_auxiliary as aux

#plotting
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
matplotlib.use("Agg")
matplotlib.rcParams['mathtext.fontset'] = 'stix'


# dataset
with open('datasets/lv_stochastic.pkl','rb') as f:
    dataset = pkl.load(f)
data,ic,true_params = list(dataset)
alpha,beta,gamma,delta = list(true_params)
sample_size = data.shape[0]


### Model
t_max = 10
dt = 0.02

def lv_tau_leaping(X0,t_max,params,dt = 0.02,K=1000):
    alpha,beta,gamma,delta = list(params)
    batch_size = X0.shape[0]
    times = np.arange(0,t_max + dt,dt)
    n_time_points = times.shape[0]
    X = np.zeros(shape=(batch_size,n_time_points,2))
    X[:,0,:] = X0 # Initial conditions

    for i in range(n_time_points-1):
        #Propensities:
        R = X[:,i,0]
        F = X[:,i,1]

        #Propensities
        pa = alpha * R * dt
        pe = alpha * R**2 * dt / K
        pb = beta * R * F * dt
        pc = gamma * F * dt
        pd = delta * beta * R * F * dt

        #Draws
        qa = np.random.poisson(pa)
        qe = np.random.poisson(pe)
        qb = np.random.poisson(pb)
        qc = np.random.poisson(pc)
        qd = np.random.poisson(pd)

        R_update = R + qa - qb - qe
        F_update = F - qc + qd
        X[:,i+1,0] = np.maximum(R_update,0)
        X[:,i+1,1] = np.maximum(F_update,0)

    return times,X

def model(p,Bi):
    params = list(p) + [1] #[1] is delta
    obs = data[Bi,:,:]
    _,sim = lv_tau_leaping(ic[Bi,:],t_max,params,dt = 0.02)
    return sim, obs


# Prior
prior = aux.uniform_prior(np.array([[0.0,28.0],[0.0,0.04],[0.0,28.0]]))


# Stats Function
def stats_func_lv(x, trunc_ind=200):
    r = x[:, trunc_ind:, 0]
    f = x[:, trunc_ind:, 1]

    mu_r = np.mean(r, axis=1)
    mu_f = np.mean(f, axis=1)

    # Use Log Coefficient of Variation to break collinearity with the mean
    # std / mean provides a 'unitless' measure of noise
    cv_r = np.log(np.std(r, axis=1) / (mu_r + 1e-5) + 1e-5)

    # Correlation (already unitless, keep as is)
    def get_row_corr(a, b):
        ma = a - np.mean(a, axis=1, keepdims=True)
        mb = b - np.mean(b, axis=1, keepdims=True)
        num = np.sum(ma * mb, axis=1)
        den = np.sqrt(np.sum(ma**2, axis=1) * np.sum(mb**2, axis=1))
        return num / (den + 1e-8)

    corr_rf = get_row_corr(r, f)

    # ACF_1 is excellent for alpha, and is also unitless
    r_t0, r_t1 = r[:, :-1], r[:, 1:]
    acf_r = get_row_corr(r_t0, r_t1)

    # 5 Statistics total
    features = np.column_stack([
        np.log(mu_r + 1), # Log-transform means to compress scale
        np.log(mu_f + 1),
        cv_r,
        corr_rf,
        acf_r
    ])

    return features

def stats_func(x,y,sf = stats_func_lv):
    return sf(x),sf(y)


init_params = {
    'data':data,
    'model':model,
    'stats_func':stats_func,
    'prior':prior,
    'num_particles':500,
    'cores':4,
    'alpha':0.5,
    'batch_size_min':2,
    'seed':0
}


# Convergence criteria
def continue_func(est):
    if est.generation <= 1:
        return True
    else:
        ar_criterion = est.acceptance_rate > 0.02
        snr_criterion = est.snr > 2
        return ar_criterion and snr_criterion

seeds = [i for i in range(10)]
threshold_list = []
gens_list = []
for s in seeds:
    init_params.update({'seed':s})
    est = schemes.constant_init(init_params,p = [sample_size])
    est.counter = 0
    while continue_func(est):
        schemes.constant_loop(est)
        if est.generation == 1:
            last_noise = est.v_total_est / est.batch_size
            last_epsilon = est.next_alpha_threshold
            last_snr = (last_epsilon**2)/last_noise
        else:
            est.snr = (est.next_alpha_threshold**2)/(est.v_total_est/est.batch_size)
            print(f"SNR: {est.snr}")
    threshold_list.append(est.next_alpha_threshold)
    gens_list.append(est.generation)

print(f"Quantiles of the final threshold: 0.8: {np.quantile(threshold_list, 0.8)}")
print(f"Number of generations: {gens_list}")
