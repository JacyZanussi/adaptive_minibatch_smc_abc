import numpy as np
import pickle as pkl
from smc_abc import smc_abc_iterator as abc_iter
import adaptive_smc_abc_schemes as schemes
import smc_abc_auxiliary as aux

import snapshots
smfish_sims = snapshots.snapshot_simulator()

# Basic parameters
sample_size = 300
kminus = 10
T,dt = 5,0.01


# dataset
with open('datasets/synthetic_smfish_data.pkl','rb') as f:
    data_tuple = pkl.load(f)
lengths = data_tuple[0]
sites = data_tuple[1]
data = data_tuple[2]

length_index = np.arange(sample_size)
def bndry_func(x,ind,i=length_index): # ind is the wrong kind of indices, these are internal to the smfish simulator.
    return (x > 0) & (x < lengths[i[ind]])

def model(p,i):
    z = sites[i,]
    bfunc = lambda x,ind: bndry_func(x,ind,i=i)
    obs = data[i]
    sim = smfish_sims.simulate(p[0], kminus, p[1], p[2], T, dt, z, bfunc)
    return sim, obs


# Prior
prior = aux.uniform_prior(np.array([[1.0,100.0],[1.0,100.0],[0.0,1.0]]))


# Stats Function
def stats_func0(x):
    counts = np.array([u.shape[0] for u in x], dtype=float)
    mean_counts = np.mean(counts)
    var_influence = (counts - mean_counts)**2
    rmsd = np.array([np.std(u) if u.size >=2 else 0 for u in x], dtype=float)
    stats = np.vstack((counts, var_influence, rmsd)).T
    return stats
def stats_func(x,y,sf = stats_func0):
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
hdpr_list = []
threshold_list = []
gens_list = []
for s in seeds:
    init_params.update({'seed':s})
    est = schemes.constant_init(init_params,p = [sample_size])
    est.counter = 0
    while continue_func(est):
        schemes.constant_loop(est)
        est.snr = (est.next_alpha_threshold**2)/(est.v_total_est/est.batch_size)
    hdpr_list.append(est.log_hdpr_product)
    threshold_list.append(est.next_alpha_threshold)
    gens_list.append(est.generation)


print(f"Quantiles of the log hdpr product: [0.1,0.25,0.5,0.75,0.9]: {np.quantile(hdpr_list, [0.1,0.25,0.5,0.75,0.9])}")
print(f"Quantiles of the final threshold: [0.1,0.25,0.5,0.75,0.9]: {np.quantile(threshold_list, [0.1,0.25,0.5,0.75,0.9])}")

print(f"Choice of common final threshold: 0.8: {np.quantile(threshold_list, 0.8)}")
print(f"Number of generations: {gens_list}")
