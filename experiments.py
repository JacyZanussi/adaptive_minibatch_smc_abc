'''
Experimental Setups

Below are setup functions that take in a list of parameters and configure them to run experiments.
They follow the ipynb contents relatively closely.
The parameters are decomposed into two types: hyperparameters and experiment parameters.
Hyperparameters tell us information like seeding, adaptive threshold quantiles, and number of particles which should stay constant across experiments.
Experiment parameters are ones that are varied.
Here, the experimental parameters are basically restricted to heterogeneity perturbations, physical perturbations, and SMC ABC method hyperparameters like minibatch size.

Usage:
make list of dictionaries for experimental parameters.
run the function with **dictionary. 
Should save results to a pickle file. 
Results: Dictionary with keys of varied parameters and values of the results of the experiment.



use:
import...
parameter_sweep = dict...
experiment(parameter_sweep)

experiment then handles the sweep over parameters, and the dictionary keys are used to name the output parameters.
We save a pickled structure from SMC ABC that should probably be like.....
parameters
- list of replicates
  - dictionary of gathered values for each replicate.
So, we could see something like: a constant smc abc sweep over minibatch sample sizes [[n]]
to obtain the total time for replicate 1 for the last generation for parameter set [32], we would have
to obtain total time at the last generation for replicate 1 for parameter set [32],
results[(32,)][0][-1]['total_time'] # I think it has to be a tuple to work for dicitonaries 
Then we have data processing functions that do stuff like:
turn this into a non-uniform matrix of times:
times[[n]] = matrix(potentially not uniform) of total times across generations


Data that we want to track: 
(generation) 
total sims
total time
alpha threshold
acceptance rate
v estimate
batch size

'''

## Import smc abc class and schemes
import smc_abc_schemes as schemes
import smc_abc_utils as utils

import numpy as np
import pickle
import time
from itertools import product

from numba import njit

from matplotlib import pyplot as plt

#A class that generates smfish transcriptional dynamics snapshots
import lotka_volterra as lv
import transcriptional_dynamics as td


##### Models
## Lotka Volterra
def lotka_volterra_step(
    seed=None, 
    #a = 4, b = 0.01, g = 4, d = 1, #Physical parameters
    physical_params = [4,0.01,4], d = 1,
    input_filename = None, heterogeneities = [999], #Heterogeneities 
    scheme = 'constant', # Scheme
    scheme_params = [2], # Scheme hyperparameter
    stop_func = None,
    sample_size = 128, T = 5, dt = 0.01, #simulation parameters (base)
    alpha = 0.5, num_particles = 500, cores = 4, #estimator parameters
    prior_domain = [[0.0,10.0],[0.0,0.05],[0.0,10.0]],
    method = 'constant',
    **kwargs
):
    a,b,g = list(physical_params)
    rng = np.random.default_rng(seed)

    ### Data generation or loading:
    if input_filename is None:
        print(f"Simulating data with parameters: {physical_params} and heterogeneities: {heterogeneities}")
        ## Heterogeneity - uniformly distributed initial conditions centered around 500
        ic_range = heterogeneities[0]
        assert (ic_range >= 1) and (ic_range < 1000)
        shift = 500 - (ic_range - 1)/2
        ic = rng.choice(ic_range,(sample_size,2)) + shift
        _,data_temp = lv.tau_leaping(ic,T,a,b,g,d,dt = dt, seed=seed if seed is not None else -1)
    else:
        print(f"Loading data from: {input_filename}")
        with open(input_filename,'rb') as f:
            lv_dat = pickle.load(f) #third argument is ground truth parameters
            data_temp = lv_dat['data']
            ic = lv_dat['ic']
    data = stats_func_lv(data_temp)

    def stats_func(x,y,sf=stats_func_lv):
        return x,y

    #s_data = stats_func_lv(data)
    N_stats = data.shape[1]
    def model(p, Bi, num_reps=3, seed=None):
        alpha, beta, gamma = p
        s_sim = np.zeros((Bi.shape[0],N_stats))
        rep_rng = np.random.default_rng(seed) if seed is not None else None
        for _ in range(3):
            rep_seed = int(rep_rng.integers(np.iinfo(np.int64).max)) if rep_rng is not None else -1
            _, sim = lv.tau_leaping(ic[Bi], T,alpha, beta, gamma, d, dt=dt, seed=rep_seed)
            s_sim += stats_func_lv(sim)
        s_sim /= num_reps
        s_obs = data[Bi]
        return s_sim, s_obs
    ### Prior
    prior = utils.uniform_prior(prior_domain, seed=seed)

    ### Scheme
    if scheme == 'constant':
        init = schemes.constant_init
        loop = schemes.constant_loop
    elif scheme == 'fvc':
        init = schemes.fvc_init
        loop = schemes.fvc_loop
    else:
        raise ValueError(f"Scheme {scheme} not identified.")
    
    ######## This makes a dictionary of information shared with the adaptive minibatch schemes
    init_params = {
        'data':data,
        'model':model,
        'stats_func':stats_func,
        'prior':prior,
        'num_particles':num_particles,
        'cores':cores,
        'alpha':alpha,
        'seed':seed
    }
    stop = default_stop if stop_func is None else stop_func

    #Maybe make this a function since it'll be repeated code...
    results_list = []
    est = init(init_params,scheme_params)
    attr_list = info_dict[method]
    while not stop(est):
        loop(est)
        results = get_info(est,attr_list)
        results_list.append(results)

    return results_list

# Summary Statistics: predator prey model
@njit
def batch_corr_single(a, b, eps=1e-8):
    """Correlation for a single pair of vectors (numba compatible)"""
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < eps or std_b < eps:
        return 0.0
    ma = a - np.mean(a)
    mb = b - np.mean(b)
    num = np.sum(ma * mb)
    den = np.sqrt(np.sum(ma**2) * np.sum(mb**2))
    return num / (den + eps)

@njit
def stats_func_lv(x, max_lag=3, step_size=10):
    n_replicates = x.shape[0]
    n_timesteps = x.shape[1]
    eps = 1e-8
    
    # Pre-calculate number of features
    # 1 (ext_ind) + 5 (logs) + 1 (cross-corr) + 2 * max_lag (auto-corrs)
    num_features = 6 + (2 * max_lag)
    out = np.zeros((n_replicates, num_features))
    
    for i in range(n_replicates):
        r_full = x[i, :, 0]
        f_full = x[i, :, 1]
        
        # --- 1. Find Extinction Index ---
        ext_ind = n_timesteps # Default to full length
        for t in range(n_timesteps):
            if r_full[t] <= 1 or f_full[t] <= 1:
                ext_ind = t+1
                break
        
        # Slice the data for this replicate
        r = r_full[:ext_ind]
        f = f_full[:ext_ind]
        
        # --- 2. Base Moments ---
        #out[i, 0] = float(ext_ind) #Excluding extinction since this can result in no variance in mahalanobis covariance calculation
        out[i, 0] = batch_corr_single(r, f, eps)
        out[i, 1] = np.log(np.mean(r) + eps)
        out[i, 2] = np.log(np.mean(f) + eps)
        out[i, 3] = np.log(np.var(r) + eps)
        out[i, 4] = np.log(np.min(r) + eps)
        out[i, 5] = np.log(np.min(f) + eps)
        #out[i, 6] = batch_corr_single(r, f, eps)
        
        # --- 3. Autocorrelation Lags ---
        # Note: If ext_ind < lag, these will be NaNs. 
        # Usually handled by the ABC or Mahalanobis later.
        for l_idx in range(1, max_lag + 1):
            lag = l_idx * step_size
            col_r = 6 + (l_idx - 1) * 2
            col_f = col_r + 1
            
            if ext_ind > lag:
                out[i, col_r] = batch_corr_single(r[:-lag], r[lag:], eps)
                out[i, col_f] = batch_corr_single(f[:-lag], f[lag:], eps)
            else:
                out[i, col_r] = 0 #np.nan
                out[i, col_f] = 0 #np.nan
                
    return out



## Transcriptional Dynamics
def transcriptional_dynamics_step(
    seed=None, 
    #kplus = 30, kminus = 10, rburst = 10, diffusivity = 0.1, #Physical parameters
    physical_params = [30,10,0.1], kminus = 10,
    input_filename = None, #Heterogeneities - resimulate for 
    heterogeneities = [2,4], #bp = 2, gp = 4
    scheme = 'constant', # Scheme
    scheme_params = [2], # Scheme hyperparameter
    stop_func = None,
    sample_size = 512, T = 5, dt = 0.01, #simulation parameters (base)
    alpha = 0.5, num_particles = 500, cores = 4, #estimator parameters
    prior_domain = [[1,100],[1,100],[0,1.0]],
    method = 'constant',
    **kwargs
):
    '''
    A self-contained application of SMC ABC to the transcriptional dynamics model.
    Constitutes one replicate application of SMC ABC in the benchmark.
    Records data according to get_info()
    '''
    kplus,rburst,diffusivity = list(physical_params)
    rng = np.random.default_rng(seed)

    ### Data generation or loading:
    if input_filename is None:
        print(f"Simulating data with parameters: {physical_params} and heterogeneities: {heterogeneities}")
        ## Heterogeneity
        bp,gp = list(heterogeneities)
        if gp == None:
            lengths = np.ones((sample_size,))
        else:
            lengths = rng.gamma(shape = gp, scale = 1/gp, size = (sample_size,))
        if bp == None:
            sites = 0.5 * lengths
        else:
            sites = rng.beta(a = bp,b = bp, size = (sample_size,)) * lengths
        ## Simulate data
        data_ = td.simulate(kplus, kminus, rburst, diffusivity, T, dt, sites, lengths, seed=seed if seed is not None else -1)
        data = [np.asarray(d) for d in data_]
    else:
        print(f"Loading data from: {input_filename}")
        with open(input_filename,'rb') as f:
            td_dat = pickle.load(f)
            data = td_dat['data']
            sites = td_dat['sites']
            lengths = td_dat['lengths']

    ### Simulator
    def model(p,Bi,seed=None):
        z = sites[Bi,]
        l = lengths[Bi,]
        obs_data = [data[i] for i in Bi]
        sim = td.simulate(p[0], kminus, p[1], p[2], T, dt, z, l, seed if seed is not None else -1)
        return sim, obs_data

    ### Prior
    prior = utils.uniform_prior(prior_domain,seed = seed)

    def stats_func(x,y,sf = stats_func_td):
        return sf(x),sf(y)

    ### Scheme
    if scheme == 'constant':
        init = schemes.constant_init
        loop = schemes.constant_loop
    elif scheme == 'fvc':
        init = schemes.fvc_init
        loop = schemes.fvc_loop
    else:
        raise ValueError(f"Scheme {scheme} not identified.")
    
    ######## This makes a dictionary of information shared with the adaptive minibatch schemes
    init_params = {
        'data':data,
        'model':model,
        'stats_func':stats_func,
        'prior':prior,
        'num_particles':num_particles,
        'cores':cores,
        'alpha':alpha,
        'seed':seed
    }
    stop = default_stop if stop_func is None else stop_func

    #Maybe make this a function since it'll be repeated code...
    results_list = []
    est = init(init_params,scheme_params)
    attr_list = info_dict[method]
    while not stop(est):
        loop(est)
        results = get_info(est,attr_list)
        results_list.append(results)

    return results_list

# Summary Statistics: Transcriptional Dynamics
@njit
def stats_func_td(x):
    batch_size = len(x)
    out = np.zeros((batch_size, 3))
    for i in range(batch_size):
        out[i, 0] = len(x[i])
    m_counts = np.mean(out[:, 0])
    for i in range(batch_size):
        out[i, 1] = (out[i, 0] - m_counts)**2
        if out[i, 0] >= 2:
            out[i, 2] = np.std(x[i])
        else:
            out[i, 2] = 0.0
    return out


##### Experiment function and dictionary of models
model_dict = {
    'lv':lotka_volterra_step,
    'td':transcriptional_dynamics_step
}

def simulate_experiment(output_filename, parameter_set_list, model_name = None, num_sims=10,seeded = True):
    '''
    parameter_set_list is a list of dictionaries of inputs to the models.
    parameters
    ----------
    output_filename: string
        filename of the output
    parameter_set_list: dictionary of dictionaries
        
    model_name: string
        'td' for transcriptional dynamics, 'lv' for lotka volterra
    num_sims: int
        number of repetitions per parameter
    seeded: bool, list, nparray
        bool: if false or None, uses None as the seed (no randomization)
        list: must be num_sims in length. A list of integers. Don't use repeated integers.
    '''
    tic = time.time()
    if model_name is None:
        #check if it's in the parameter_set_list.?
        if 'model_name' in [k for k in parameter_set_list]:
            model_name = parameter_set_list['model_name']
        else:
            print("No model_name given. Either add it to your parameter_set_list as an entry or set model_name")
    
    ## Check seeding...
    if (seeded is None) or (seeded == False):
        seed = None
    elif (seeded == True):
        seed = np.arange(num_sims)
    if type(seeded) == np.ndarray:
        if seeded.shape[0] != num_sims:
            raise ValueError(f"seeded length = {seeded.shape[0]}, but num_sims = {num_sims}")
        else:
            seed = seeded
    if type(seeded) == list:
        if len(seeded) != num_sims:
            raise ValueError(f"seeded length = {len(seeded)}, but num_sims = {num_sims}")
        else:
            seed = seeded

    experimental_step = model_dict[model_name]
    sweep_dict = {}
    for p in parameter_set_list:
        p_key = to_key(p)
        print("Parameter set: ", p, 'Key: ', p_key)
        results_list = []
        for i in range(num_sims):
            s = None if seed is None else seed[i]
            print(f"Replicate Number: {i}. Seed: {s}")
            results = experimental_step(seed = s,**p)
            results_list.append(results)
            print("Done.")
            print()
        #Results_list should be: a list of lists of dictionaries of information.
        #Specificaly: Replicate, generation, information for the generation
        sweep_dict[p_key] = results_list
    save(output_filename,sweep_dict)
    print('Time taken: ', time.time() - tic)
    print("Results saved to: ",output_filename)


def to_key(p):
    keys = [k for k in p.keys()]
    tup = ()
    if 'scheme_params' in keys:
        tup += tuple(p['scheme_params'])
    if 'physical_params' in keys:
        tup += tuple(p['physical_params'])
    if 'heterogeneities' in keys:
        tup += tuple(p['heterogeneities'])
    if len(tup) == 0:
        print("Key not defined. Defaulting to ... (no default defined yet)")
    return tup

## SNR of 2 because we don't want to learn noise. This is lower limit before that.
## acceptance rate < 0.02 for a similar reason: we don't want to spend more than 100 sims per particle. 
## Average worst-case scenario suggests acceptance shouldn't drop much faster than alpha quantile.
def default_stop(est):
    return ((est.snr < 2) or (est.acceptance_rate < 0.02)) and (est.generation > 5)


## Maybe make these values reportable from the smc abc class? 
constant_attr_list = [
    'total_time','total_sims', # cost
    'log_hdpr_product', 'alpha_threshold', 'v_total_est', 'snr','acceptance_rate', # accuracy
    'batch_size',
    'hdpr_marginal', 'ESS' #In case we need it.
]
fvc_attr_list = constant_attr_list.copy()
fvc_attr_list.append('v_total')
fvc_attr_list.append('c')

info_dict = {
    'constant':constant_attr_list,
    'fvc':fvc_attr_list
}

#Observes the data in est. Gets the information to store in the output
def get_info(est,attr_list = constant_attr_list):
    dict = {}
    for attr in attr_list:
        dat = getattr(est,attr)
        if callable(dat):
            dict[attr] = dat()
        else:
            dict[attr] = dat
    return dict

def make_parameter_list(sweep_config):
    '''
    parameters
    ----------
    sweep_config : dictionary
        : A dictionary of parameters where each value is a list of other parameters to sweep.
    '''
    keys = sweep_config.keys()
    parameter_set_list = [
        dict(zip(keys, values)) 
        for values in product(*sweep_config.values())
    ]
    return parameter_set_list



######## Information processing
'''
Outputs of Experiments are shaped like: 
Dictionary of list of list of dictionary
Dictionary with keys of setup parameters | list of replicates | list of generations | output data from experiments with keys of information.

Below are helper functions. They are structured by the following input, -> output, rules. All take in the pickled results. 
Notation: dictionaries are like D[key | value]. lists are like L[iterate interpretation | value]. 
results, attribute, trunc = True -> D[parameter set][array of attributes across generations (N_reps,N^i_gens)] #N^i_gens = constant if trunc = True - the last common geneartion
results, attribute, gen = -1, true = False -> D[parameter set][Last generation's attribute (N_reps,N^i_gens)] 


'''

def get_attr(results, attr, trunc=True, slice=None):
    """
    Extract an attribute across all parameter sets and replicates.

    Parameters
    ----------
    results : dict
        Nested dict: results[param_tuple][replicate_idx][generation_idx][attr]
    attr : str
        The key to extract from each generation dict.
    trunc : bool
        If True, truncate all replicates to the maximum common generation
        (i.e. the minimum number of generations across all replicates).
        Ignored when slice is not None.
    slice : int or None
        If None, return a time series (all generations, subject to trunc).
        If a non-negative int, return the value at that generation index.
        If a negative int (e.g. -1), return the last available value per
        replicate, allowing heterogeneous lengths (ignores trunc).

    Returns
    -------
    output : dict
        Keys are param tuples from results.
        Values are lists (one per replicate) of:
          - a list of attr values across generations (time series mode), or
          - a single attr value (slice mode).
    """

    # ------------------------------------------------------------------ #
    # Helper: all replicates for one param set as a list-of-lists         #
    # ------------------------------------------------------------------ #
    def _collect(param_replicates):
        """Return list-of-lists: replicates × generations for attr."""
        return [
            [gen[attr] for gen in replicate]
            for replicate in param_replicates
        ]

    output = {}

    for param, replicates in results.items():
        series = _collect(replicates)          # list of lists
        lengths = [len(s) for s in series]    # num generations per replicate
        min_len = min(lengths)

        # -------- SLICE MODE ------------------------------------------ #
        if slice is not None:

            if slice < 0:
                # Negative index → last available value, heterogeneous lengths
                output[param] = [s[slice] for s in series]

            else:
                # Non-negative index
                if trunc:
                    # Respect max-common generation; skip if out of range
                    cap = min_len
                    output[param] = [
                        s[slice] if slice < cap else s[cap - 1]
                        for s in series
                    ]
                else:
                    # Best-effort: use what each replicate has
                    output[param] = [
                        s[slice] if slice < len(s) else s[-1]
                        for s in series
                    ]

        # -------- TIME SERIES MODE ------------------------------------ #
        else:
            if trunc:
                # Truncate every replicate to the shortest one
                output[param] = [s[:min_len] for s in series]
            else:
                # Return full series for each replicate (heterogeneous ok)
                output[param] = series

    return output

#helps get quantiles as used in the plots.
def get_quantiles(attr_dict,q = [0.25,0.5,0.75]):
    out = {}
    for k in attr_dict.keys():
        out[k] = np.quantile(attr_dict[k],q,axis=0)
    return out



#### load results
def load(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

def save(filename,data):
    with open(filename, 'wb') as f:
        pickle.dump(data,f)


###### Plotting Functions
'''
Plotting functions.

Pareto Frontiers: Comparison of methods in accuracy and cost
Time Series: Comparison of dynamics of adaptive and constant SMC ABC behaviors

helper plots take in and return the figure and axis so you can add to it as needed.
These helper plotting functions are then stitched together to create a full figure plot.

Helper plotting functions: 
Plot a Time series (x,y)
'''
def plot_ts(x_axis,y_axis,fig_ax = None, grid_lines = False, #basics
            errorbar_params = {'linewidth':1}, errorbar_cmap = None, #dictionary of params for plt.errorbar
            grid_line_params = {'color':'gray','linewidth':1,'alpha':0.5} #dictionary of params for grid lines
            ):
    '''
    Plots a time series with error bars
    parameters
    ----------
    x_axis : dictionary
        This a result of get_attr(). A dictionary with keys of parameters and values of matricies of time series (N_replicates,attribute)
    y_axis : dictinoary
        Same as x_axis, just plotted in the y axis.
    '''
    fig,ax = plt.subplots() if fig_ax is None else fig_ax
    errorbar_cmap = 'viridis' if errorbar_cmap is None else errorbar_cmap
    # eb_cmap = # If errorbar_cmap is a string, we fetch the cmap from matplotlib. if it's callable, we assume that it's ready to use.
    
    x = get_quantiles(x_axis)
    y = get_quantiles(y_axis)
    if grid_lines: #these are lines across parameters to match them. Used for pareto frontiers.
        #store lines in a list. append.
        #pass them to a small plot function.
        grid_list = []
        pass
    for k in x.keys(): #keys are assumed to be the same...
        x_ext = np.array(x[k],dtype = object)
        y_ext = np.array(y[k],dtype = object)
        x_ext = x_ext[:,None] if x_ext.ndim < 2 else x_ext
        y_ext = y_ext[:,None] if y_ext.ndim < 2 else y_ext
        x_med = x_ext[1,:]
        y_med = y_ext[1,:]
        if grid_lines:
            grid_list.append(np.stack((x_med,y_med),axis=1)) #Should be shaped like (N,2)?
        x_err = np.stack((x_ext[2,:] - x_ext[1,:], x_ext[1,:] - x_ext[0,:]),axis=0)
        y_err = np.stack((y_ext[2,:] - y_ext[1,:], y_ext[1,:] - y_ext[0,:]),axis=0)
        ax.errorbar(x_med,y_med,xerr=x_err,yerr=y_err,**errorbar_params)
    if grid_lines:
        grid_list = np.array(grid_list,dtype = object)
        max_common_gen = min([len(l) for l in grid_list.T])
        grid_list = np.array([l[:max_common_gen] for l in grid_list])
        for i in range(max_common_gen):
            ax.plot(grid_list[:,i,0],grid_list[:,i,1],**grid_line_params)
        pass #plott here
    return fig,ax

