'''
Experimental Setups
We have lotka-volterra and transcriptional dynamics.
We measure data for each generation and pickle it at the end. 
The outputs are in the format: 
    Dictionary of keys (parameter sets) values (list of replicates (list of generations (dictionary of information)))
'''


## Import smc abc class and schemes
import smc_abc_schemes as schemes
import smc_abc_utils as utils

import numpy as np
import pickle
import time
from itertools import product

from numba import njit

#from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import numpy as np


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
    def model(p, Bi, num_reps=1, seed=None):
        alpha, beta, gamma = p
        s_sim = np.zeros((Bi.shape[0],N_stats))
        rep_rng = np.random.default_rng(seed) if seed is not None else None
        for _ in range(num_reps):
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
        'seed':seed,
        'parallel_args':dict(max_nbytes = '100M', timeout=99999)
    }
    stop = default_stop if stop_func is None else stop_func

    #Maybe make this a function since it'll be repeated code...
    results_list = []
    est = init(init_params,scheme_params)
    while not stop(est):
        loop(est)
        results = get_info(est)
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
def stats_func_lv(x, max_lag=1, step_size=10):
    n_replicates = x.shape[0]
    n_timesteps = x.shape[1]
    eps = 1e-8
    
    # Pre-calculate number of features
    # 1 (ext_ind) + 5 (logs) + 1 (cross-corr) + 2 * max_lag (auto-corrs)
    num_features = 4 + (2 * max_lag)
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
        out[i, 0] = batch_corr_single(r, f, eps)
        out[i, 1] = np.log(np.mean(r) + eps)
        out[i, 2] = np.log(np.mean(f) + eps)
        out[i, 3] = np.log(np.var(r) + eps)

        # --- 3. Autocorrelation Lags ---
        for l_idx in range(1, max_lag + 1):
            lag = l_idx * step_size
            col_r = 4 + (l_idx - 1) * 2
            col_f = col_r + 1
            
            if ext_ind > lag:
                out[i, col_r] = batch_corr_single(r[:-lag], r[lag:], eps)
                out[i, col_f] = batch_corr_single(f[:-lag], f[lag:], eps)
            else:
                out[i, col_r] = 0
                out[i, col_f] = 0
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
    prior_domain = [[1,100],[1,100],[0,1.0]], stats_per_batch = False,
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
    if stats_per_batch:
        statsf = stats_func_td_per_batch
    else:
        statsf = stats_func_td

    def stats_func(x,y,sf = statsf):
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
        'seed':seed,
        'parallel_args':dict(max_nbytes = '100M', timeout=99999)
    }
    stop = default_stop if stop_func is None else stop_func

    #Maybe make this a function since it'll be repeated code...
    results_list = []
    est = init(init_params,scheme_params)
    while not stop(est):
        loop(est)
        results = get_info(est)
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

@njit
def stats_func_td_per_batch(x):
    batch_size = len(x)
    out = np.zeros((3,))
    counts = np.zeros((batch_size,))
    std_list = np.zeros((batch_size,))
    for i in range(batch_size):
        counts[i] = len(x[i])
        if counts[i] >= 2:
            std_list[i] = np.std(x[i])
        else:
            std_list[i] = 0.0
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    mean_std_pos = np.mean(std_list)
    out[0] = mean_count
    out[1] = var_count
    out[2] = mean_std_pos
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


attr_list = [
    'total_time','total_sims', # cost
    'log_hdpr_product', 'alpha_threshold', 'v_total_est', 'snr','acceptance_rate', # accuracy
    'batch_size', 'v_total', 'c',
    'hdpr_marginal', 'ESS'
]

#Observes the data in est. Gets the information to store in the output
def get_info(est,attr_list = attr_list):
    dict = {}
    for attr in attr_list:
        if hasattr(est,attr):
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


#### Post-hoc stopping criterion
def ph_stop(data, stop_func):
    """
    Apply a post-hoc stopping criterion to experimental results.
    
    This function takes loaded results (from experiments.load) and truncates each replicate's
    generations based on a user-defined stopping function.
    
    Parameters
    ----------
    data : dict
        Nested dict structure: data[param_tuple][replicate_idx][generation_idx][attr_name]
        Loaded from experiments.load()
    stop_func : callable
        A function that takes a generation dictionary (output of get_info) and returns True
        when the stopping criterion is met. The function should have signature:
        bool = stop_func(gen_dict)
        
        Example:
        >>> def stop_func(gen_dict):
        >>>     return (gen_dict['snr'] < 2 and gen_dict['acceptance_rate'] < 0.02)
    
    Returns
    -------
    data_ph : dict
        Same structure as input, but with each replicate truncated to stop at the 
        generation where stop_func first returns True (inclusive of that generation).
    
    Usage
    -----
    >>> data = experiments.load('results/experiment.pkl')
    >>> def my_stop(gen_dict):
    >>>     return gen_dict['acceptance_rate'] < 0.01
    >>> data_truncated = experiments.ph_stop(data, my_stop)
    """
    data_ph = {}
    
    for param_key, replicates in data.items():
        truncated_replicates = []
        
        for replicate in replicates:
            # Apply stopping criterion to this replicate
            truncated_replicate = []
            for gen_dict in replicate:
                truncated_replicate.append(gen_dict)
                # Stop after including the generation where stop_func returns True
                if stop_func(gen_dict):
                    break
            truncated_replicates.append(truncated_replicate)
        
        data_ph[param_key] = truncated_replicates
    
    return data_ph


def subset(data, keys_to_keep):
    """
    Subset experimental results by keeping only specified parameter keys.
    
    This function filters a loaded results dictionary to keep only the specified
    parameter configurations, removing all others.
    
    Parameters
    ----------
    data : dict
        Nested dict structure: data[param_tuple][replicate_idx][generation_idx][attr_name]
        Loaded from experiments.load()
    keys_to_keep : list
        List of parameter tuple keys to keep. Examples:
        - [(32,), (64,), (128,)] if parameters are single values
        - [(1, 0.01), (1, 0.05)] if parameters are tuples
        - Can also pass indices (converted to corresponding keys)
    
    Returns
    -------
    data_subset : dict
        Same nested structure as input, but containing only the specified parameter keys.
    
    Raises
    ------
    KeyError
        If any of the specified keys are not found in the data.
    
    Usage
    -----
    >>> data = experiments.load('results/experiment.pkl')
    >>> # Keep only specific parameter configurations
    >>> data_small = experiments.subset(data, [(32,), (64,), (128,)])
    >>> # Or keep by index
    >>> data_small = experiments.subset(data, [0, 2, 3])  # Keep 1st, 3rd, 4th configs
    """
    data_subset = {}
    all_keys = list(data.keys())
    
    for key_spec in keys_to_keep:
        # If integer index, convert to actual key
        if isinstance(key_spec, int):
            if key_spec < 0 or key_spec >= len(all_keys):
                raise IndexError(f"Index {key_spec} out of range for {len(all_keys)} parameter keys")
            actual_key = all_keys[key_spec]
        else:
            actual_key = key_spec
        
        if actual_key not in data:
            raise KeyError(f"Parameter key {actual_key} not found in results. Available keys: {list(data.keys())}")
        
        data_subset[actual_key] = data[actual_key]
    
    return data_subset


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


def noise(res,trunc,slice):
    v_total = get_attr(res,'v_total_est',trunc=trunc,slice=slice)
    n = get_attr(res,'batch_size',trunc=trunc,slice=slice)
    output = {}
    for k in res.keys():
        output[k] = np.array(v_total[k]) / np.array(n[k])
    return output
noise.__name__ = 'noise'

def time_series(
    results_filename,
    attr_list=[
        'total_time', 'total_sims', 'log_hdpr_product', 'acceptance_rate',
        'v_total_est', 'alpha_threshold', 'snr', 'batch_size', 'ESS', 'noise'
    ],
    attr_ylabels=[
        r'\mathrm{Wall\ Time\ (s)}',
        r'\mathrm{Simulations}',
        r'\log\!\left(\prod_i \ell_i\right)',
        r'\mathrm{Acceptance\ Rate}',
        r'v_t',
        r'\epsilon_t',
        r'\mathrm{SNR} = \epsilon_t^2 / (v_t / n)',
        r'\mathrm{Batch\ Size}\ (n)',
        r'\mathrm{ESS}',
        r'\mathrm{Noise\ Variance}\ (v_t / n)'
    ],
    attr_transform=[
        'id','id','id','id','id','id','id','id','id','id'
    ],
    legend_key_index=0,
    legend_title=None,        # e.g. r'\alpha' — defaults to \theta_{i}
    errorbar_cmap='winter',
    quantiles=[0.25, 0.5, 0.75],
    trunc = True, slice = None,
    out_handle = None,
    post_hoc_stop = None,
    plot_args={},
):
    """
    Plot each attribute in attr_list as a time series over SMC ABC generations,
    with inter-replicate quantile error bars. One SVG is saved per attribute.

    Parameters
    ----------
    results_filename : str
        Path to a .pkl results file.
    attr_list : list of str
        Attributes to extract and plot.
    attr_ylabels : list of str
        Raw LaTeX strings for y-axis labels (no outer $…$ needed).
    attr_transform : list of {'id', 'log'}
        Per-attribute transform applied before quantile computation.
    legend_key_index : int
        Index into each param tuple used as the legend entry value.
    legend_title : str or None
        LaTeX string for the legend title. Defaults to r'\theta_{i}'.
    errorbar_cmap : str or Colormap
        Matplotlib colormap for colouring each param-tuple curve.
    quantiles : list of 3 floats
        [lower, median, upper] quantile levels passed to get_quantiles.
    plot_args : dict
        Extra kwargs forwarded to ax.errorbar.
    """

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import AutoMinorLocator
    import numpy as np

    # ── LaTeX + font setup ────────────────────────────────────────────────────
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'lines.linewidth': 1.4,
        'errorbar.capsize': 2.5,
    })

    results = load(results_filename)
    if post_hoc_stop is not None: 
        results = ph_stop(results, post_hoc_stop)

    transforms = {
        'id':  lambda x: x,
        'log': lambda x: np.log(x),
    }
    ylabel_prefixes = {
        'id':  '',
        'log': r'\ln\,',
    }

    # ── Resolve colormap ──────────────────────────────────────────────────────
    cmap = (
        plt.get_cmap(errorbar_cmap)
        if isinstance(errorbar_cmap, str)
        else errorbar_cmap
    )

    for attr, ylab, transform_name in zip(attr_list, attr_ylabels, attr_transform):
        f      = transforms[transform_name]
        tylab  = ylabel_prefixes[transform_name]

        # Apply transform before quantile computation (statistically correct)
        # raw         = get_attr(results, attr, trunc=True)
        # Inside the attr loop, replace the raw = get_attr(...) line:
        if attr == 'noise':
            raw = noise(results, trunc=trunc, slice=slice)
        else:
            raw = get_attr(results, attr, trunc=trunc, slice=slice)

        transformed = {k: f(np.array(v, dtype=float)) for k, v in raw.items()}
        res_attr    = get_quantiles(transformed, q=quantiles)

        # ── Partition keys: sweep vs auto-c ───────────────────────────────────
        def _is_auto(k):
            return k[legend_key_index] is None

        sweep_keys = [k for k in res_attr.keys() if not _is_auto(k)]
        auto_keys  = [k for k in res_attr.keys() if     _is_auto(k)]

        # ── Estimate mean c for auto keys (from raw results, attr 'c') ────────
        def calc_mean_c(results):
            """Average c across all replicates and generations."""
            if not auto_keys:
                return {}
            outputs = {}
            try:
                c_list = get_attr(results, 'c', trunc=False, slice=-1)
                for k in auto_keys:
                    outputs[k] = np.mean(c_list[k]) if k in c_list else float('nan')
            except:
                for k in auto_keys:
                    outputs[k] = float('nan')
            return outputs

        auto_c_estimates = calc_mean_c(results)

        # ── Colors for sweep points ───────────────────────────────────────────
        n_sweep = len(sweep_keys)
        sweep_colors = [cmap(i / max(n_sweep - 1, 1)) for i in range(n_sweep)]

        # ── Figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 3.5))

        # Plot sweep keys
        for color, k in zip(sweep_colors, sweep_keys):
            q_arr = res_attr[k]          # (3, T)
            T     = q_arr.shape[1]
            gens  = np.arange(T)

            med   = q_arr[1, :]
            lower = q_arr[1, :] - q_arr[0, :]   # median − q_low  (positive)
            upper = q_arr[2, :] - q_arr[1, :]   # q_high − median (positive)

            legend_val = k[legend_key_index]
            # Format: integers cleanly, floats in scientific if small
            if isinstance(legend_val, float) and abs(legend_val) < 0.01:
                label = rf'${legend_val:.2e}$'
            else:
                label = rf'${legend_val}$'

            ax.errorbar(
                gens, med,
                yerr=np.stack([lower, upper], axis=0),
                color=color,
                label=label,
                linewidth=1.4,
                capsize=2.5,
                capthick=0.9,
                elinewidth=0.7,
                **plot_args,
            )

        # Plot auto-c keys (magenta star marker, labelled with estimated c)
        for k in auto_keys:
            q_arr = res_attr[k]          # (3, T)
            T     = q_arr.shape[1]
            gens  = np.arange(T)

            med   = q_arr[1, :]
            lower = q_arr[1, :] - q_arr[0, :]   # median − q_low  (positive)
            upper = q_arr[2, :] - q_arr[1, :]   # q_high − median (positive)

            mean_c = auto_c_estimates[k]
            label = rf'$\mathrm{{auto}},\ \bar{{c}} \approx {mean_c:.2f}$'

            ax.errorbar(
                gens, med,
                yerr=np.stack([lower, upper], axis=0),
                color='magenta',
                marker='*',
                markersize=7,
                label=label,
                linewidth=1.4,
                capsize=2.5,
                capthick=0.9,
                elinewidth=0.7,
                zorder=5,
                **plot_args,
            )

        # ── Axis labels ───────────────────────────────────────────────────────
        ax.set_xlabel(r'Generation')
        ax.set_ylabel(rf'${tylab}{ylab}$')

        # ── Ticks: publication style (inward, all four sides) ────────────────
        ax.tick_params(axis='both', which='major',
                       direction='in', top=True, right=True, length=4)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='minor',
                       direction='in', top=True, right=True, length=2)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        # Integer x-ticks only (generations are whole numbers)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── Legend ────────────────────────────────────────────────────────────
        ltitle = (
            legend_title
            if legend_title is not None
            else rf'$\theta_{{{legend_key_index}}}$'
        )
        ax.legend(
            title=ltitle,
            framealpha=0.85,
            edgecolor='0.75',
            handlelength=1.8,
        )

        fig.tight_layout()
        if out_handle is None:
            handle = 'results/output'
            out_path = f'{handle}_{attr}.svg'
        else:
            out_path = f'{out_handle}_{attr}.svg'
        plt.savefig(out_path, dpi=600, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')



# To do: Change the marker of one of them.
# To do: fix output filename default 
def pareto_frontier(
    ref_filename,
    novel_filename,
    x_attr='total_time',
    y_attr='log_hdpr_product',
    x_label=r'\mathrm{Wall\ Time\ (s)}',
    y_label=r'\log\!\left(\prod_i \ell_i\right)',
    x_transform='id',
    y_transform='id',
    ref_label=r'\mathrm{Constant}',
    novel_label=r'\mathrm{Adaptive}',
    ref_cmap='autumn',
    novel_cmap='winter',
    legend_key_index_ref=0,
    legend_key_index=0,
    legend_title=None,
    auto_c_color='magenta',
    auto_c_marker='*',
    auto_c_label=r'\mathrm{auto}',
    plot_args={},
    out_filename=None,
    post_hoc_stop = None
):
    """
    Plot Pareto frontiers comparing constant minibatch SMC-ABC (reference)
    against adaptive SMC-ABC (novel), sweeping over hyperparameters.

    Each point is the last-generation value per replicate (no truncation),
    with quantile error bars across replicates.

    Keys with None at legend_key_index are treated as the auto-selected c run:
    plotted in a contrasting color (default magenta) with a star marker, and
    labelled with the estimated mean c across replicates/generations.

    Parameters
    ----------
    ref_filename : str
        Path to reference (constant minibatch) .pkl results file.
    novel_filename : str
        Path to novel (adaptive) .pkl results file.
    x_attr : str or callable
        Cost metric. String key or fn(gen_dict)->scalar.
    y_attr : str or callable
        Accuracy metric. String key or fn(gen_dict)->scalar.
    x_label : str
        Raw LaTeX string for x-axis label.
    y_label : str
        Raw LaTeX string for y-axis label.
    x_transform : {'id', 'log'}
        Transform applied to x values before quantile computation.
    y_transform : {'id', 'log'}
        Transform applied to y values before quantile computation.
    ref_label : str
        LaTeX display name for the reference method (legend title).
    novel_label : str
        LaTeX display name for the novel method (legend title).
    ref_cmap : str or Colormap
        Colormap for reference method curves.
    novel_cmap : str or Colormap
        Colormap for novel method curves.
    legend_key_index : int
        Index into each param tuple used as the legend entry value.
    legend_title : str or None
        Shared sweep-parameter label, e.g. r'\alpha'. Defaults to \theta_{i}.
    auto_c_color : str
        Color for the auto-c point. Default 'magenta'.
    auto_c_marker : str
        Marker for the auto-c point. Default '*'.
    auto_c_label : str
        Raw LaTeX string appended as the legend label for the auto-c point.
    plot_args : dict
        Extra kwargs forwarded to ax.errorbar for sweep points.
    out_filename : str or None
        Output SVG path. Defaults to '{ref_handle}_vs_{novel_handle}_pareto.svg'.
    """

    # import matplotlib.pyplot as plt
    # import matplotlib.ticker as ticker
    # from matplotlib.ticker import AutoMinorLocator
    # import numpy as np

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'lines.linewidth': 1.4,
    })

    transforms = {'id': lambda x: x, 'log': np.log}
    fx = transforms[x_transform]
    fy = transforms[y_transform]

    def _resolve_cmap(c):
        return plt.get_cmap(c) if isinstance(c, str) else c

    ref_cmap_fn   = _resolve_cmap(ref_cmap)
    novel_cmap_fn = _resolve_cmap(novel_cmap)

    # ── Load ──────────────────────────────────────────────────────────────────
    ref_results   = load(ref_filename)
    novel_results = load(novel_filename)
    if post_hoc_stop is not None:
        ref_results   = ph_stop(ref_results, post_hoc_stop)
        novel_results = ph_stop(novel_results, post_hoc_stop)

    def _extract(results, attr, f):
        """Last-gen value per replicate, no truncation, then transform."""
        if callable(attr):
            #raw = get_derived(results, attr, trunc=False)
            raw = attr(results)
        else:
            raw = get_attr(results, attr, trunc=False, slice=-1)
        return {k: f(np.array(v, dtype=float)) for k, v in raw.items()}

    def _quantiles(d):
        return get_quantiles(d, q=[0.25, 0.5, 0.75])

    ref_x   = _quantiles(_extract(ref_results,   x_attr, fx))
    ref_y   = _quantiles(_extract(ref_results,   y_attr, fy))
    novel_x = _quantiles(_extract(novel_results, x_attr, fx))
    novel_y = _quantiles(_extract(novel_results, y_attr, fy))

    # ── Partition novel keys: sweep vs auto-c ─────────────────────────────────
    def _is_auto(k):
        return k[legend_key_index] is None

    novel_sweep_keys = [k for k in novel_x.keys() if not _is_auto(k)]
    novel_auto_keys  = [k for k in novel_x.keys() if     _is_auto(k)]

    # ── Estimate mean c for auto keys (from raw results, attr 'c') ───────────
    def _mean_c(results, keys):
        """Average c across all replicates and generations for each auto key."""
        estimates = {}
        for k in keys:
            replicates = results[k]
            vals = [
                gen['c']
                for replicate in replicates
                for gen in replicate
                if 'c' in gen
            ]
            estimates[k] = np.mean(vals) if vals else float('nan')
        return estimates
    def check_c(results):
        '''If c is in first result, returns True.'''
        keys = [k for k in results.keys()]
        if 'c' in results[keys[0]][0][0]:
            return True
        else:
            return False
    def calc_mean_c(results):
        if check_c(results):
            c_list = get_attr(results,'c',trunc = False,slice = -1)
            outputs ={}
            for k in results.keys():
                outputs[k] = np.mean(c_list[k])
        else:
            outputs = {}
            for k in results.keys():
                outputs[k] = np.nan
        return outputs
    #auto_c_estimates = _mean_c(novel_results, novel_auto_keys)
    auto_c_estimates = calc_mean_c(novel_results)

    # ── Colors for sweep points ───────────────────────────────────────────────
    def _colors(cmap_fn, n, lo=0.35, hi=0.90):
        return [cmap_fn(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]

    ref_colors         = _colors(ref_cmap_fn,   len(ref_x))
    novel_sweep_colors = _colors(novel_cmap_fn, len(novel_sweep_keys))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # shared errorbar style
    _eb_kw = dict(
        linewidth=0, marker='o', markersize=4.5,
        capsize=2.5, capthick=0.9, elinewidth=0.8,
    )

    def _label_val(val):
        if isinstance(val, float) and abs(val) < 0.01:
            return rf'${val:.2e}$'
        return rf'${val}$'

    # ── Reference sweep ───────────────────────────────────────────────────────
    ref_handles, ref_labels = [], []
    gray_line_list = []
    for color, k in zip(ref_colors, ref_x.keys()):
        qx, qy = ref_x[k], ref_y[k]
        gray_line_list.append([qx[1],qy[1]])
        eb = ax.errorbar(
            qx[1], qy[1],
            xerr=[[qx[1]-qx[0]], [qx[2]-qx[1]]],
            yerr=[[qy[1]-qy[0]], [qy[2]-qy[1]]],
            color=color, **_eb_kw, **plot_args,
        )
        ref_handles.append(eb)
        ref_labels.append(_label_val(k[legend_key_index_ref]))
    # Grey connecting line
    gray_line_list = np.array(gray_line_list)
    ax.plot(gray_line_list[:,0],gray_line_list[:,1],color = 'gray',linewidth=1,alpha = 0.5)

    # ── Novel sweep ───────────────────────────────────────────────────────────
    novel_handles, novel_labels = [], []
    for color, k in zip(novel_sweep_colors, novel_sweep_keys):
        qx, qy = novel_x[k], novel_y[k]
        eb = ax.errorbar(
            qx[1], qy[1],
            xerr=[[qx[1]-qx[0]], [qx[2]-qx[1]]],
            yerr=[[qy[1]-qy[0]], [qy[2]-qy[1]]],
            color=color, **_eb_kw, **plot_args,
        )
        novel_handles.append(eb)
        novel_labels.append(_label_val(k[legend_key_index]))

    # ── Auto-c points (magenta star, one per auto key) ────────────────────────
    auto_handles, auto_labels = [], []
    for k in novel_auto_keys:
        qx, qy   = novel_x[k], novel_y[k]
        mean_c   = auto_c_estimates[k]
        eb = ax.errorbar(
            qx[1], qy[1],
            xerr=[[qx[1]-qx[0]], [qx[2]-qx[1]]],
            yerr=[[qy[1]-qy[0]], [qy[2]-qy[1]]],
            color=auto_c_color,
            marker=auto_c_marker,
            markersize=7,
            capsize=2.5, capthick=0.9, elinewidth=0.8,
            linewidth=0,
            zorder=5,
        )
        auto_handles.append(eb)
        auto_labels.append(
            rf'${auto_c_label},\ \bar{{c}} \approx {mean_c:.2f}$'
        )



    # # ── Two side-by-side legends, top-right ───────────────────────────────────
    # # Novel legend (right), then ref legend placed to its left via bbox offset.
    # # We avoid get_window_extent (requires renderer) by using axes-fraction coords.
    ltitle = legend_title if legend_title is not None else rf'$\theta_{{{legend_key_index}}}$'
    novel_leg_handles = novel_handles + auto_handles
    novel_leg_labels  = novel_labels  + auto_labels

    # 1. Create the Novel Legend (anchored to top-right)
    leg_novel = ax.legend(
        handles=novel_leg_handles,
        labels=novel_leg_labels,
        title='$' + novel_label + '$' + '\n' + '$' + ltitle + '$',
        alignment='center',
        loc='upper right',
        framealpha=0.85,
        edgecolor='0.75',
        handlelength=1.0,
        borderpad=0.6,
    )
    leg_novel.get_title().set_multialignment('center')
    ax.add_artist(leg_novel)

    # 2. Force a draw to calculate the legend size
    fig.canvas.draw()

    # 3. Get the LEFT edge of the novel legend in axes coordinates
    inv = ax.transAxes.inverted()
    bbox_novel = leg_novel.get_window_extent().transformed(inv)
    left_edge = bbox_novel.x0  # x0 is the left boundary

    # 4. Create the Ref Legend, anchored to the left_edge of leg_novel
    leg_ref = ax.legend(
        handles=ref_handles,
        labels=ref_labels,
        title='$' + ref_label + '$' + '\n' + '$\mathrm{Batch Size}$',
        alignment='center',
        loc='upper right',  # Keep 'upper right' so its right edge touches the anchor
        framealpha=0.85,
        edgecolor='0.75',
        handlelength=1.0,
        borderpad=0.6,
        bbox_to_anchor=(left_edge - 0.01, 1.0), # Subtract a small gap (0.01)
        bbox_transform=ax.transAxes,
    )
    leg_ref.get_title().set_multialignment('center')
    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlabel(rf'${x_label}$')
    ax.set_ylabel(rf'${y_label}$')

    ax.tick_params(axis='both', which='major',
                   direction='in', top=True, right=True, length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='minor',
                   direction='in', top=True, right=True, length=2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_filename is None:
        ref_handle   = ref_filename[:-4]
        novel_handle = novel_filename[:-4]
        out_filename = f'{ref_handle}_vs_{novel_handle}_pareto.svg'

    plt.savefig(out_filename, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_filename}')


def pareto_frontier_scatter(
    ref_filename,
    novel_filename,
    x_attr='total_time',
    y_attr='log_hdpr_product',
    x_label=r'\mathrm{Wall\ Time\ (s)}',
    y_label=r'\log\!\left(\prod_i \ell_i\right)',
    x_transform='id',
    y_transform='id',
    ref_label=r'\mathrm{Constant}',
    novel_label=r'\mathrm{Adaptive}',
    ref_cmap='autumn',
    novel_cmap='winter',
    legend_key_index_ref=0,
    legend_key_index=0,
    legend_title=None,
    auto_c_color='magenta',
    auto_c_marker='*',
    auto_c_label=r'\mathrm{auto}',
    markersize=6,
    alpha_scatter=0.6,
    out_filename=None,
    post_hoc_stop=None
):
    """
    Scatter plot version of pareto frontier comparing reference vs novel methods.
    
    Instead of plotting quantile error bars, plots every replicate as a scatter point,
    colored by parameter sweep. Reference and novel methods are shown separately
    with different colormaps.

    Parameters
    ----------
    ref_filename : str
        Path to reference (constant minibatch) .pkl results file.
    novel_filename : str
        Path to novel (adaptive) .pkl results file.
    x_attr : str
        Cost metric (e.g., 'total_time').
    y_attr : str
        Accuracy metric (e.g., 'log_hdpr_product').
    x_label : str
        Raw LaTeX string for x-axis label.
    y_label : str
        Raw LaTeX string for y-axis label.
    x_transform : {'id', 'log'}
        Transform applied to x values.
    y_transform : {'id', 'log'}
        Transform applied to y values.
    ref_label : str
        LaTeX display name for reference method.
    novel_label : str
        LaTeX display name for novel method.
    ref_cmap : str or Colormap
        Colormap for reference scatter points.
    novel_cmap : str or Colormap
        Colormap for novel scatter points.
    legend_key_index_ref : int
        Index into param tuple for reference legend.
    legend_key_index : int
        Index into param tuple for novel legend.
    legend_title : str or None
        Shared sweep-parameter label.
    auto_c_color : str
        Color for auto-c points.
    auto_c_marker : str
        Marker for auto-c points.
    auto_c_label : str
        Label for auto-c in legend.
    markersize : int
        Size of scatter points.
    alpha_scatter : float
        Transparency of scatter points (0-1).
    out_filename : str or None
        Output SVG path.
    post_hoc_stop : callable or None
        Post-hoc stopping function to apply.
    """

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'lines.linewidth': 1.4,
    })

    transforms = {'id': lambda x: x, 'log': np.log}
    fx = transforms[x_transform]
    fy = transforms[y_transform]

    def _resolve_cmap(c):
        return plt.get_cmap(c) if isinstance(c, str) else c

    ref_cmap_fn = _resolve_cmap(ref_cmap)
    novel_cmap_fn = _resolve_cmap(novel_cmap)

    # ── Load ──────────────────────────────────────────────────────────────────
    ref_results = load(ref_filename)
    novel_results = load(novel_filename)
    if post_hoc_stop is not None:
        ref_results = ph_stop(ref_results, post_hoc_stop)
        novel_results = ph_stop(novel_results, post_hoc_stop)

    def _extract_all_replicates(results, attr, f):
        """
        Extract all replicate values at last generation.
        Returns dict: param_key -> list of values (one per replicate).
        """
        raw = get_attr(results, attr, trunc=False, slice=-1)
        return {k: f(np.array(v, dtype=float)) for k, v in raw.items()}

    ref_x = _extract_all_replicates(ref_results, x_attr, fx)
    ref_y = _extract_all_replicates(ref_results, y_attr, fy)
    novel_x = _extract_all_replicates(novel_results, x_attr, fx)
    novel_y = _extract_all_replicates(novel_results, y_attr, fy)

    # ── Partition novel keys: sweep vs auto-c ─────────────────────────────────
    def _is_auto(k):
        return k[legend_key_index] is None

    novel_sweep_keys = [k for k in novel_x.keys() if not _is_auto(k)]
    novel_auto_keys = [k for k in novel_x.keys() if _is_auto(k)]

    # ── Calculate mean c for auto keys ────────────────────────────────────────
    def check_c(results):
        '''If c is in first result, returns True.'''
        keys = [k for k in results.keys()]
        if 'c' in results[keys[0]][0][0]:
            return True
        else:
            return False

    def calc_mean_c(results):
        if check_c(results):
            c_list = get_attr(results, 'c', trunc=False, slice=-1)
            outputs = {}
            for k in results.keys():
                outputs[k] = np.mean(c_list[k])
        else:
            outputs = {}
            for k in results.keys():
                outputs[k] = np.nan
        return outputs

    auto_c_estimates = calc_mean_c(novel_results)

    # ── Colors for parameter sweep ────────────────────────────────────────────
    def _colors(cmap_fn, n, lo=0.35, hi=0.90):
        return [cmap_fn(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]

    ref_colors = _colors(ref_cmap_fn, len(ref_x))
    novel_sweep_colors = _colors(novel_cmap_fn, len(novel_sweep_keys))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    def _label_val(val):
        if isinstance(val, float) and abs(val) < 0.01:
            return rf'${val:.2e}$'
        return rf'${val}$'

    # ── Reference sweep: scatter all replicates ───────────────────────────────
    ref_handles, ref_labels = [], []
    for color, k in zip(ref_colors, ref_x.keys()):
        x_vals = ref_x[k]
        y_vals = ref_y[k]
        scatter = ax.scatter(
            x_vals, y_vals,
            color=color,
            s=markersize**2,
            alpha=alpha_scatter,
            edgecolors='none',
        )
        ref_handles.append(scatter)
        ref_labels.append(_label_val(k[legend_key_index_ref]))

    # ── Novel sweep: scatter all replicates ───────────────────────────────────
    novel_handles, novel_labels = [], []
    for color, k in zip(novel_sweep_colors, novel_sweep_keys):
        x_vals = novel_x[k]
        y_vals = novel_y[k]
        scatter = ax.scatter(
            x_vals, y_vals,
            color=color,
            s=markersize**2,
            alpha=alpha_scatter,
            edgecolors='none',
        )
        novel_handles.append(scatter)
        novel_labels.append(_label_val(k[legend_key_index]))

    # ── Auto-c points (magenta star, one point per auto key) ──────────────────
    auto_handles, auto_labels = [], []
    for k in novel_auto_keys:
        x_vals = novel_x[k]
        y_vals = novel_y[k]
        mean_c = auto_c_estimates[k]
        scatter = ax.scatter(
            x_vals, y_vals,
            color=auto_c_color,
            marker=auto_c_marker,
            s=markersize**2 * 2,  # Make auto-c points slightly bigger
            alpha=alpha_scatter,
            edgecolors='none',
            zorder=5,
        )
        auto_handles.append(scatter)
        auto_labels.append(
            rf'${auto_c_label},\ \bar{{c}} \approx {mean_c:.2f}$'
        )

    # ── Legends: side-by-side at top-right ────────────────────────────────────
    ltitle = legend_title if legend_title is not None else rf'$\theta_{{{legend_key_index}}}$'
    novel_leg_handles = novel_handles + auto_handles
    novel_leg_labels = novel_labels + auto_labels

    # Novel legend (right)
    leg_novel = ax.legend(
        handles=novel_leg_handles,
        labels=novel_leg_labels,
        title='$' + novel_label + '$' + '\n' + '$' + ltitle + '$',
        alignment='center',
        loc='upper right',
        framealpha=0.85,
        edgecolor='0.75',
        handlelength=1.0,
        borderpad=0.6,
        scatterpoints=1,
    )
    ax.add_artist(leg_novel)

    # Force draw to get legend dimensions
    fig.canvas.draw()

    # Get left edge of novel legend
    inv = ax.transAxes.inverted()
    bbox_novel = leg_novel.get_window_extent().transformed(inv)
    left_edge = bbox_novel.x0

    # Reference legend (left of novel legend)
    leg_ref = ax.legend(
        handles=ref_handles,
        labels=ref_labels,
        title='$' + ref_label + '$' + '\n' + '$' + ltitle + '$',
        alignment='center',
        loc='upper right',
        framealpha=0.85,
        edgecolor='0.75',
        handlelength=1.0,
        borderpad=0.6,
        bbox_to_anchor=(left_edge - 0.01, 1.0),
        bbox_transform=ax.transAxes,
        scatterpoints=1,
    )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlabel(rf'${x_label}$')
    ax.set_ylabel(rf'${y_label}$')

    ax.tick_params(axis='both', which='major',
                   direction='in', top=True, right=True, length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='minor',
                   direction='in', top=True, right=True, length=2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_filename is None:
        ref_handle = ref_filename[:-4]
        novel_handle = novel_filename[:-4]
        out_filename = f'{ref_handle}_vs_{novel_handle}_pareto_scatter.svg'

    plt.savefig(out_filename, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_filename}')