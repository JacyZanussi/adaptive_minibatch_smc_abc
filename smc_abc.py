'''
SMC ABC implementation suited for minibatch sampling.

Author: Jacy Zanussi
'''

import os
import sys
import time

import numpy as np
from numpy import random as rnd

import joblib
from joblib import Parallel, delayed
from functools import partial

from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from sklearn.covariance import ledoit_wolf

#Necessary for parallel processing with joblib
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
    print("cwd appended to system path")




class smc_abc_iterator:
    """
    Sequential Monte Carlo ABC iterator for simulation-based inference.
    """

    def __init__(self,data,model,stats_func,prior,dist_func='mahalanobis',mweight_mat = None,num_particles=1000,alpha=0.5,ess_prop = 0.5, seed = None,
                sample_size = np.inf,batch_size=np.inf,batch_size_min=1,batch_size_max=np.inf,num_reps=1,sample_with_replacement=True, print_output = True,
                cores=-1,parallel_batch_size='auto',backend='loky', rcond = 1e-15, epsilon = 1e-6, low_mem = False):
        self.data = data
        self.model = model
        self.stats_func = stats_func
        ## prior
        prior_ = prior.copy() 
        prior_[1] = prior_[1].T
        self.prior = prior_

        ## Batch size specifications
        if sample_size < np.inf:
            self.sample_size = sample_size
        else:
            self.sample_size = int(data.shape[0]) if type(data) == np.ndarray else len(data)
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max 
        self.batch_size = batch_size
        self.reps = num_reps
        self.sample_with_replacement = sample_with_replacement

        ## Distance function initialization
        self.rcond = rcond
        if dist_func is None:
            self.dist_func = self.weighted_dist
        elif dist_func == 'mahalanobis':
            if mweight_mat is None:
                # Minor safety net for larger datasets
                _,stats_obs = stats_func(data,data)
                num_stats = stats_obs.shape[1] if stats_obs.ndim > 1 else stats_obs.shape[0]
                size = np.minimum(self.sample_size, 50*num_stats)
                batch = np.random.choice(self.sample_size,size)
                data_batched = data[batch] if type(data) == np.ndarray else [data[i] for i in batch]
                _,stats_obs = stats_func(data_batched,data_batched)
                #Take per-sample summary statistics of shape (sample_size,N_s), compute covariance.
                # if stats_obs.ndim > 1:
                #     stds = np.std(stats_obs, axis = 0, ddof=1)
                #     stds[stds == 0] = 1.0
                #     stats_scaled = stats_obs / stds
                #     shrunk_corr,_ = ledoit_wolf(stats_scaled)
                #     cov = np.outer(stds, stds) * shrunk_corr
                if stats_obs.ndim <= 1:
                    # NOTE: this is defunct. It turns out you can't easily bootstrap summary covariances per batch per particle (N_boot by N_particles)
                    # Per-batch summary: bootstrap to estimate covariance
                    n_bootstrap = 1000
                    boot_stats = np.empty((n_bootstrap, stats_obs.shape[0]))  # (n_boot, N_s)
                    for b in range(n_bootstrap):
                        idx = np.random.choice(self.sample_size, size=self.sample_size, replace=True)
                        data_boot = self.data[idx] if isinstance(self.data, np.ndarray) else [self.data[i] for i in idx]
                        _, s_boot = stats_func(data_boot,data_boot)
                        boot_stats[b] = s_boot # s_boot is shape (N_s,)
                    stats_obs = boot_stats #reassign. Should be shaped (1000,N_s)
                # ledoit_wolf for numerical stability of the inverse
                # Inverse is calculated directly as below to save computational time, as opposed to computing a solve each simulation for distance.
                stds = np.std(stats_obs, axis = 0, ddof=1)
                stds[stds == 0] = 1.0
                stats_scaled = stats_obs / stds
                shrunk_corr,_ = ledoit_wolf(stats_scaled)
                cov = np.outer(stds, stds) * shrunk_corr
                self.mahalanobis_cov = cov
                W_inv = np.linalg.pinv(cov,rcond = self.rcond)
                self.W_inv = W_inv
            else:
                self.W_inv = mweight_mat
            def mahalanobis(x,y,thr):
                x_ = np.mean(x,axis=0) if x.ndim > 1 else x
                y_ = np.mean(y,axis=0) if y.ndim > 1 else y
                diff = x_ - y_
                assert diff.ndim == 1
                dist_sq = diff.T @ self.W_inv @ diff 
                dist = np.sqrt(dist_sq)
                return dist, (dist < thr)
            self.dist_func = mahalanobis
            ### Additionally, we define effective scalar variance and v_total estimates here
            self.esv = lambda S: np.linalg.trace(self.W_inv @ S)
            self.v_total_est = np.inf
            self.init_sigma = self.esv(self.W_inv)
            def estimate_v_total(est):
                observed_stats_repeated = np.repeat(est.posterior_stats_ref, repeats = est.reps, axis=1)
                delta = est.posterior_stats - observed_stats_repeated
                if delta.ndim <= 2: 
                    # NOTE: Critical flaw here. We can't bootstrap covariances of batches here, and this is covariance over particles, not observations.
                    # This would be covariance of (N_particles, N_stats), which would give us covariance of stats across particles
                    #Sigma = np.cov(delta.T)
                    pass
                else: #The dimensions are (N particles by N_bs batch size by N_s stats)
                    cov_list = np.array([np.cov(x.T) for x in delta]) #(N by N_s by N_s)
                    Sigma = np.mean(cov_list,axis=0)
                    delta_reshaped = delta.reshape(self.num_particles, self.batch_size, self.reps, delta.shape[-1]) #Handles replicates
                    batch_means = np.mean(delta_reshaped, axis=2)
                    batch_centers = batch_means.mean(axis=1, keepdims=True)
                    diff = batch_means - batch_centers
                    Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(self.num_particles * (self.batch_size - 1))
                self.Sigma = Sigma
                self.v_total_est = self.esv(Sigma)
            self.estimate_v_total = estimate_v_total
        else:
            if callable:
                self.dist_func = dist_func
            else:
                raise ValueError('Valid distance function not supplied. Provide "mahalanobis" for Mahalanobis distance, nothing for L2 distance, or a callable that takes x,y,threshold and returns distance, (distance < threhsold)')

        ## SMC ABC hyperparameters
        self.num_particles = num_particles
        self.alpha = alpha
        self.alpha_threshold = np.inf
        self.next_alpha_threshold = np.inf
        
        # For the stopping function
        self.acceptance_rate = 1.0

        ## Parallelization parameters
        self.cores = cores
        self.backend = backend
        self.parallel_batch_size = parallel_batch_size
        self.epsilon = epsilon
        
        ## Auxiliary or updated per generation
        self.seed = seed
        self._seedseq = np.random.SeedSequence(seed)
        self.generation = 0
        sample = prior_[0]()
        self.num_params = sample.shape[0]
        self.total_time = 0
        self.total_sims = 0
        self.posterior = []
        self.weights = np.ones(shape=(self.num_particles,)) / self.num_particles
        self.ess_prop = ess_prop
        self.print_output = print_output
        vol = np.prod(prior[1][:,1] - prior[1][:,0])
        self.log_hdpr_product = vol #Set to the volume of the prior bounds. The posterior log HDPR can't be larger than that.
        self.dtype = np.float32 if low_mem else np.float64
        # warm up njit
        self.stats_func(*self.model(prior[0](),np.arange(self.sample_size)))


    ###### Step 1: Generation Step - Samples from the proposal distribution, simulates, and accepts particles. Updates internal data to reflect this.
    def generation_step(self):
        # Track time. Time calculated is only added to the total in the post-generation step. 
        self.__tic__ = time.time()
        # Storing the provided posterior and weights is necessary for the new weight calculations
        self.last_posterior = self.posterior
        self.last_weights = self.weights
        self.posterior_last_gen = self.posterior
        self.posterior_weights_last_gen = self.weights
        self.__parallelized_generation__()
        


    # __parallelized_generation__
    def __parallelized_generation__(self):
        #Generate seeds from seed generator
        child_seeds = self._seedseq.spawn(self.accepted_per_generation)

        parloop_partial = partial(
            parloop,
            batch_size = self.__batch_size_round__,
            sample_size = self.sample_size,
            generation = self.generation,
            sample_func = self.sample_func,
            prior = self.prior,
            model = self.model,
            stats_func = self.stats_func,
            dist_func = self.dist_func,
            threshold = self.alpha_threshold,
            replace = self.sample_with_replacement,
            resample_batch_size = self.accepted_per_generation 
        )

        rind = np.random.choice(self.sample_size,self.__batch_size_round__)
        dat = self.data[rind] if type(self.data) == np.ndarray else [self.data[i] for i in rind]
        s,_ = self.stats_func(dat,dat)
        self.stats_shape = s.shape

        thetas = np.empty((self.accepted_per_generation,self.num_params))
        #stats = np.empty((self.accepted_per_generation,*self.stats_shape),dtype=self.dtype)
        stats = []
        dists = np.empty(self.accepted_per_generation,dtype=self.dtype)
        batch_matrix = np.empty((self.accepted_per_generation,self.__batch_size_round__),dtype=int)
        num_sims = 0
        del s, rind
        for p,params,sim_stats, dist, bi, ns in Parallel(n_jobs=self.cores,prefer='processes', temp_folder=None,
                     batch_size=self.parallel_batch_size,backend=self.backend,return_as = 'generator')(
            delayed(parloop_partial)(p,seed = child_seeds[p]) for p in range(self.accepted_per_generation)
            ):
            thetas[p] = params
            #stats[p]  = sim_stats
            stats.append(sim_stats)
            dists[p]  = dist
            batch_matrix[p] = bi
            num_sims += ns
        stats = np.array(stats)
        self.accepted_particles = thetas
        self.accepted_stats = stats
        self.accepted_dists = dists
        self.batch_indices = batch_matrix
        self.num_sims = num_sims
        del thetas, stats, dists, num_sims
 


    #generic sampling function for generations > 0
    #Can be generalized
    def sample_func(self):
        prior_domain = self.prior[1]
        in_prior_domain = False
        while not in_prior_domain:
            w = self.weights
            p_ = self.posterior[rnd.choice(w.shape[0],p=w)]
            if self.num_params == 1:
                p = rnd.normal(p_,self.kernel_std)
                in_prior_domain = (prior_domain[0] <= p) and (p <= prior_domain[1])
            elif self.num_params > 1:
                p = rnd.multivariate_normal(p_, self.kernel_var)
                in_prior_domain = np.all(prior_domain[0] <= p) and np.all(p <= prior_domain[1])
            else:
                print("Something's wrong with the sample function")
        return p


    # defines the number of particles accepted per generation
    @property
    def accepted_per_generation(self):
        return int(np.ceil(self.num_particles/self.alpha))


    ###### Step 2: Post-generation Step - Update distance threshold, weights, kernel, and performance statistics, then prints it.
    def post_generation_step(self,print_output = True):
        self.__select_N_best__()
        self.__update_importance_weights__()
        self.__update_kernel_var__()
        self.__update_comp_stats__()
        # if print_output is None:
        #     print_output = self.print_output
        if print_output:
            print(f"Generation: {self.generation}. Batch Size: {self.__batch_size_round__}. Acceptance Rate: {self.acceptance_rate:.3f}. Step time: {self.step_time:.3f}. Total time {self.total_time:.3f}",end=". ")
            print(f"post-filter threshold: {self.next_alpha_threshold:.4f}. ESS: {self.ESS:.2f}. Log HDPR vol: {self.log_hdpr_product:.3f}")


    ### __select_N_best__
    #Selects the num_particles particles with the lowest distances. Sets alpha_threshold
    def __select_N_best__(self):
        p = self.accepted_particles
        s = self.accepted_stats
        d = self.accepted_dists
        dist_inds = np.argsort(d)
        nth_dist = d[dist_inds[self.num_particles-1]]
        self.current_alpha_threshold = self.alpha_threshold
        self.next_alpha_threshold = nth_dist
        self.alpha_threshold = nth_dist
        #select N best parameters by distance
        topn_index = dist_inds[:self.num_particles]
        self.posterior = p[topn_index]
        self.posterior_stats = s[topn_index]
        self.posterior_dists = d[topn_index]
        self.posterior_indices = topn_index
        del self.accepted_particles
        del self.accepted_stats
        del self.accepted_dists


    def __update_importance_weights__(self):
        if self.generation == 0:
            self.weights = np.ones((self.num_particles,)) / self.num_particles
        else:
            #Get importane density. update weights as prior_density / importance_density (normalized)
            self.__importance_density__()
            prior_densities = self.prior[2](self.posterior)
            weights = prior_densities / self.importance_density
            self.weights = weights / np.sum(weights)
    

    def __importance_density__(self):
        # assert np.isclose(np.sum(self.last_weights), 1, rtol=1e-15)
        log_densities = []
        for i in range(self.num_particles):
            mvn = multivariate_normal(mean=self.last_posterior[i],
                                    cov=self.kernel_var,
                                    allow_singular=True)
            log_densities.append(mvn.logpdf(self.posterior))  # shape: (num_particles,)
        log_densities = np.array(log_densities)  # shape: (num_particles, num_particles)
        log_weighted = log_densities + np.log(self.last_weights[:, None])
        log_importance_density = logsumexp(log_weighted, axis=0)
        self.importance_density = np.exp(log_importance_density)
        eps = 1e-300
        self.importance_density = np.maximum(self.importance_density, eps)
        del log_importance_density, log_densities, log_weighted, mvn
        assert np.all(self.importance_density > 0), "Importance density contains zeros or negatives"


    def __update_kernel_var__(self):
        # Save the previous kernel variance for possible diagnostics
        self.last_kernel_var = self.kernel_var if self.generation > 1 else np.nan
        u = self.posterior
        w = self.weights[:, None]
        mu = np.sum(w * u, axis=0)
        centered = u - mu  # shape (N, d)
        v = 2 * (w * centered).T @ centered  # shape (d, d)
        # Regularization for numerical stability
        v += np.eye(v.shape[0]) * self.epsilon
        # If 1D, reduce to scalar
        v = v if v.shape != (1, 1) else v[0, 0]
        # Save kernel std (vector or scalar)
        self.kernel_var = v
        if self.num_params == 1:
            self.kernel_std = np.sqrt(v)


    ### __update_comp_stats__
    def __update_comp_stats__(self):
        self.step_time = time.time() - self.__tic__
        self.total_time += self.step_time
        self.generation += 1 
        self.total_sims += self.num_sims
        self.acceptance_rate = self.accepted_per_generation/self.num_sims
        _,_,l = self.hdpr_marginal()
        self.log_hdpr_product = np.log(np.prod(l))
            
        
    ###### utility functions and properties
    # Generation - wraps up generation step and post generation step. Separated for flexibility
    def generate(self,print_output=True):
        self.generation_step()
        self.post_generation_step(print_output=print_output)

    def generate_until_stop(self,continue_func,loop_func = None,print_output=True):
        cont = continue_func(self)
        def loop_(est):
            est.generate(print_output)
        loop = loop_ if loop_func is None else loop_func
        while cont == True:
            loop(self)
            cont = continue_func(self)
   
    # Default distance function
    def weighted_dist(self,x,y,threshold,sigma=1,ord = 2):
        dist = np.linalg.norm((x-y)*sigma,ord=ord)
        return dist, (dist < threshold)
    
    # Batch Size - this propoerty stores a float to reduce rounding error for recursive updates.
    @property
    def batch_size(self):
        return self.__batch_size__
    @batch_size.setter
    def batch_size(self,value):
        self.__batch_size__ = np.maximum(np.minimum(value,self.batch_size_max),self.batch_size_min)
        self.__batch_size_round__ = round(self.__batch_size__)
    @batch_size.getter
    def batch_size(self):
        return self.__batch_size_round__
    #minimum batch size
    @property
    def batch_size_min(self):
        return self.__batch_size_min__
    @batch_size_min.setter
    def batch_size_min(self,value):
        self.__batch_size_min__ = int(np.maximum(value,2))
    #maximum batch size 
    @property
    def batch_size_max(self):
        return self.__batch_size_max__
    @batch_size_max.setter
    def batch_size_max(self,value):
        self.__batch_size_max__ = int(np.minimum(value,self.sample_size))
    @property
    def posterior_stats_ref(self):
        stats_ref = np.empty((self.num_particles,*self.stats_shape),dtype=self.dtype)
        posterior_batch_indices = self.batch_indices[self.posterior_indices]
        for i,batch_idx in enumerate(posterior_batch_indices):
            obs_subset = self.data[batch_idx] if type(self.data) == np.ndarray else [self.data[i] for i in batch_idx]
            stats_ref[i] = self.stats_func(obs_subset,obs_subset)[1]
        del posterior_batch_indices
        return stats_ref
    #repeats - This property stores a float to reduce rounding error for recursive updates.
    @property
    def reps(self):
        return self.__reps__
    @reps.setter
    def reps(self,value):
        self.__reps__ = np.maximum(value,1)
        self.__reps_round__ = round(np.round(self.__reps__))
    @reps.getter
    def reps(self):
        return self.__reps_round__
    

    ###### Additional Functions: Post-estimation, Prangle's adaptive distances, plotting, wrappers, etc
    ### ESS Resampling - Note: ESS resample will resample if not given a proportion
    def ESS_resample(self,proportion = None):
        if proportion is None:
            proportion = self.ess_prop
        if self.ESS <= np.ceil(self.num_particles * proportion):
            self.posterior = self.posterior[np.random.choice(range(self.num_particles),p=self.weights,size=self.num_particles,replace = True)]
            self.weights = np.ones((self.num_particles,)) / self.num_particles 
            print("Resampling")
    @property
    def ESS(self):
        return 1.0/np.sum(self.weights**2)

    def hdpr_marginal(self,alpha = 0.05):
        c = 1-alpha
        MAP = []
        intervals = []
        lengths = []
        for i in range(self.num_params):
            marginal_posterior = self.posterior[:,i]
            map,ints,lens = hdpr(marginal_posterior,self.weights,c)
            MAP.append(map)
            intervals.append(ints)
            lengths.append(lens)
        MAP = np.array(MAP)
        lengths = np.array(lengths)
        return MAP,intervals,lengths
    


def parloop(p,batch_size,sample_size,generation,sample_func,prior,model,stats_func,dist_func,threshold,replace,resample_batch_size,seed):
    #Seeding
    global_int = int(seed.generate_state(1)[0])
    np.random.seed(global_int)
    rng = np.random.default_rng(seed)

    num_sims = 0
    dist_criteria = False
    while not dist_criteria:
        #generate a list of choices
        if replace:
            batch_indices = rng.choice(sample_size,size=(100,batch_size),replace=replace)
        else:
            batch_indices = np.array([rng.permutation(sample_size)[:batch_size] for _ in range(resample_batch_size)])
        for bi in batch_indices:
            num_sims += 1
            params = np.array(sample_func()) if generation > 0 else np.array(prior[0]())
            sim,ref_sim = model(params,bi) 
            sim_stats,ref_sim_stats = stats_func(sim,ref_sim)
            dist,dist_criteria = dist_func(sim_stats,ref_sim_stats,threshold)
            if dist_criteria:
                break
    return p, params, sim_stats, dist, bi, num_sims
    #return params, sim_stats, ref_sim_stats, dist, num_sims


def hdpr(x, w, c):
    densities = gaussian_kde(x, weights=w)
    x_interp = np.linspace(np.min(x), np.max(x), num=1000)
    densities_interp = densities.pdf(x_interp)
    
    # Find density threshold
    sorted_densities = np.sort(densities_interp)[::-1]
    dx = x_interp[1] - x_interp[0]
    cumulative_prob = np.cumsum(sorted_densities * dx)
    cumulative_prob /= cumulative_prob[-1]  # normalize
    
    threshold_idx = np.where(cumulative_prob >= c)[0][0]
    density_threshold = sorted_densities[threshold_idx]
    
    # Find all regions above threshold
    high_density_mask = densities_interp >= density_threshold
    high_density_points = x_interp[high_density_mask]
    
    # Handle potentially disconnected intervals
    if len(high_density_points) == 0:
        return np.array([])
    
    # Find connected components
    diff = np.diff(high_density_points)
    breaks = np.where(diff > dx * 1.5)[0]  # gaps larger than grid spacing
    
    if len(breaks) == 0:
        MAP = x_interp[np.argmax(densities_interp)]
        intervals = np.array([np.min(high_density_points), np.max(high_density_points)])
        length = intervals[1] - intervals[0]
        return MAP, intervals, length
    else:
        MAP = x_interp[np.argmax(densities_interp)]
        # Return multiple intervals
        intervals = []
        start = 0
        for break_point in breaks:
            intervals.append([high_density_points[start], high_density_points[break_point]])
            start = break_point + 1
        intervals.append([high_density_points[start], high_density_points[-1]])
        intervals = np.array(intervals)
        length = np.sum(intervals.T[1] - intervals.T[0])
        return MAP, np.array(intervals), length

