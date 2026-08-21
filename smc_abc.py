'''
SMC ABC implementation suited for minibatch sampling.

Author: Jacy Zanussi
'''

import os
import sys
import time

import numpy as np

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

# Number of Ledoit-Wolf shrinkage-covariance bootstrap draws used when the summary
# statistic is scalar per batch (see __init__, dist_func='mahalanobis' branch).
MAHALANOBIS_BOOTSTRAP_SAMPLES = 1000
# Cap on the number of observations subsampled to estimate the Mahalanobis weight
# matrix, expressed as a multiple of the number of summary statistics.
MAHALANOBIS_SUBSAMPLE_STATS_MULTIPLIER = 50
# Floor used to keep the importance density strictly positive in log-space.
IMPORTANCE_DENSITY_MIN_DENSITY = 1e-300




class smc_abc_iterator:
    """
    Sequential Monte Carlo ABC iterator with adaptive minibatch/replicate sampling.

    Each call to `generate()` (or one iteration of `generate_until_stop()`) performs one
    SMC generation: it draws proposals (from the prior on generation 0, otherwise from a
    Gaussian perturbation kernel around the current weighted posterior), simulates and
    accepts particles whose distance to the observed data is below the current threshold,
    then updates importance weights, the perturbation kernel, and the acceptance threshold
    for the next generation. Simulation and acceptance-rejection are parallelized across
    particles with joblib.

    Typical usage is via the wrapper schemes in `smc_abc_schemes.py`
    (e.g. `schemes.constant_init` / `schemes.constant_loop`), which configure this class
    for a particular minibatch-adaptation strategy rather than constructing it directly.
    """

    def __init__(self,data,model,stats_func,prior,dist_func='mahalanobis',mweight_mat = None,num_particles=1000,alpha=0.5,ess_prop = 0.5, seed = None,
                sample_size = np.inf,batch_size=np.inf,batch_size_min=2,batch_size_max=np.inf,num_reps=1,sample_with_replacement=True, print_output = True,
                cores=-1,parallel_batch_size='auto',parallel_args = {}, rcond = 1e-15, epsilon = 1e-6, low_mem = False):
        """
        Parameters
        ----------
        data : np.ndarray or sequence
            The full observed dataset; `model`/`stats_func` are called on index subsets of it.
        model : callable
            `model(params, batch_indices, seed=None) -> (sim, ref_sim)`, simulating from `params`
            for the observations at `batch_indices`. Must accept a `seed` keyword or fall back
            gracefully (a `TypeError` on the call without `seed` is caught in `parloop`).
        stats_func : callable
            `stats_func(sim, ref_sim) -> (sim_stats, ref_stats)` computing summary statistics.
        prior : list
            `[sample_func, prior_domain, density_func]`, e.g. from `smc_abc_utils.uniform_prior`.
        dist_func : {"mahalanobis", None} or callable
            "mahalanobis" estimates a Mahalanobis weight matrix from the data (see below);
            None uses the unweighted L2 `weighted_dist`; or supply a callable
            `dist_func(sim_stats, ref_stats, threshold) -> (distance, distance < threshold)`.
        mweight_mat : np.ndarray or None
            Precomputed Mahalanobis precision (inverse-covariance) matrix. If None and
            `dist_func == 'mahalanobis'`, it is estimated from a subsample of `data` using
            Ledoit-Wolf shrinkage.
        num_particles : int
            Number of particles kept in the posterior each generation.
        alpha : float
            Target acceptance rate per generation; `accepted_per_generation = ceil(num_particles/alpha)`.
        ess_prop : float
            Proportion of `num_particles` below which `ESS_resample` triggers resampling.
        seed : int or None
            Seed for the main RNG and the `SeedSequence` used to spawn per-particle seeds.
        sample_size : int
            Number of observations to treat `data` as containing; defaults to `len(data)`.
        batch_size, batch_size_min, batch_size_max : float
            Initial/min/max minibatch size (number of observations per simulated replicate).
        num_reps : int
            Initial number of stochastic replicates simulated per particle.
        sample_with_replacement : bool
            Whether minibatches are drawn with replacement.
        print_output : bool
            Default verbosity for `post_generation_step`'s per-generation summary line.
        cores : int
            Number of joblib worker processes (`-1` = all available).
        parallel_batch_size : str or int
            Passed through to joblib's `Parallel(batch_size=...)`.
        parallel_args : dict
            Extra keyword arguments forwarded to joblib's `Parallel(...)`.
        rcond : float
            Cutoff for small singular values used when pseudo-inverting the Mahalanobis covariance.
        epsilon : float
            Diagonal regularization added to the perturbation kernel covariance for numerical stability.
        low_mem : bool
            If True, store simulated statistics/distances as float32 instead of float64.
        """
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
        self.seed = seed
        self._main_rng = np.random.default_rng(seed)

        ## Distance function initialization
        self.rcond = rcond
        if dist_func is None:
            self.dist_func = self.weighted_dist
        elif dist_func == 'mahalanobis':
            self.__init_mahalanobis__(mweight_mat)
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
        self.epsilon = epsilon
        
        ## Auxiliary or updated per generation
        self.seed = seed
        self._seedseq = np.random.SeedSequence(seed)
        self._main_rng = np.random.default_rng(seed)
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
        self.parallel_args = parallel_args

    def __init_mahalanobis__(self, mweight_mat):
        """Set up the Mahalanobis distance function, its weight matrix, and the v_total estimator."""
        if mweight_mat is None:
            # Minor safety net for larger datasets
            _,stats_obs = self.stats_func(self.data,self.data)
            num_stats = stats_obs.shape[1] if stats_obs.ndim > 1 else stats_obs.shape[0]
            size = np.minimum(self.sample_size, MAHALANOBIS_SUBSAMPLE_STATS_MULTIPLIER*num_stats)
            batch = self._main_rng.choice(self.sample_size,size)
            data_batched = self.data[batch] if type(self.data) == np.ndarray else [self.data[i] for i in batch]
            _,stats_obs = self.stats_func(data_batched,data_batched)
            if stats_obs.ndim <= 1:
                # Per-batch (scalar) summary statistic: bootstrap resample the whole dataset to
                # approximate a covariance across observations, since there is no batch axis to use directly.
                boot_stats = np.empty((MAHALANOBIS_BOOTSTRAP_SAMPLES, stats_obs.shape[0]))  # (n_boot, N_s)
                for b in range(MAHALANOBIS_BOOTSTRAP_SAMPLES):
                    idx = self._main_rng.choice(self.sample_size, size=self.sample_size, replace=True)
                    data_boot = self.data[idx] if isinstance(self.data, np.ndarray) else [self.data[i] for i in idx]
                    _, s_boot = self.stats_func(data_boot,data_boot)
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
                # Covariance of (N_particles, N_stats): variability of the summary statistic
                # across particles, used as a proxy for its variability across observations.
                Sigma = np.cov(delta.T)/est.batch_size
            else: #The dimensions are (N particles by N_bs batch size by N_s stats)
                delta_reshaped = delta.reshape(est.num_particles, est.batch_size, est.reps, delta.shape[-1]) #Handles replicates
                batch_means = np.mean(delta_reshaped, axis=2)
                batch_centers = batch_means.mean(axis=1, keepdims=True)
                diff = batch_means - batch_centers
                Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(est.num_particles * (est.batch_size - 1))
            est.Sigma = Sigma
            est.v_total_est = est.esv(Sigma)
        self.estimate_v_total = estimate_v_total


    ###### Step 1: Generation Step - Samples from the proposal distribution, simulates, and accepts particles. Updates internal data to reflect this.
    def generation_step(self):
        """Snapshot the current posterior/weights as "last generation", then run the parallel accept-reject loop."""
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
        """
        Draw `accepted_per_generation` accepted particles in parallel via `parloop`, storing the
        resulting parameters, statistics, distances, and the observation batch each particle used.
        """
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

        rind = self._main_rng.choice(self.sample_size,self.__batch_size_round__)
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
        for p,params,sim_stats, dist, bi, ns in Parallel(n_jobs=self.cores,prefer='processes', temp_folder=os.environ.get('JOBLIB_TEMP_FOLDER'),
                     batch_size='auto',backend='loky',return_as = 'generator',**self.parallel_args)(
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
 


    #generic sampling function for generations > 0: perturb a weighted-resampled particle with the current kernel, rejecting draws outside the prior domain
    #Can be generalized
    def sample_func(self, rng=None):
        """Draw one proposal: resample a posterior particle by weight, then perturb it with the Gaussian kernel, rejecting draws outside the prior domain."""
        if rng is None:
            rng = self._main_rng
        prior_domain = self.prior[1]
        in_prior_domain = False
        while not in_prior_domain:
            w = self.weights
            proposal_center = self.posterior[rng.choice(w.shape[0],p=w)]
            if self.num_params == 1:
                p = rng.normal(proposal_center,self.kernel_std)
                in_prior_domain = (prior_domain[0] <= p) and (p <= prior_domain[1])
            elif self.num_params > 1:
                p = rng.multivariate_normal(proposal_center, self.kernel_var)
                in_prior_domain = np.all(prior_domain[0] <= p) and np.all(p <= prior_domain[1])
            else:
                print("Something's wrong with the sample function")
        return p


    # defines the number of particles accepted per generation
    @property
    def accepted_per_generation(self):
        """Number of particles to accept this generation so that, at the target rate `alpha`, `num_particles` survive."""
        return int(np.ceil(self.num_particles/self.alpha))


    ###### Step 2: Post-generation Step - Update distance threshold, weights, kernel, and performance statistics, then prints it.
    def post_generation_step(self,print_output = True):
        """Select the best particles, update importance weights and the perturbation kernel, update timing/acceptance stats, and optionally print a summary."""
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
        """Keep the `num_particles` accepted particles with the smallest distances; the next threshold is the largest kept distance."""
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
        """Weights are uniform on generation 0 (sampled from the prior); afterwards, weight = prior_density / importance_density, normalized to sum to 1."""
        if self.generation == 0:
            self.weights = np.ones((self.num_particles,)) / self.num_particles
        else:
            self.__importance_density__()
            prior_densities = self.prior[2](self.posterior)
            weights = prior_densities / self.importance_density
            self.weights = weights / np.sum(weights)
    

    def __importance_density__(self):
        """Importance density of each current particle under the previous generation's weighted Gaussian-kernel mixture, computed in log-space for stability."""
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
        self.importance_density = np.maximum(self.importance_density, IMPORTANCE_DENSITY_MIN_DENSITY)
        del log_importance_density, log_densities, log_weighted, mvn
        assert np.all(self.importance_density > 0), "Importance density contains zeros or negatives"


    def __update_kernel_var__(self):
        """Recompute the Gaussian perturbation kernel as 2x the weighted posterior covariance (Beaumont et al. rule of thumb), with a small diagonal regularizer."""
        # Save the previous kernel variance for possible diagnostics
        self.last_kernel_var = self.kernel_var if self.generation > 1 else np.nan
        posterior_particles = self.posterior
        w = self.weights[:, None]
        weighted_mean = np.sum(w * posterior_particles, axis=0)
        centered = posterior_particles - weighted_mean  # shape (N, d)
        weighted_cov = 2 * (w * centered).T @ centered  # shape (d, d)
        # Regularization for numerical stability
        weighted_cov += np.eye(weighted_cov.shape[0]) * self.epsilon
        # If 1D, reduce to scalar
        weighted_cov = weighted_cov if weighted_cov.shape != (1, 1) else weighted_cov[0, 0]
        # Save kernel std (vector or scalar)
        self.kernel_var = weighted_cov
        if self.num_params == 1:
            self.kernel_std = np.sqrt(weighted_cov)


    ### __update_comp_stats__
    def __update_comp_stats__(self):
        """Update per-generation timing, acceptance rate, and the log-HDPR-volume convergence diagnostic."""
        self.step_time = time.time() - self.__tic__
        self.total_time += self.step_time
        self.generation += 1 
        self.total_sims += self.num_sims
        self.acceptance_rate = self.accepted_per_generation/self.num_sims
        _,_,interval_lengths = self.hdpr_marginal()
        self.log_hdpr_product = np.log(np.prod(interval_lengths))
            
        
    ###### utility functions and properties
    # Generation - wraps up generation step and post generation step. Separated for flexibility
    def generate(self,print_output=True):
        """Run one full SMC generation: sample/simulate/accept, then update weights, kernel, and stats."""
        self.generation_step()
        self.post_generation_step(print_output=print_output)

    def generate_until_stop(self,continue_func,loop_func = None,print_output=True):
        """Repeatedly call `loop_func` (default: one `generate()`) while `continue_func(self)` is True."""
        cont = continue_func(self)
        def loop_(est):
            est.generate(print_output)
        loop = loop_ if loop_func is None else loop_func
        while cont == True:
            loop(self)
            cont = continue_func(self)
   
    # Default distance function
    def weighted_dist(self,x,y,threshold,sigma=1,ord = 2):
        """Unweighted (or scalar-weighted) L2/Lp distance between two summary-statistic vectors."""
        dist = np.linalg.norm((x-y)*sigma,ord=ord)
        return dist, (dist < threshold)
    
    # Batch Size - this property stores a float to reduce rounding error for recursive updates.
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
        """Re-simulate-free reference statistics for each surviving particle's own observation batch, for use in variance estimation."""
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
        """Resample the posterior (with replacement, by weight) and reset weights to uniform if ESS falls below `proportion * num_particles`."""
        if proportion is None:
            proportion = self.ess_prop
        if self.ESS <= np.ceil(self.num_particles * proportion):
            self.posterior = self.posterior[self._main_rng.choice(range(self.num_particles),p=self.weights,size=self.num_particles,replace = True)]
            self.weights = np.ones((self.num_particles,)) / self.num_particles 
            print("Resampling")
    @property
    def ESS(self):
        """Effective sample size of the weighted posterior."""
        return 1.0/np.sum(self.weights**2)

    def hdpr_marginal(self,alpha = 0.05):
        """Per-parameter marginal (1-alpha) highest-density posterior region: MAP, interval(s), and total interval length for each parameter."""
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
    """
    Worker for one accepted particle: repeatedly propose+simulate+accept/reject minibatches
    until one satisfies `dist_func(...) < threshold`, then return it.

    Batches of candidate observation-index draws are generated 100 (or `resample_batch_size`)
    at a time to amortize the cost of RNG calls across many rejected simulations.

    Returns
    -------
    tuple : (particle_index p, accepted params, accepted sim stats, distance, batch indices used, number of simulations tried)
    """
    #Seeding
    global_int = int(seed.generate_state(1)[0])
    rng = np.random.default_rng(global_int)

    num_sims = 0
    dist_criteria = False
    while not dist_criteria:
        #generate a list of choices
        if replace:
            batch_indices = rng.choice(sample_size,size=(100,batch_size),replace=replace)
        else:
            batch_indices = np.array([rng.permutation(sample_size)[:batch_size] for _ in range(resample_batch_size)])
        for batch_idx in batch_indices:
            num_sims += 1
            params = np.array(sample_func(rng)) if generation > 0 else np.array(prior[0](rng))
            sim_seed = int(rng.integers(np.iinfo(np.int64).max))
            try:
                sim,ref_sim = model(params,batch_idx,seed=sim_seed)
            except TypeError:
                sim,ref_sim = model(params,batch_idx)
            sim_stats,ref_sim_stats = stats_func(sim,ref_sim)
            dist,dist_criteria = dist_func(sim_stats,ref_sim_stats,threshold)
            if dist_criteria:
                break
    return p, params, sim_stats, dist, batch_idx, num_sims


def hdpr(x, w, c):
    """
    Highest-density posterior region for a 1D weighted sample.

    Estimates the density of `x` (weighted by `w`) via a weighted Gaussian KDE, then finds the
    smallest set of (possibly disconnected) intervals whose density exceeds a threshold chosen
    so that the enclosed probability mass equals `c`.

    Returns
    -------
    MAP : float
        Location of maximum estimated density.
    intervals : np.ndarray, shape (n_intervals, 2)
        The interval(s) making up the credible region.
    length : float
        Total length of all intervals combined.
    """
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

