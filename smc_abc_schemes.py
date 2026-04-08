### Adaptive Schemes for SMC ABC
from smc_abc import smc_abc_iterator as abc_iter
import numpy as np


### Constant minibatch 
# p = [n_0]
def constant_init(init_params,p=[1]):
    est = abc_iter(**init_params)
    est.batch_size = p[0]
    # est.esv = lambda S: np.linalg.trace(est.W_inv @ S)
    # est.v_total_est = np.inf
    # est.init_sigma = est.esv(est.W_inv)
    # def estimate_v_total(est):
    #     observed_stats_repeated = np.repeat(est.posterior_stats_ref, repeats = est.reps, axis=1)
    #     delta = est.posterior_stats - observed_stats_repeated
    #     if delta.ndim <= 2: 
    #         # NOTE: Critical flaw here. We can't bootstrap covariances of batches here, and this is covariance over particles, not observations.
    #         #Sigma = np.cov(delta.T)
    #         pass
    #     else: #The dimensions are (N particles by N_bs batch size by N_s stats)
    #         cov_list = np.array([np.cov(x.T) for x in delta]) #(N by N_s by N_s)
    #         Sigma = np.mean(cov_list,axis=0)
    #         delta_reshaped = delta.reshape(est.num_particles, est.batch_size, est.reps, delta.shape[-1]) #Handles replicates
    #         batch_means = np.mean(delta_reshaped, axis=2)
    #         batch_centers = batch_means.mean(axis=1, keepdims=True)
    #         diff = batch_means - batch_centers
    #         Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(est.num_particles * (est.batch_size - 1))
    #     est.Sigma = Sigma
    #     est.v_total_est = est.esv(Sigma)
    # est.estimate_v_total = estimate_v_total
    est.snr = np.inf
    return est
def constant_loop(est,ess_resample=True,ess_prop = 0.5):
    est.generate()
    if ess_resample:
        est.ESS_resample(ess_prop)
    est.estimate_v_total(est)
    v = est.v_total_est
    nt = est.batch_size
    N = est.sample_size
    esq = est.alpha_threshold**2
    est.c_est_ = np.sqrt((v/esq)*(1/nt - 1/N))
    snr = esq / (v / nt)
    est.snr = snr
    print(f"Estimated v_total = {v:.3f}. Noise: v/n = {v/nt:.3f}. Noise floor estimate: v/N = {v/N:.3f}. Estimated c = {est.c_est_:.3f}. SNR: eps^2 / (v/n) = {snr:.3f}. Time per sim: {est.step_time / est.num_sims:.3f}")





### Full Variance Control
# p = [n_0, c, lambda]
def fvc_init(args,p=[2,None,0.5]):
    est = constant_init(args)
    est.batch_size = p[0]
    est.c = p[1]
    est.lambda_ = p[2] #since "lambda" is a special word, add an undercore
    l = p[2]
    N = est.sample_size
    # W_inv = est.W_inv
    # est.esv = lambda S: np.linalg.trace(W_inv @ S)
    def update_mbs():
        # Exponential Moving Average update v_total
        v_total = est.v_total_est
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.c_est = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.current_alpha_threshold**2))
  
        # Automatic c selection
        if est.c is None:
            est.c = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        est.batch_size = est.v_total * N / (N*(est.c*est.alpha_threshold)**2 + est.v_total)
        print(f"Estimated v_total with ema: {est.v_total:.3f}. Batch size update: {est.batch_size}")
    est.update_mbs = update_mbs
    return est



def fvc_loop(est,ess_resample=True,ess_prop = 0.5):
    constant_loop(est,ess_resample,ess_prop)
    est.update_mbs()


###### Experimental
### Exponential
# p = [n_0, gamma]
def exponential_init(args,p=[32,1.1]):
    est = constant_init(args)
    est.batch_size_min = p[0]
    est.batch_size = p[0]
    est.factor = p[1]
    return est

def exponential_loop(est,ess_resample=True,ess_prop = 0.5):
    constant_loop(est,ess_resample,ess_prop)
    est.batch_size = est.batch_size_min * (est.factor ** est.generation)

### Full Variance Control of replicates
def fvc_rep_init(args,p=[2,2,None,0.5], asymptotic = False):
    '''
    Variance-control adaptive minibatch SMC ABC applied to adapting replicates. For stochastic simulators only. 
    
    Parameters
    ----------
    args : dictionary
        A dictionary of inputs to the SMC ABC class. Requires data, model, stats function, and prior at least.
    p : list
        Hyper parameters for the method: [initial minibatch size, c, lambda]
    asymptotic : Bool
        If False, uses the finite sample size formula for adapting replicates. 
    '''
    est = constant_init(args)
    est.batch_size = p[0]
    est.c = p[1]
    est.reps = p[2]
    est.lambda_ = p[3] #since "lambda" is a special word, add an undercore
    l = p[3]
    N = est.sample_size
    W_inv = est.W_inv
    est.esv = lambda S: np.linalg.trace(W_inv @ S)
    est.base_model = est.model #necessary for handling recursion
    est.model = lambda p,Bi : est.base_model(p,Bi,est.reps)
    def update_mbs():
        ## posterior stats must be shaped like: (N_particles, N_batch_size * N_reps, N_stats).
        ## The reps must be sequential like: [[batch0] * N_reps + [batch1] * N_reps + ...]
        observed_stats_repeated = np.repeat(est.posterior_stats_ref, repeats = est.reps, axis=1)
        delta = est.posterior_stats - observed_stats_repeated
        N_stats = delta.shape[-1]
        delta_reshaped = delta.reshape(est.num_particles, est.batch_size, est.reps, N_stats) #(N_particles, N_batch_size, N_reps, N_stats)
        #Calculate covariance within replicates, then average over replicates.
        rep_list = delta.reshape(-1, est.reps, N_stats)
        diff = rep_list - np.mean(rep_list, axis=1, keepdims=True)
        Sigma_rep = np.einsum('ijk,ijl->kl', diff, diff) / ((est.reps - 1) * est.num_particles * est.batch_size)
        
        batch_means = np.mean(delta_reshaped, axis=2)
        batch_centers = batch_means.mean(axis=1, keepdims=True)
        diff = batch_means - batch_centers
        Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(est.num_particles * (est.batch_size - 1))
        v_total_rep = est.esv(Sigma_rep)
        v_total = est.esv(est.Sigma)

        est.Sigma_rep = Sigma_rep
        est.Sigma = Sigma
        # Exponential Moving Average update v_total
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.v_total_rep = (1-l) * est.v_total_rep + l * v_total_rep if est.generation > 1 else v_total_rep
        est.c_est = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.current_alpha_threshold**2))

        #automatic c selection
        if est.c is None:
            if asymptotic:
                est.c = np.sqrt(est.v_total/(est.reps * est.batch_size * est.alpha_threshold**2))
            else:
                est.c = np.sqrt((est.v_total*(N - est.reps*est.batch_size))/(est.reps*est.batch_size*N*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        if asymptotic:
            num_reps = int(est.v_total / (est.batch_size * est.c**2 * est.alpha_threshold**2))
        else:
            num_reps = int(est.v_total * N / (N*(est.c*est.alpha_threshold)**2 + est.v_total))
        est.reps = np.maximum(num_reps,1)
        num_reps = est.reps
        est.model = lambda p,Bi : est.base_model(p,Bi,num_reps)
        print(f"Estimated v_total with ema: {est.v_total:.3f}. Batch size update: {est.batch_size}")
    est.update_mbs = update_mbs
    return est


## A mix of both where we update the vector [n,k].
def fvc_kn_init(args,p=[2,2,None,1.0], asymptotic = False):
    '''
    Variance-control adaptive minibatch SMC ABC applied to adapting replicates. For stochastic simulators only. 
    
    Parameters
    ----------
    args : dictionary
        A dictionary of inputs to the SMC ABC class. Requires data, model, stats function, and prior at least.
    p : list
        Hyper parameters for the method: [initial minibatch size, c, lambda, alpha]
    asymptotic : Bool
        If False, uses the finite sample size formula for adapting replicates. 
    '''
    est = constant_init(args)
    est.batch_size = p[0]
    est.reps = p[1]
    est.c = p[2]
    est.lambda_ = p[3] #since "lambda" is a special word, add an undercore
    l = p[3]
    N = est.sample_size
    W_inv = est.W_inv
    est.esv = lambda S: np.linalg.trace(W_inv @ S)
    est.base_model = est.model #necessary for handling recussion
    est.model = lambda p,Bi : est.base_model(p,Bi,est.reps)
    def update_mbs():
        ## posterior stats must be shaped like: (N_particles, N_batch_size * N_reps, N_stats).
        ## The reps must be sequential like: [[batch0] * N_reps + [batch1] * N_reps + ...]
        observed_stats_repeated = np.repeat(est.posterior_stats_ref, repeats = est.reps, axis=1)
        delta = est.posterior_stats - observed_stats_repeated
        N_stats = delta.shape[-1]
        delta_reshaped = delta.reshape(est.num_particles, est.batch_size, est.reps, N_stats) #(N_particles, N_batch_size, N_reps, N_stats)
        #Calculate covariance within replicates, then average over replicates.
        rep_list = delta.reshape(-1, est.reps, N_stats)
        diff = rep_list - np.mean(rep_list, axis=1, keepdims=True)
        Sigma_rep = np.einsum('ijk,ijl->kl', diff, diff) / ((est.reps - 1) * est.num_particles * est.batch_size)
        
        batch_means = np.mean(delta_reshaped, axis=2)
        batch_centers = batch_means.mean(axis=1, keepdims=True)
        diff = batch_means - batch_centers
        Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(est.num_particles * (est.batch_size - 1))
        v_total_rep = est.esv(Sigma_rep)
        v_total = est.esv(est.Sigma)

        est.Sigma_rep = Sigma_rep
        est.Sigma = Sigma
        # Exponential Moving Average update v_total
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.v_total_rep = (1-l) * est.v_total_rep + l * v_total_rep if est.generation > 1 else v_total_rep

        est.c_est = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.current_alpha_threshold**2))

        #automatic c selection
        if est.c is None:
            if asymptotic:
                est.c = np.sqrt(est.v_total/(est.reps * est.batch_size * est.alpha_threshold**2))
            else:
                est.c = np.sqrt((est.v_total*(N - est.reps*est.batch_size))/(est.reps*est.batch_size*N*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        #Sqrt because we want kn approximately c^2e^2 + vtotal?
        #Alternative: calculate k as is, then fill in the rest as n. We calculate:
        # C = v_total * N / (N*(est.c*est.alpha_threshold)**2 + v_total), then take n = C/k so that kn approx C.
        # if asymptotic:
        #     n = np.sqrt(est.v_total / (est.c**2 * est.alpha_threshold**2))
        # else:
        #     n = np.sqrt(est.v_total * N / (N * (est.c**2 * est.alpha_threshold**2) + est.v_total))
        # k = np.sqrt(est.v_total_rep / (est.c**2 * est.alpha_threshold**2))

        if asymptotic:
            C = est.v_total / (est.c**2 * est.alpha_threshold**2)
        else:
            C = est.v_total * N / (N * (est.c**2 * est.alpha_threshold**2) + est.v_total)
        k = np.sqrt(est.v_total_rep / (est.c**2 * est.alpha_threshold**2))
        n = C / k

        est.batch_size = n
        est.reps = int(np.maximum(k,2))
        num_reps = est.reps
        est.model = lambda p,Bi : est.base_model(p,Bi,num_reps)
        print(f"Within-batch variance: {est.v_total:.3e}. Within-replicate variance: {est.v_total_rep:.3e}. Batch size: {est.batch_size:.1f}. Reps: {num_reps:.1f}")
    est.update_mbs = update_mbs
    return est




## A mix of both where we update the vector [n,k].
def fvc_kn2_init(args,p=[2,2,None,1.0], asymptotic = False):
    '''
    Variance-control adaptive minibatch SMC ABC applied to adapting replicates. For stochastic simulators only. 
    
    Parameters
    ----------
    args : dictionary
        A dictionary of inputs to the SMC ABC class. Requires data, model, stats function, and prior at least.
    p : list
        Hyper parameters for the method: [initial minibatch size, c, lambda, alpha]
    asymptotic : Bool
        If False, uses the finite sample size formula for adapting replicates. 
    '''
    est = constant_init(args)
    est.batch_size = p[0]
    est.reps = p[1]
    est.c = p[2]
    est.lambda_ = p[3] #since "lambda" is a special word, add an undercore
    l = p[3]
    N = est.sample_size
    W_inv = est.W_inv
    est.esv = lambda S: np.linalg.trace(W_inv @ S)
    est.base_model = est.model #necessary for handling recussion
    est.model = lambda p,Bi : est.base_model(p,Bi,est.reps)
    def update_mbs():
        ## posterior stats must be shaped like: (N_particles, N_batch_size * N_reps, N_stats).
        ## The reps must be sequential like: [[batch0] * N_reps + [batch1] * N_reps + ...]
        observed_stats_repeated = np.repeat(est.posterior_stats_ref, repeats = est.reps, axis=1)
        delta = est.posterior_stats - observed_stats_repeated
        N_stats = delta.shape[-1]
        delta_reshaped = delta.reshape(est.num_particles, est.batch_size, est.reps, N_stats) #(N_particles, N_batch_size, N_reps, N_stats)
        #Calculate covariance within replicates, then average over replicates.
        rep_list = delta.reshape(-1, est.reps, N_stats)
        diff = rep_list - np.mean(rep_list, axis=1, keepdims=True)
        Sigma_rep = np.einsum('ijk,ijl->kl', diff, diff) / ((est.reps - 1) * est.num_particles * est.batch_size)
        
        batch_means = np.mean(delta_reshaped, axis=2)
        batch_centers = batch_means.mean(axis=1, keepdims=True)
        diff = batch_means - batch_centers
        Sigma = np.einsum('ijk,ijl->kl', diff, diff)/(est.num_particles * (est.batch_size - 1))
        v_total_rep = est.esv(Sigma_rep)
        v_total = est.esv(est.Sigma)

        est.Sigma_rep = Sigma_rep
        est.Sigma = Sigma
        # Exponential Moving Average update v_total
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.v_total_rep = (1-l) * est.v_total_rep + l * v_total_rep if est.generation > 1 else v_total_rep

        est.c_est = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.current_alpha_threshold**2))

        #automatic c selection
        if est.c is None:
            if asymptotic:
                est.c = np.sqrt(est.v_total/(est.reps * est.batch_size * est.alpha_threshold**2))
            else:
                est.c = np.sqrt((est.v_total*(N - est.reps*est.batch_size))/(est.reps*est.batch_size*N*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        #Sqrt because we want kn approximately c^2e^2 + vtotal?
        #Alternative: calculate k as is, then fill in the rest as n. We calculate:
        #C = v_total * N / (N*(est.c*est.alpha_threshold)**2 + v_total), then take n = C/k so that kn approx C.
        if asymptotic:
            n = np.sqrt(est.v_total / (est.c**2 * est.alpha_threshold**2))
        else:
            n = np.sqrt(est.v_total * N / (N * (est.c**2 * est.alpha_threshold**2) + est.v_total))
        k = np.sqrt(est.v_total_rep / (est.c**2 * est.alpha_threshold**2))

        # if asymptotic:
        #     C = est.v_total / (est.c**2 * est.alpha_threshold**2)
        # else:
        #     C = est.v_total * N / (N * (est.c**2 * est.alpha_threshold**2) + est.v_total)
        # k = np.sqrt(est.v_total_rep / (est.c**2 * est.alpha_threshold**2))
        # n = C / k

        est.batch_size = n
        est.reps = int(np.maximum(k,2))
        num_reps = est.reps
        est.model = lambda p,Bi : est.base_model(p,Bi,num_reps)
        print(f"Within-batch variance: {est.v_total:.3e}. Within-replicate variance: {est.v_total_rep:.3e}. Batch size: {est.batch_size:.1f}. Reps: {num_reps:.1f}")
    est.update_mbs = update_mbs
    return est