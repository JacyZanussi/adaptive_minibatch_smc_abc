### Adaptive Schemes for SMC ABC
from smc_abc import smc_abc_iterator as abc_iter
import numpy as np


### Constant minibatch 
# p = [n_0]
def constant_init(init_params,p=[1]):
    est = abc_iter(**init_params)
    est.batch_size = p[0]
    est.esv = lambda S: np.linalg.trace(est.W @ S)
    est.v_total_est = np.inf
    est.init_sigma = est.esv(est.W)
    def estimate_v_total(est):
        delta = est.posterior_stats - est.posterior_stats_ref
        if delta.ndim <= 2:
            Sigma = np.cov(delta.T)
        else: #The dimensions are (N particles by N_bs batch size by N_s stats)
            cov_list = np.array([np.cov(x.T) for x in delta]) #(N by N_s by N_s)
            Sigma = np.mean(cov_list,axis=0)
        est.Sigma = Sigma
        est.v_total_est = est.esv(Sigma)
    est.estimate_v_total = estimate_v_total
    return est
def constant_loop(est,ess_resample=True,ess_prop = 0.5):
    est.generate()
    if ess_resample:
        est.ESS_resample(ess_prop)
    est.estimate_v_total(est)
    v = est.v_total_est
    nt = est.batch_size
    N = est.sample_size
    esq = est.next_alpha_threshold**2
    est.c_est_ = np.sqrt((v/esq)*(1/nt - 1/N))
    est.rhs = v/N + (est.c_est_)**2 * esq
    est.lhs = v/nt
    snr = v/(nt*esq)
    print(f"Estimated v = {v:.3f}. Noise: v/n = {v/nt:.3f}. v/N = {v/N:.3f}. Estimated c = {est.c_est_:.3f}. SNR: v/(ne^2) = {snr:.3f}. ")


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


### Full Variance Control
# p = [n_0, c, lambda]
def fvc_init(args,p=[3,0.1,0.1]):
    est = constant_init(args)
    est.batch_size = p[0]
    est.c = p[1]
    est.lambda_ = p[2] #since "lambda" is a special word, add an undercore
    l = p[2]
    N = est.sample_size
    W = est.W
    est.esv = lambda S: np.linalg.trace(W @ S)
    def update_mbs():
        delta = est.posterior_stats - est.posterior_stats_ref
        if delta.ndim <= 2:
            Sigma = np.cov(delta.T)
        else: #The dimensions are (N particles by N_bs batch size by N_s stats)
            cov_list = np.array([np.cov(x.T) for x in delta]) #(N by N_s by N_s)
            Sigma = np.mean(cov_list,axis = 0) #(N_s,N_s)
        est.Sigma = Sigma
        v_total = est.esv(Sigma)
        # Exponential Moving Average update v_total
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.c_est = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.current_alpha_threshold**2))

        #automatic c selection
        if est.c is None:
            est.c = np.sqrt((est.v_total*(N - est.batch_size))/(est.batch_size*N*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        est.batch_size = est.v_total * N / (N*(est.c*est.alpha_threshold)**2 + est.v_total)
        print(f"Variance estimate: {2*est.v_total/N + (est.c*est.alpha_threshold)**2}. C estimate: {est.c_est}. V_total: {est.v_total}. batch size {est.batch_size}")
    est.update_mbs = update_mbs
    return est


def fvc_v2_init(args,p=[3,0.1,0.1]):
    est = constant_init(args)
    est.batch_size = p[0]
    est.c = p[1]
    est.lambda_ = p[2] #since "lambda" is a special word, add an undercore
    l = p[2]
    N = est.sample_size
    W = est.W
    est.esv = lambda S: np.linalg.trace(W @ S)
    def update_mbs():
        delta = est.posterior_stats - est.posterior_stats_ref
        if delta.ndim <= 2:
            Sigma = np.cov(delta.T)
        else: #The dimensions are (N particles by N_bs batch size by N_s stats)
            cov_list = np.array([np.cov(x.T) for x in delta]) #(N by N_s by N_s)
            Sigma = np.mean(cov_list,axis = 0) #(N_s,N_s)
        est.Sigma = Sigma
        v_total = est.esv(Sigma)
        # Exponential Moving Average update v_total
        est.v_total = (1-l) * est.v_total + l * v_total if est.generation > 1 else v_total
        est.c_est = np.sqrt(est.v_total/(est.batch_size*est.current_alpha_threshold**2))

        #automatic c selection
        if est.c is None:
            est.c = np.sqrt(est.v_total/(est.batch_size*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        est.batch_size = est.v_total / ((est.c*est.alpha_threshold)**2)
        print(f"Variance estimate: {2*est.v_total/N + (est.c*est.alpha_threshold)**2}. C estimate: {est.c_est}. V_total: {est.v_total}. batch size {est.batch_size}")
    est.update_mbs = update_mbs
    return est

def fvc_loop(est,ess_resample=True,ess_prop = 0.5):
    constant_loop(est,ess_resample,ess_prop)
    est.update_mbs()