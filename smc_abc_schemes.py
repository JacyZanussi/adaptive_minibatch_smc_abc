"""
Adaptive minibatch schemes for SMC ABC.

Each scheme wraps `smc_abc_iterator` (`smc_abc.py`) with a particular strategy for choosing
the minibatch size (and, for the `*_rep`/`*_kn*` variants, the number of stochastic
replicates) each generation. Every scheme follows the same two-function pattern:

- `<scheme>_init(init_params, p=[...])`: builds the `smc_abc_iterator` and attaches any
  scheme-specific state/hyperparameters (from the positional list `p`) to it.
- `<scheme>_loop(est, ...)`: runs one `est.generate()` and then updates the scheme's
  hyperparameters (e.g. batch size) for the next generation.

`constant` is the fixed-batch-size baseline; `fvc` ports adapt batch size (and, for the
`_rep`/`_kn*` variants, replicate count) to keep the total posterior-selection variance
near a target level ("Full Variance Control"). See each function's docstring for the
meaning of its `p` hyperparameter list.
"""
from smc_abc import smc_abc_iterator as abc_iter
import numpy as np


### Constant minibatch 
# p = [n_0]
def constant_init(init_params,p=[1]):
    """Build an smc_abc_iterator with a fixed minibatch size `p[0]` (the baseline, non-adaptive scheme)."""
    est = abc_iter(**init_params)
    est.batch_size = p[0]
    est.snr = np.inf
    return est
def constant_loop(est,ess_resample=True,ess_prop = 0.5):
    """
    Run one generation, then estimate the total posterior-selection variance `v_total` and the
    resulting signal-to-noise ratio `snr = alpha_threshold^2 / (v_total / batch_size)`, which the
    `fvc*` schemes use to adapt batch size/replicates. Shared by all scheme `_loop` functions.
    """
    est.generate()
    if ess_resample:
        est.ESS_resample(ess_prop)
    # Compute values for printing
    est.estimate_v_total(est)
    v_hat = est.v_total_est
    batch_size = est.batch_size
    sample_size = est.sample_size
    alpha_threshold = est.alpha_threshold**2
    est.c_est_ = np.sqrt((v_hat/alpha_threshold)*(1/batch_size - 1/sample_size))
    snr = alpha_threshold / (v_hat / batch_size)
    est.snr = snr
    print(f"Estimated v_total = {v_hat:.3f}. Noise: v/n = {v_hat/batch_size:.3f}. Noise floor estimate: v/N = {v_hat/sample_size:.3f}. Estimated c = {est.c_est_:.3f}. SNR: eps^2 / (v/n) = {snr:.3f}. Time per sim: {est.step_time / est.num_sims:.3f}")


### Full Variance Control
# p = [n_0, c, lambda]
def fvc_init(args,p=[2,None,0.5]):
    """
    Full Variance Control: adapts batch size each generation to keep the signal-to-noise ratio
    near a target level.

    Parameters
    ----------
    args : dict
        Inputs forwarded to `smc_abc_iterator` (data, model, stats_func, prior, ...).
    p : list
        `[n_0, c, lambda_]`: initial batch size; target noise-to-threshold ratio `c` (if None,
        selected automatically from the first generation's variance estimate); EMA smoothing
        factor `lambda_` in (0, 1] for the running `v_total` estimate (1.0 = no smoothing).
    """
    est = constant_init(args)
    est.batch_size = p[0]
    est.c = p[1]
    est.lambda_ = p[2] # "lambda" is a special word, so an underscore is added to avoid confusion
    lambda_ema = p[2]
    sample_size = est.sample_size
    def update_mbs():
        """Exponential-moving-average the variance estimate, then set the batch size to hit the target noise level `c`."""
        # Exponential Moving Average update v_total
        v_total = est.v_total_est
        est.v_total = (1-lambda_ema) * est.v_total + lambda_ema * v_total if est.generation > 1 else v_total
        est.c_est = np.sqrt((est.v_total*(sample_size - est.batch_size))/(est.batch_size*sample_size*est.current_alpha_threshold**2))
  
        # Automatic c selection
        if est.c is None:
            est.c = np.sqrt((est.v_total*(sample_size - est.batch_size))/(est.batch_size*sample_size*est.alpha_threshold**2))
            new_hyperparams = [est.batch_size,est.c,est.lambda_]
            print(f"Automatic c selection: c = {est.c}")
            print(f"New hyperparameter set: {new_hyperparams}")
        est.batch_size = est.v_total * sample_size / (sample_size*(est.c*est.alpha_threshold)**2 + est.v_total)
        print(f"Estimated v_total with ema: {est.v_total:.3f}. Batch size update: {est.batch_size}")
    est.update_mbs = update_mbs
    return est
def fvc_loop(est,ess_resample=True,ess_prop = 0.5):
    """One `constant_loop` generation, followed by the Full-Variance-Control batch-size update."""
    constant_loop(est,ess_resample,ess_prop)
    est.update_mbs()
