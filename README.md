# Adaptive Minibatch SMC-ABC

Code for adaptive minibatch Sequential Monte Carlo Approximate Bayesian Computation
(SMC-ABC), applied to simulation-based inference for stochastic biological models
(Lotka-Volterra predator-prey dynamics, transcriptional dynamics/smFISH snapshots, and the
Kilic gene-expression model).

> **Citation:** to be added upon publication.

## What this is

Standard SMC-ABC simulates a fixed number of observations ("minibatch size") per particle
each generation. This repository implements schemes that instead *adapt* the minibatch size
(and, for stochastic simulators, the number of replicates) each generation to control the
total variance of the posterior-selection step, trading simulation cost for accuracy more
efficiently than a fixed schedule. See `smc_abc_schemes.py` for the available schemes
(`constant` baseline and `fvc` variance-controlled batch size).

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

See [example_gmm.ipynb](example_gmm.ipynb) for a minimal, self-contained example (2D Gaussian
mixture model) showing how to build a `model`/`stats_func`/`prior`, initialize a scheme, and
run generations. [example_predator_prey.ipynb](example_predator_prey.ipynb) and
[example_transcriptional_dynamics.ipynb](example_transcriptional_dynamics.ipynb) apply the
same pattern to the biological models used in the paper.

Minimal usage pattern:

```python
import smc_abc_schemes as schemes

init_params = dict(data=data, model=model, stats_func=stats_func, prior=prior, num_particles=1000)
est = schemes.constant_init(init_params, p=[32])       # or schemes.fvc_init(init_params, p=[n0, c, lambda_])
while not est.generation >= 20:
    schemes.constant_loop(est)                          # or schemes.fvc_loop(est)
```

## Repository structure

| Path | Purpose |
|---|---|
| `smc_abc.py` | Core `smc_abc_iterator` class implementing one SMC-ABC generation (sampling, parallel accept-reject, weighting, kernel update). |
| `smc_abc_schemes.py` | Minibatch/replicate adaptation schemes (`constant`, `fvc`) built on top of `smc_abc.py`. |
| `smc_abc_utils.py` | Priors, stopping-criterion helpers, and simulator benchmarking utilities. |
| `lotka_volterra.py`, `transcriptional_dynamics.py` | Stochastic simulators for the two biological case studies. |
| `experiments.py` | Model wrappers (`lotka_volterra_step`, `transcriptional_dynamics_step`), summary statistics, experiment-sweep runner (`simulate_experiment`), and result-processing/plotting helpers (`get_attr`, `pareto_frontier`, `time_series`, ...). |
| `experiment_hyperparameters.py`, `experiment_physical_parameters.py`, `experiment_heterogeneities.py` | SLURM-array entry points reproducing paper Figures 2, 3, and 4 respectively (see each file's docstring for the array-index-to-sweep mapping). |
| `generate_synthetic_data.py` | One-off script that generates the reference datasets in `datasets/`. |
| `plotting_td_lv.py` | Generates the TD/LV paper figures from pickled results in `results/` (not included; produced by the `experiment_*.py` scripts). |
| `kilic/` | Application to the Kilic et al. gene-expression dataset (model, simulation, and plotting scripts). |
| `job_scripts/` | SLURM submission scripts for the HPC cluster used to run the sweeps. |
| `datasets/` | Reference/observed datasets (synthetic + the Kilic experimental dataset). |
| `*.ipynb` | Worked examples and figure-generation notebooks. |

## Reproducing paper results

1. `python generate_synthetic_data.py` to (re)create `datasets/lv_stochastic.pkl` and `datasets/td.pkl`.
2. Submit `job_scripts/hps_job.sh`, `job_scripts/pps_job.sh`, `job_scripts/hs_job.sh` (or run
   `experiment_hyperparameters.py` / `experiment_physical_parameters.py` /
   `experiment_heterogeneities.py` directly with a manual array index) to populate `results/`.
3. Run `python plotting_td_lv.py` to regenerate the figures from `results/`.
4. `job_scripts/kilic_job.sh` / `kilic/application_kilic.py` and
   `kilic/plot_kilic_smc_results.py` reproduce the Kilic application figure.

## License

See [LICENSE](LICENSE).
