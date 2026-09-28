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

The versions used for the paper are pinned in `paper/requirements.txt`. The paper's figure
scripts also need a LaTeX installation (matplotlib's `text.usetex`).

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
| `experiment_hyperparameters.py`, `experiment_physical_parameters.py`, `experiment_heterogeneities.py` | SLURM-array entry points for the hyperparameter, physical-parameter and heterogeneity sweeps (see each file's docstring for the array-index-to-sweep mapping). The fits behind the paper's figures are run by `paper/campaign.py`. |
| `generate_synthetic_data.py` | One-off script that generates the reference datasets `datasets/td.pkl` and `datasets/lv_stochastic.pkl`. |
| `kilic/` | Application to the *E. coli* mRNA counts of Kilic et al. (2023): telegraph-model simulator, paired summaries, SMC-ABC fit (`application_kilic.py`), and exact-likelihood reference posterior (`likelihood_reference.py`). |
| `paper/` | Reproduces the paper: every fit (`campaign.py`) and the figure scripts. |
| `job_scripts/` | SLURM submission scripts for the HPC cluster used to run the sweeps. |
| `datasets/` | Synthetic reference datasets, a Gillespie simulation at the Wang et al. (2019) slow-growth estimates (`Wang2020_SFig21_*.mat`), and the *E. coli* data (`kilic2023_fig4/`, `wang2019/`, each with a README giving its source). |
| `*.ipynb` | Worked examples and figure-generation notebooks. |

## Reproducing the paper

1. Run the fits: `python paper/campaign.py list <campaign>` lists them, and
   `python paper/campaign.py run <campaign> <index|range|all>` runs them (8 worker processes per
   fit by default). The campaigns are `main` (108 fits), `robustness` (1,610), `nsweep` (33) and
   `ecoli` (35). The fits are independent, so a campaign can be split across machines by index.
2. Compute the *E. coli* likelihood reference: `python kilic/likelihood_reference.py --run`.
3. Draw the figures into `paper/results/figures/`: `python paper/collect_results.py`, then
   `plotting_td_lv.py`, `plotting_posteriors.py`, `plotting_hdr_pairplot.py`,
   `plotting_nsweep.py` and `plot_ecoli.py` in `paper/`.
   The schematic figures are drawn from the pieces made in `cartoon_generation.ipynb`.

## License

See [LICENSE](LICENSE).
