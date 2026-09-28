"""
SMC ABC for the two-state telegraph model on the E. coli lacZ smFISH data.

Every observation is one fixed cell with its own covariate, the time at which it was imaged. The
data are the mature mRNA counts of the Fig. 4 Source Data of Kilic et al. (2023), from the
experiments of Wang et al. (2019): 16 fixation times from 15 to 1200 s after induction, N = 3833
cells (`kilic_data.load_kilic_fig4`, t = 0 dropped).

* A minibatch is n indices drawn with replacement from the N cells.
* For each index the simulator runs one independent cell (promoter OFF, no mRNA at t = 0,
  degradation rate fixed at the Wang et al. slow-growth value) up to that cell's own time.
* The summary of a cell is its row of cumulative count indicators within its time block
  (`paired_summary.cumulative_rows`, 80 entries), and `model` returns the simulated rows together
  with the observed rows of the same cells, so the distance compares the batch means of the two.
* The Mahalanobis weight is estimated once from all observed rows (`paired_summary.build_W_inv_rows`).
* The particles are the base-10 logarithms of [K_12, K_21, B_1, B_2], with independent
  N(-1, 0.5^2) priors; everything saved is converted back to rates (1/s).
* A proposal in which any cell runs out of SSA steps is rejected.

A fit stops at the first generation after generation 5 whose acceptance rate is below 0.01 or whose
SNR is below 1, or when its SMC time exceeds `time_limit`.

Usage (from the repository root):
    python kilic/application_kilic.py fvc 64 100 --cores 8          # adaptive, n_0 = 64, seed 100
    python kilic/application_kilic.py constant 256 100 --cores 8    # fixed n = 256

Each fit writes `ecoli_{scheme}_n{size:04d}_s{seed}.pkl` holding `(tracked, posterior, weights,
meta)`: `tracked` maps each generation to its record (particles and weights before that
generation's resampling, cumulative cost), and the file is rewritten after every generation.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import smc_abc_schemes as schemes           # noqa: E402
import paired_summary as ps                 # noqa: E402
import kilic_data                           # noqa: E402


# =========================
# Settings
# =========================
HIST_EDGES = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)   # indicators 1{count <= 0, 1, 3, 7, 15}
ROW_DTYPE = np.int8
W_SHRINK = 0.1
MAX_STEPS = 200000                  # SSA steps per cell
ALPHA = 0.5
LAMBDA = 1.0                        # adaptive scheme: no smoothing of v_total
GAMMA = kilic_data.WANG2019_SLOW[-1]
PARAM_NAMES = ["K_12", "K_21", "B_1", "B_2"]
PRIOR_MU = np.array([-1.0, -1.0, -1.0, -1.0])
PRIOR_SIGMA = np.diag([0.5] * 4) ** 2
STOP_MIN_GENERATION = 5
STOP_ACCEPTANCE = 0.01
STOP_SNR = 1.0
TIME_LIMIT = 28800.0                # seconds of SMC time
DEFAULT_OUT = os.path.join(ROOT, "paper", "results", "runs", "ecoli")


def load_problem():
    """The observed cells, their summary rows and the Mahalanobis weight."""
    obs_t, counts, time_idx, obs_counts = ps.load_pool()
    n_times = int(obs_t.size)
    obs_rows = ps.cumulative_rows(time_idx, obs_counts, HIST_EDGES, n_times, dtype=ROW_DTYPE)
    W_inv, W_diag = ps.build_W_inv_rows(obs_rows, shrink=W_SHRINK)
    return dict(obs_t=obs_t, n_cells=np.array([c.size for c in counts]), time_idx=time_idx,
                obs_counts=obs_counts, n_times=n_times, obs_rows=obs_rows,
                W_inv=W_inv, W_diag=W_diag)


def make_model(obs_t, time_idx, obs_rows, n_times):
    """
    `model(params, batch_idx, seed)` -> (simulated rows, observed rows) for the cells in the batch.

    `params` are log10 rates. If any cell ran out of SSA steps the simulated rows are NaN, which
    the distance turns into an infinite distance.
    """
    n_stats = obs_rows.shape[1]

    def model(params, batch_idx, seed=None):
        idx = np.asarray(batch_idx, dtype=np.int64).ravel()
        tk = time_idx[idx]
        rates = 10.0 ** np.asarray(params, dtype=float)
        full = np.hstack((rates, GAMMA))
        sim_seed = None if seed is None else int(seed) % (2 ** 31 - 1)
        counts, n_trunc = ps.simulate_paired(full, tk, obs_t, MAX_STEPS, sim_seed)
        ps.record_truncation(idx.size, n_trunc, params=full[:4], max_steps=MAX_STEPS)
        if n_trunc:
            return np.full((idx.size, n_stats), np.nan), obs_rows[idx]
        rows = ps.cumulative_rows(tk, counts.astype(np.int64), HIST_EDGES, n_times, dtype=ROW_DTYPE)
        return rows, obs_rows[idx]

    return model


def stats_func(x, y):
    """Identity: `model` already returns the per-cell summary rows."""
    return x, y


def make_distance(W_inv):
    """Mahalanobis distance between the batch means of the simulated and observed rows; a
    non-finite batch mean (a truncated proposal) gives an infinite distance."""
    def paired_mahalanobis(x, y, thr, _W=W_inv):
        xm = np.mean(x, axis=0) if np.ndim(x) > 1 else np.asarray(x, dtype=float)
        ym = np.mean(y, axis=0) if np.ndim(y) > 1 else np.asarray(y, dtype=float)
        diff = xm - ym
        if not np.all(np.isfinite(diff)):
            return np.inf, False
        d2 = float(diff @ _W @ diff)
        d = np.sqrt(d2) if d2 > 0.0 else 0.0
        return d, (d < thr)
    return paired_mahalanobis


def make_prior(seed):
    """Independent N(-1, 0.5^2) on the log10 rates: [sampler, domain, density]."""
    mu, Sigma = PRIOR_MU, PRIOR_SIGMA
    L_chol = np.linalg.cholesky(Sigma)
    base_rng = np.random.default_rng(seed)
    domain = np.repeat([[-np.inf, np.inf]], 4, axis=0)
    log_norm = (-0.5 * len(mu) * np.log(2 * np.pi)
                - 0.5 * 2.0 * np.sum(np.log(np.diag(L_chol))))

    def prior_func(rng=None):
        r = rng if rng is not None else base_rng
        return r.multivariate_normal(mu, Sigma)

    def density(x, mu=mu):
        x = np.atleast_2d(x).astype(float)
        y = np.linalg.solve(L_chol, (x - mu).T)
        return np.exp(log_norm - 0.5 * np.sum(y ** 2, axis=0))

    return [prior_func, domain, density]


def generation_record(est, n_used, cells):
    """One generation's record. `posterior` and `weights` are taken before this generation's
    resampling, with the particles converted to rates."""
    return dict(
        generation=est.generation,
        batch_size_used=int(n_used),
        batch_size_next=int(est.batch_size),
        total_time=float(est.total_time),
        step_time=float(est.step_time),
        total_sims=int(est.total_sims),
        sim_cells=int(cells),
        acceptance=float(est.acceptance_rate),
        ESS=float(est.ESS),
        alpha_threshold=float(est.alpha_threshold),                   # tolerance for the next generation
        current_alpha_threshold=float(est.current_alpha_threshold),   # tolerance used in this one
        v_total_est=float(est.v_total_est),
        v_total_ema=float(getattr(est, "v_total", np.nan)),
        snr=float(est.snr),
        c=float(est.c) if getattr(est, "c", None) is not None else np.nan,
        c_est=float(getattr(est, "c_est_", np.nan)),
        log_hdpr_product=float(est.log_hdpr_product),
        posterior=10.0 ** np.asarray(est.posterior, dtype=float),
        weights=np.asarray(est.weights).copy(),
    )


def stop_reason(est, time_limit, max_generations=None):
    if est.total_time > time_limit:
        return "time_limit"
    if est.generation > STOP_MIN_GENERATION:
        if est.acceptance_rate < STOP_ACCEPTANCE:
            return "min_acceptance"
        if est.snr < STOP_SNR:
            return "snr_floor"
    if max_generations is not None and est.generation >= max_generations:
        return "max_generations"
    return None


def run_fit(scheme, size, seed, out_dir, particles=1000, cores=8, time_limit=TIME_LIMIT,
            max_generations=None):
    """
    Run one fit and return the path of its record.

    scheme : "constant" (fixed batch `size`) or "fvc" (adaptive, initial batch `size`, automatic c)
    """
    if scheme not in ("constant", "fvc"):
        raise ValueError(f"unknown scheme {scheme!r}")
    size, seed = int(size), int(seed)
    problem = load_problem()
    obs_rows = problem["obs_rows"]
    n_pool = int(obs_rows.shape[0])
    init_params = dict(
        data=obs_rows,
        model=make_model(problem["obs_t"], problem["time_idx"], obs_rows, problem["n_times"]),
        stats_func=stats_func,
        prior=make_prior(seed),
        dist_func="mahalanobis",
        mweight_mat=problem["W_inv"],
        num_particles=particles,
        alpha=ALPHA,
        cores=cores,
        sample_with_replacement=True,
        seed=seed,
        parallel_args=dict(max_nbytes="100M", timeout=99999),
    )
    print(f"E. coli {scheme} n={size} seed={seed}: {problem['n_times']} times, N = {n_pool} cells, "
          f"{obs_rows.shape[1]} summaries, cond(W) = {problem['W_diag']['W_cond']:.4g}", flush=True)
    if scheme == "fvc":
        sp = [size, None, LAMBDA]
        est = schemes.fvc_init(init_params, p=sp)
        loop = schemes.fvc_loop
    else:
        sp = [size]
        est = schemes.constant_init(init_params, p=sp)
        loop = schemes.constant_loop
    # Same Mahalanobis distance, but a proposal whose simulation hit the step cap gets +inf
    est.dist_func = make_distance(problem["W_inv"])
    est.estimate_v_total = ps.make_v_total_estimator(obs_rows, report=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ecoli_{scheme}_n{size:04d}_s{seed}.pkl")
    tracked = {}
    cells = n_pool              # the estimator's warm-up simulates every cell once
    prev_sims = 0
    reason = None
    while reason is None:
        n_used = int(est.batch_size)
        loop(est, ess_resample=False)
        cells += (int(est.total_sims) - prev_sims) * n_used
        prev_sims = int(est.total_sims)
        rec = generation_record(est, n_used, cells)
        tracked[est.generation] = rec
        print(f"[{scheme}] gen {est.generation}: n_used={n_used} n_next={rec['batch_size_next']} "
              f"eps={rec['alpha_threshold']:.4f} v={rec['v_total_est']:.4g} snr={rec['snr']:.3f} "
              f"acc={rec['acceptance']:.4f} ESS={rec['ESS']:.1f} t={rec['total_time']:.1f}s "
              f"cells={cells}", flush=True)
        reason = stop_reason(est, time_limit, max_generations)
        scheme_params = [size, float(est.c), LAMBDA] if scheme == "fvc" else sp
        meta = dict(
            scheme=scheme, scheme_params=scheme_params, seed=seed, num_particles=particles,
            alpha=ALPHA, cores=cores, time_limit=time_limit, stop_reason=reason or "running",
            generations=est.generation, obs_t=problem["obs_t"], n_cells=problem["n_cells"],
            N_pool=n_pool, time_idx=problem["time_idx"], obs_counts=problem["obs_counts"],
            hist_edges=HIST_EDGES, row_dtype=np.dtype(ROW_DTYPE).name, w_shrink=W_SHRINK,
            w_diagnostics=problem["W_diag"], max_steps=MAX_STEPS, gamma_fixed=GAMMA,
            param_names=list(PARAM_NAMES), param_space="log10", posterior_units="rates",
            prior=dict(kind="log10-normal", mu=PRIOR_MU, Sigma=PRIOR_SIGMA),
            sample_with_replacement=True, warmup_cells=n_pool, wang2019=kilic_data.WANG2019_SLOW)
        with open(out_path + ".tmp", "wb") as f:
            pickle.dump((tracked, 10.0 ** np.asarray(est.posterior, dtype=float),
                         np.asarray(est.weights), meta), f)
        os.replace(out_path + ".tmp", out_path)
        if reason is None:
            est.ESS_resample(proportion=0.5)
    print(f"[{scheme}] stopped after generation {est.generation} ({reason}): "
          f"total_time={est.total_time:.0f}s cells={cells}; saved {out_path}", flush=True)
    ps.truncation_report(where=f"after {scheme} (parent process)", quiet_if_zero=False)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="One SMC ABC fit to the E. coli lacZ data.")
    p.add_argument("scheme", choices=["constant", "fvc"])
    p.add_argument("size", type=int, help="fixed batch size (constant) or initial batch size n_0 (fvc)")
    p.add_argument("seed", type=int)
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    p.add_argument("--max-generations", type=int, default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run_fit(a.scheme, a.size, a.seed, a.out, particles=a.particles, cores=a.cores,
            time_limit=a.time_limit, max_generations=a.max_generations)


if __name__ == "__main__":
    main()
