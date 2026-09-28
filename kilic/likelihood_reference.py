"""
Likelihood-based reference posterior for the two-state telegraph model on the E. coli lacZ data.

The telegraph model is small enough that its likelihood can be evaluated exactly: the chemical
master equation on a truncated state space, one matrix exponential per time step. Sampling that
posterior under the same model and prior as the ABC fits gives the reference the E. coli figure
scores the ABC posteriors against.

The likelihood
--------------
State = (promoter state s in {OFF, ON}) x (mRNA count m in {0, ..., M}). Reactions:

    OFF -> ON        rate K_12
    ON  -> OFF       rate K_21
    m -> m + 1       rate B_1 (OFF) or B_2 (ON)
    m -> m - 1       rate gamma * m          (gamma fixed at 0.32 per minute)

The CME is dP/dt = A P; production out of m = M is dropped, so probability is conserved, and M is
chosen by a mass check (M = 140 for these data). Cells at different times are independent
snapshots, so the likelihood is a product over time points of the marginal count distribution at
that time, over the cells imaged then (Kilic et al. 2023, Eq. 1). Cells start OFF with no mRNA.

Prior: independent Normal(-1, 0.5^2) on log10 of each rate, the prior of Kilic et al. (2023) and of
the ABC fits. Sampling is in log10 space, so the prior enters without a Jacobian.

Usage (from the repository root)
--------------------------------
    python kilic/likelihood_reference.py --run         # MCMC -> paper/results/likelihood_reference.pkl
    python kilic/likelihood_reference.py --validate    # likelihood checks on the Wang et al. counts

The output pickle is a dict with
    samples_log10   (n_chains, n_kept, 4)  draws of log10([K_12, K_21, B_1, B_2])
    samples         (n_chains * n_kept, 4) the same, flattened, in rates (1/s)
    log_post / log_lik / log_prior         (n_chains, n_kept)
    map_log10, map_log_post, rhat, rhat_log_post, ess, accept_rate, obs_t, n_cells, M, init,
    gamma, prior, settings
"""
import argparse
import os
import pickle as pkl
import sys
import time

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import kilic_data                                 # noqa: E402

PARAM_NAMES = ["K_12", "K_21", "B_1", "B_2"]
GAMMA_DEFAULT = 0.00533333333333333               # 0.32 per minute, in 1/s
PRIOR_MU = -1.0                                   # log10-space prior mean, all four rates
PRIOR_SD = 0.5
LOG_FLOOR = 1e-300
DEFAULT_OUT = os.path.join(HERE, "..", "paper", "results", "likelihood_reference.pkl")

# Parameter sets on the natural scale, [K_12, K_21, B_1, B_2]: the joint MAP sample and the
# posterior means of the slow-growth MCMC chain of Kilic et al. (2023), and the slow-growth
# estimate of Wang et al. (2019). They start the MAP search and bracket the truncation check.
KILIC_MAP = np.array([0.0032979, 0.0098066, 2.44e-7, 0.095729])
KILIC_CHAIN_MEAN = np.array([0.003233, 0.009744, 0.000177, 0.09987])
WANG_SLOW = np.array([0.00533, 0.03, 0.0, 0.166])


# =========================================================================================
# CME
# =========================================================================================
def build_generator(K12, K21, B1, B2, gamma, M):
    """
    Dense CME generator A for the telegraph model on states (s, m), s in {0 = OFF, 1 = ON},
    m in {0..M}, flattened as s*(M+1) + m. dP/dt = A P, so A[j, i] is the rate of i -> j.
    """
    n = M + 1
    N = 2 * n
    A = np.zeros((N, N))
    betas = (B1, B2)
    switch = (K12, K21)                       # OFF -> ON, ON -> OFF
    for s in range(2):
        b = betas[s]
        ks = switch[s]
        other = 1 - s
        for m in range(n):
            i = s * n + m
            # promoter switch
            A[other * n + m, i] += ks
            A[i, i] -= ks
            # transcription (none out of m = M, which conserves probability)
            if m < M:
                A[s * n + m + 1, i] += b
                A[i, i] -= b
            # degradation
            if m > 0:
                A[s * n + m - 1, i] += gamma * m
                A[i, i] -= gamma * m
    return A


def initial_distribution(M, init="delta0"):
    """P(s, m) at t = 0. The promoter starts OFF. "delta0": no mRNA. "exp1": P(m) proportional to
    exp(-m), the initial distribution of Kilic et al.'s code, used only in `validate`."""
    n = M + 1
    p = np.zeros(2 * n)
    if init == "delta0":
        p[0] = 1.0
    elif init == "exp1":
        w = np.exp(-np.arange(n, dtype=float))
        p[:n] = w / w.sum()
    else:
        raise ValueError(f"unknown init {init!r} (expected 'delta0' or 'exp1')")
    return p


def step_plan(times, max_steps=4000):
    """
    The times are multiples of a common base step (0, 15, ..., 1200 -> 15 s), so one matrix
    exponential of A * base can be applied repeatedly. Returns `(base, steps)` with `steps[k]` the
    number of base steps from time k-1 to time k, or `(None, gaps)` if there is no small common
    base, in which case `marginals` computes one exponential per gap.
    """
    t = np.asarray(times, dtype=float)
    gaps = np.diff(np.concatenate([[0.0], t]))
    if np.any(gaps < 0):
        raise ValueError("observation times must be non-decreasing")
    nz = gaps[gaps > 0]
    if nz.size == 0:
        return None, gaps
    scaled = nz * 1000.0
    if np.all(np.abs(scaled - np.round(scaled)) < 1e-6):
        g = np.gcd.reduce(np.round(scaled).astype(np.int64)) / 1000.0
        if g > 0 and t[-1] / g <= max_steps:
            return float(g), np.round(gaps / g).astype(int)
    return None, gaps


def marginals(theta, times, M, gamma=GAMMA_DEFAULT, init="delta0", plan=None):
    """
    P(m at time t_k) summed over the promoter state, shape (len(times), M + 1).
    `theta` is [K_12, K_21, B_1, B_2] in 1/s.
    """
    K12, K21, B1, B2 = (float(v) for v in theta)
    A = build_generator(K12, K21, B1, B2, gamma, M)
    n = M + 1
    p = initial_distribution(M, init)
    out = np.empty((len(times), n))
    base, steps = plan if plan is not None else step_plan(times)
    if base is not None:
        E = expm(A * base)
        for k, s in enumerate(steps):
            for _ in range(int(s)):
                p = E @ p
            out[k] = p[:n] + p[n:]
    else:
        cache = {}
        for k, dt in enumerate(steps):
            if dt > 0:
                key = round(float(dt), 9)
                if key not in cache:
                    cache[key] = expm(A * dt)
                p = cache[key] @ p
            out[k] = p[:n] + p[n:]
    return out


def counts_to_hist(counts, M):
    """(len(counts), M + 1) table of how many observed cells carry each count at each time."""
    H = np.zeros((len(counts), M + 1))
    for k, c in enumerate(counts):
        c = np.asarray(c, dtype=int)
        if c.size and c.max() > M:
            raise ValueError(f"observed count {c.max()} exceeds truncation M = {M}")
        H[k] = np.bincount(c, minlength=M + 1)[: M + 1]
    return H


def log_likelihood(theta, times, hist, M, gamma=GAMMA_DEFAULT, init="delta0", plan=None,
                   per_time=False):
    """Sum over time points and cells of log P(count | theta); `hist` comes from `counts_to_hist`."""
    P = marginals(theta, times, M, gamma=gamma, init=init, plan=plan)
    logP = np.log(np.clip(P, LOG_FLOOR, None))
    per = np.einsum("km,km->k", hist, logP)
    return per if per_time else float(per.sum())


def choose_M(theta, times, gamma=GAMMA_DEFAULT, init="delta0", max_obs=0,
             tol=1e-10, candidates=(41, 60, 80, 100, 140, 200, 300)):
    """
    Smallest candidate truncation whose tail mass (the top five states, at every time point) stays
    below `tol` and that exceeds the largest observed count by at least five. Returns (M, tail).
    """
    for M in candidates:
        if M < max_obs + 5:
            continue
        P = marginals(theta, times, M, gamma=gamma, init=init)
        tail = float(P[:, -5:].sum(axis=1).max())
        if tail < tol:
            return int(M), tail
    M = int(candidates[-1])
    P = marginals(theta, times, M, gamma=gamma, init=init)
    return M, float(P[:, -5:].sum(axis=1).max())


# =========================================================================================
# Posterior
# =========================================================================================
def log_prior(log10_theta):
    """Independent Normal(-1, 0.5^2) on log10 of each rate."""
    z = (np.asarray(log10_theta, dtype=float) - PRIOR_MU) / PRIOR_SD
    return float(-0.5 * np.sum(z ** 2) - z.size * (np.log(PRIOR_SD) + 0.5 * np.log(2 * np.pi)))


def make_log_post(times, hist, M, gamma=GAMMA_DEFAULT, init="delta0"):
    """Closure returning (log posterior, log likelihood, log prior) at a log10 parameter vector."""
    plan = step_plan(times)

    def f(log10_theta):
        lp = log_prior(log10_theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, lp
        if np.any(log10_theta > 3.0) or np.any(log10_theta < -12.0):
            return -np.inf, -np.inf, lp
        theta = 10.0 ** np.asarray(log10_theta, dtype=float)
        try:
            ll = log_likelihood(theta, times, hist, M, gamma=gamma, init=init, plan=plan)
        except (ValueError, FloatingPointError):
            return -np.inf, -np.inf, lp
        if not np.isfinite(ll):
            return -np.inf, ll, lp
        return lp + ll, ll, lp

    return f


def adaptive_metropolis(log_post, start, n_iter, burn_in, seed, adapt_start=500,
                        target_sd=None, eps=1e-8, thin=1, progress_every=0):
    """
    Adaptive Metropolis (Haario et al. 2001) in log10 space: after `adapt_start` iterations the
    proposal covariance is (2.38^2 / d) (running sample covariance + eps I), refreshed every 100
    iterations. Returns the post-burn-in draws and their log densities.
    """
    rng = np.random.default_rng(seed)
    d = len(start)
    sd = (2.38 ** 2) / d if target_sd is None else target_sd
    x = np.asarray(start, dtype=float).copy()
    lpost, ll, lpri = log_post(x)
    if not np.isfinite(lpost):
        raise ValueError(f"starting point has zero posterior density: {x}")

    C0 = np.diag(np.full(d, 0.05 ** 2))            # small isotropic proposal before adaptation
    L = np.linalg.cholesky(sd * C0)
    mean = x.copy()
    cov = np.zeros((d, d))
    n_acc = 0

    keep = list(range(burn_in, n_iter, thin))
    out_x = np.empty((len(keep), d))
    out_lp = np.empty(len(keep))
    out_ll = np.empty(len(keep))
    out_pr = np.empty(len(keep))
    ki = 0
    t0 = time.time()

    for it in range(n_iter):
        prop = x + L @ rng.standard_normal(d)
        plp, pll, ppr = log_post(prop)
        if np.log(rng.random()) < plp - lpost:
            x, lpost, ll, lpri = prop, plp, pll, ppr
            n_acc += 1

        # running mean and covariance (Welford), then refresh the proposal factor
        delta = x - mean
        mean += delta / (it + 1)
        cov += np.outer(delta, x - mean)
        if it >= adapt_start and it % 100 == 0:
            S = cov / it
            try:
                L = np.linalg.cholesky(sd * (S + eps * np.eye(d)))
            except np.linalg.LinAlgError:
                pass

        if ki < len(keep) and it == keep[ki]:
            out_x[ki] = x
            out_lp[ki] = lpost
            out_ll[ki] = ll
            out_pr[ki] = lpri
            ki += 1
        if progress_every and (it + 1) % progress_every == 0:
            print(f"    seed {seed}: {it + 1}/{n_iter} iters, acc={n_acc / (it + 1):.3f}, "
                  f"logpost={lpost:.2f}, {time.time() - t0:.0f}s", flush=True)

    return dict(x=out_x, log_post=out_lp, log_lik=out_ll, log_prior=out_pr,
                accept_rate=n_acc / n_iter, seconds=time.time() - t0)


def split_rhat(chains):
    """Split-R-hat (Gelman et al., BDA3) of an (n_chains, n_draws) array."""
    x = np.asarray(chains, dtype=float)
    m, n = x.shape
    h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)     # (2m, h)
    if h < 2:
        return np.nan
    means = s.mean(axis=1)
    variances = s.var(axis=1, ddof=1)
    W = variances.mean()
    B = h * means.var(ddof=1)
    if W <= 0:
        return np.nan
    var_hat = (h - 1) / h * W + B / h
    return float(np.sqrt(var_hat / W))


def ess(chains):
    """Bulk effective sample size of an (n_chains, n_draws) array (Stan's autocovariance rule)."""
    x = np.asarray(chains, dtype=float)
    m, n = x.shape
    if n < 4:
        return np.nan
    means = x.mean(axis=1, keepdims=True)
    var_within = x.var(axis=1, ddof=1).mean()
    if var_within <= 0:
        return np.nan
    B = n * x.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * var_within + (B / n if m > 1 else 0.0)

    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    acov = np.zeros(n)
    for c in range(m):
        y = x[c] - means[c]
        f = np.fft.rfft(y, nfft)
        a = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
        acov += a / n
    acov /= m
    # combined autocorrelation estimate: rho_t = 1 - (W - acov_t) / var_plus
    rho = 1.0 - (var_within - acov) / var_plus
    # Geyer's initial positive sequence on the paired sums
    t = 1
    tau = -1.0
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        tau += 2 * p
        t += 2
    tau = max(tau, 1.0)
    return float(m * n / tau)


def find_map(log_post, starts):
    """Nelder-Mead from several starts; returns the best (log10 parameter vector, log posterior)."""
    from scipy.optimize import minimize
    best = (None, -np.inf)
    for s in starts:
        r = minimize(lambda z: -log_post(z)[0], np.asarray(s, dtype=float),
                     method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        v = -r.fun
        if np.isfinite(v) and v > best[1]:
            best = (r.x, float(v))
    return best


# =========================================================================================
# Validation on the Wang et al. counts
# =========================================================================================
# Log-likelihoods of Wang et al.'s slow-growth TOTAL counts (datasets/wang2019/data-fig2g.mat, all
# 17 times, gamma = 0.32 per minute, M = 41) under both initial distributions, at Kilic et al.'s
# MAP sample and at Wang et al.'s estimate: the first four times, and all times. With the "exp1"
# initial distribution of Kilic et al.'s code, the total at their MAP sample agrees with the
# log-likelihood their chain records for that sample (about -6130).
VALIDATION_TARGETS = {
    ("exp1", "kilic_map", "first4"): -567.8,
    ("exp1", "wang", "first4"): -616.5,
    ("delta0", "kilic_map", "first4"): -139.4,
    ("delta0", "wang", "first4"): -181.7,
    ("exp1", "kilic_map", "total"): -6127.6,
    ("delta0", "kilic_map", "total"): -5785.2,
}


def validate(gamma=GAMMA_DEFAULT, M=41):
    """Print the log-likelihoods of `VALIDATION_TARGETS` next to their expected values."""
    ts, counts = kilic_data.load_wang("slow", n_drop_last=0, which="data_tot")
    counts = [np.clip(c, 0, M) for c in counts]
    hist = counts_to_hist(counts, M)
    print(f"Wang et al. slow-growth total counts, 17 times, gamma={gamma}, M={M}")
    print(f"{'init':>8s} {'params':>10s} {'quantity':>8s} {'value':>12s} {'expected':>12s} {'diff':>9s}")
    rows = []
    for init in ("exp1", "delta0"):
        for name, th in (("kilic_map", KILIC_MAP), ("wang", WANG_SLOW)):
            per = log_likelihood(th, ts, hist, M, gamma=gamma, init=init, per_time=True)
            for quantity, val in (("first4", float(per[:4].sum())), ("total", float(per.sum()))):
                target = VALIDATION_TARGETS.get((init, name, quantity))
                if target is None:
                    continue
                print(f"{init:>8s} {name:>10s} {quantity:>8s} {val:12.1f} {target:12.1f} "
                      f"{val - target:9.1f}")
                rows.append((init, name, quantity, val, target))
    return rows


# =========================================================================================
# Driver
# =========================================================================================
def load_data():
    """(times, counts, n_cells) of the Fig. 4 mature counts, all 17 time points."""
    ts, counts = kilic_data.load_kilic_fig4()
    n_cells = np.array([c.size for c in counts])
    return np.asarray(ts, dtype=float), counts, n_cells


def run_mcmc(args):
    from joblib import Parallel, delayed

    ts, counts, n_cells = load_data()
    max_obs = int(max(c.max() if c.size else 0 for c in counts))
    print(f"Data: {ts.size} time points {ts.astype(int)}")
    print(f"Cells per time point: {n_cells} (total {n_cells.sum()}), largest count {max_obs}")

    # Truncation: smallest M whose top-five-state mass stays below 1e-10 at every time, at
    # parameter sets bracketing the posterior (Kilic et al.'s chain mean, Wang et al.'s estimate,
    # and the chain mean with the ON-state production rate tripled).
    probe = [KILIC_CHAIN_MEAN, WANG_SLOW, KILIC_CHAIN_MEAN * np.array([1., 1., 1., 3.])]
    Ms, tails = [], []
    for th in probe:
        Mi, tail = choose_M(th, ts, gamma=args.gamma, max_obs=max_obs)
        Ms.append(Mi)
        tails.append(tail)
    M = int(max(Ms))
    print(f"Truncation: M = {M} (mass check per probe: "
          f"{[(int(a), '%.1e' % b) for a, b in zip(Ms, tails)]})")
    hist = counts_to_hist([np.asarray(c) for c in counts], M)
    log_post = make_log_post(ts, hist, M, gamma=args.gamma)

    t0 = time.time()
    starts = [np.log10(np.clip(KILIC_CHAIN_MEAN, 1e-6, None)),
              np.log10(np.clip(WANG_SLOW, 1e-6, None)),
              np.array([-1.0, -1.0, -1.0, -1.0])]
    map_x, map_lp = find_map(log_post, starts)
    print(f"MAP (log10): {np.array2string(map_x, precision=4)}  -> "
          f"{np.array2string(10 ** map_x, precision=6)}, log posterior {map_lp:.2f} "
          f"({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    chain_starts = [map_x + rng.normal(0, 0.15, size=4) for _ in range(args.chains)]
    chain_starts[0] = map_x.copy()

    print(f"Running {args.chains} adaptive-Metropolis chains, {args.iters} iterations "
          f"({args.burn} burn-in), {args.cores} cores")
    res = Parallel(n_jobs=args.cores)(
        delayed(adaptive_metropolis)(log_post, chain_starts[i], args.iters, args.burn,
                                     seed=args.seed + 1000 * i, thin=args.thin,
                                     progress_every=args.iters // 4)
        for i in range(args.chains))

    X = np.stack([r["x"] for r in res])                    # (chains, kept, 4)
    LP = np.stack([r["log_post"] for r in res])
    LL = np.stack([r["log_lik"] for r in res])
    PR = np.stack([r["log_prior"] for r in res])
    acc = np.array([r["accept_rate"] for r in res])
    secs = time.time() - t0

    rhat = np.array([split_rhat(X[:, :, j]) for j in range(4)])
    ess_ = np.array([ess(X[:, :, j]) for j in range(4)])
    rhat_lp = split_rhat(LP)

    flat = X.reshape(-1, 4)
    lin = 10.0 ** flat

    # Truncation check at the posterior's 99.9th-percentile corner
    corner = np.quantile(lin, 0.999, axis=0)
    P_corner = marginals(corner, ts, M, gamma=args.gamma)
    tail_corner = float(P_corner[:, -5:].sum(axis=1).max())
    print(f"Truncation check at the posterior 99.9% corner "
          f"{np.array2string(corner, precision=5)}: top-5-state mass {tail_corner:.2e}")
    print(f"\nDone in {secs:.0f}s. Acceptance {np.round(acc, 3)}, "
          f"R-hat(log posterior) {rhat_lp:.4f}")
    print(f"{'param':>7s} {'R-hat':>8s} {'ESS':>9s} {'median':>12s} "
          f"{'2.5%':>12s} {'97.5%':>12s} {'log10 med':>10s}")
    for j, name in enumerate(PARAM_NAMES):
        q = np.quantile(lin[:, j], [0.025, 0.5, 0.975])
        print(f"{name:>7s} {rhat[j]:8.4f} {ess_[j]:9.0f} {q[1]:12.6g} "
              f"{q[0]:12.6g} {q[2]:12.6g} {np.median(flat[:, j]):10.4f}")

    out = dict(
        samples_log10=X, samples=lin, log_post=LP, log_lik=LL, log_prior=PR,
        rhat=rhat, rhat_log_post=rhat_lp, ess=ess_, accept_rate=acc,
        param_names=PARAM_NAMES, obs_t=ts, n_cells=n_cells, M=M, init="delta0",
        gamma=args.gamma, max_obs=max_obs, map_log10=map_x, map_log_post=map_lp,
        truncation_tail_at_posterior_corner=tail_corner,
        prior=dict(kind="log10-normal", mu=PRIOR_MU, sd=PRIOR_SD),
        settings=dict(chains=args.chains, iters=args.iters, burn=args.burn, thin=args.thin,
                      seed=args.seed, cores=args.cores, wall_seconds=secs),
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    with open(out_path, "wb") as f:
        pkl.dump(out, f)
    print(f"Saved {out_path}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the MCMC and save the reference")
    p.add_argument("--validate", action="store_true",
                   help="log-likelihood checks on the Wang et al. counts")
    p.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--iters", type=int, default=40_000)
    p.add_argument("--burn", type=int, default=15_000)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.validate:
        validate(gamma=a.gamma)
    if a.run:
        run_mcmc(a)
    if not (a.validate or a.run):
        print("nothing to do: pass --run or --validate")
