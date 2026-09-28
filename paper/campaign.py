"""
The SMC ABC fits behind the figures of the paper.

Four campaigns:

    main        TD and LV: fixed batch sizes, and the adaptive rule with sweeps of its smoothing
                weight lambda, initial batch size n0 and noise ratio c. One synthetic dataset per
                model, three SMC seeds per arm.
    robustness  TD and LV: the generating-parameter and heterogeneity grids of
                experiment_physical_parameters.py and experiment_heterogeneities.py, plus the LV
                baseline condition. Ten seeds, each with its own dataset.
    nsweep      TD datasets of N = 512, 2048, 8192 and 32768 cells: full batch, fixed n = 512 and
                the adaptive rule. Three seeds, each with its own dataset.
    ecoli       The E. coli application (kilic/application_kilic.py): four fixed batch sizes and
                three initial batch sizes of the adaptive rule. Five seeds.

Usage (from anywhere):

    python paper/campaign.py list main              # index, tag and settings of every fit
    python paper/campaign.py run main 17            # one fit
    python paper/campaign.py run main 0-35          # a range of fits, one after another
    python paper/campaign.py run main all --cores 8 --out DIR

Fits are independent, so a campaign can be split over machines by index. The synthetic datasets
are generated from their seeds at the start of each fit. A synthetic fit writes
`<out>/<campaign>/<model>/<tag>.pkl`, a dict with the job, one row of diagnostics per generation
and the stop reason, and each generation's particles and weights to
`<out>/<campaign>/<model>/<tag>/particles_XXX.npz`. E. coli fits write `<out>/ecoli/*.pkl`.
`collect_results.py` turns the synthetic records into the inputs of the figure scripts.

The wall times in the paper come from running each fit alone with 8 worker processes
(`--cores 8`) and one BLAS/numba thread per process. Results do not depend on the number of
workers: every proposal draws from its own seed.
"""
import os

# One thread per process, so that the worker processes are the only parallelism
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _path in (REPO, os.path.join(REPO, 'kilic')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import experiments as E
import lotka_volterra as lv
import smc_abc_schemes as schemes
import smc_abc_utils as utils
import transcriptional_dynamics as td

RUNS = os.path.join(HERE, 'results', 'runs')

#### Settings shared by all synthetic fits
T, DT = 5.0, 0.01                                  # simulated time and time step
ALPHA = 0.5                                        # quantile a: keep the best half of proposals
PARTICLES = dict(td=500, lv=1000)
PRIOR = dict(td=[[1, 100], [1, 100], [0, 1]],      # k_+, r_burst, D
             lv=[[0, 50], [0, 0.2], [0, 50]])      # alpha, beta, gamma
DEFAULTS = dict(                                   # generating values and dataset sizes
    td=dict(physical_params=[15., 10., .1], kminus=5., heterogeneities=[[2., 2.], [10., .1]], sample_size=512),
    lv=dict(physical_params=[5., .02, 5.], heterogeneities=[1000], sample_size=128, d=1.))
MAX_GENERATIONS = 25
WALL_MINUTES = dict(main=360, robustness=360, nsweep=780)
SIM_CHUNK = 2048                                   # TD batches above this are simulated in chunks

#### E. coli
ECOLI_ARMS = [('constant', 64), ('constant', 256), ('constant', 1024), ('constant', 3833),
              ('fvc', 16), ('fvc', 64), ('fvc', 256)]
ECOLI_SEEDS = range(100, 105)
ECOLI_TIME_LIMIT = 8 * 3600                        # seconds


def stop_reason(generation, acceptance, snr, elapsed, wall_minutes, max_generations=MAX_GENERATIONS):
    """The stopping rule: after generation 5, stop once acceptance < 0.01 or SNR < 1; wall and generation caps as safeguards."""
    if elapsed >= wall_minutes * 60:
        return 'wall_cap'
    if generation > 5 and (acceptance < .01 or snr < 1):
        return 'paper_rule'
    if generation >= max_generations:
        return 'generation_cap'
    return None


#### The fits
def arm_name(scheme, params):
    if scheme == 'constant':
        return 'constant_n%d' % params[0]
    n0, c, lam = params
    return 'fvc_n%d_c%s_lam%g' % (n0, 'auto' if c is None else '%g' % c, lam)


def synthetic_fit(campaign, model, family, sweep, scheme, params, seed, rep, data_seed, settings, tag, arm=None):
    settings = dict(DEFAULTS[model], **settings, scheme=scheme, scheme_params=params)
    return dict(campaign=campaign, model=model, family=family, sweep=sweep, scheme=scheme, params=params,
                arm=arm or arm_name(scheme, params), seed=seed, rep=rep, data_seed=data_seed, settings=settings,
                tag=tag)


def main_fits():
    """18 arms per model on the seed-0 dataset, three SMC seeds each."""
    fixed = dict(td=[32, 64, 128, 256, 512], lv=[8, 16, 32, 64, 128])
    seeds = dict(td=[1556992120, 3201167925, 277065691], lv=[20260910, 20260911, 20260912])
    fits = []
    for model in ('td', 'lv'):
        arms = [('constant', [n]) for n in fixed[model]]
        arms += [('fvc', [2, None, lam]) for lam in (0., .25, .5, .75, 1.)]
        arms += [('fvc', [n0, None, 1.]) for n0 in (4, 8, 16, 32)]
        arms += [('fvc', [2, c, 1.]) for c in (.1, .2, .3, .4)]
        for scheme, params in arms:
            for rep, seed in enumerate(seeds[model], 1):
                arm = arm_name(scheme, params)
                fits.append(synthetic_fit('main', model, 'main', arm, scheme, params, seed, rep, 0, {},
                                          '%s_s%d' % (arm, rep)))
    return fits


def robustness_conditions():
    """(model, family, sweep, scheme, params, settings) for every condition: the grids of
    experiment_physical_parameters.py and experiment_heterogeneities.py."""
    conditions = []
    arms = dict(td=dict(constant=[[n] for n in (128, 256, 512)], fvc=[[n0, None, 1] for n0 in (2, 8, 32)]),
                lv=dict(constant=[[n] for n in (32, 64, 128)], fvc=[[n0, None, 1] for n0 in (2, 4, 8)]))
    physical = dict(
        td=dict(kplus=[[15 * k, 10, 0.1] for k in (0.2, 5)],
                rburst=[[15, 10 * r, 0.1] for r in (0.2, 5)],
                diffusivity=[[15, 10, 0.1 * d] for d in (0.2, 5)]),
        lv=dict(alpha=[[a * 5, 0.02, 5] for a in (0.2, 5)],
                beta=[[5, b * 0.02, 5] for b in (0.2, 5)],
                gamma=[[5, 0.02, g * 5] for g in (0.2, 5)]))
    for model in ('td', 'lv'):
        for name, values in physical[model].items():
            for scheme in ('constant', 'fvc'):
                for params in arms[model][scheme]:
                    for value in values:
                        conditions.append((model, 'physical', '%s_%s' % (scheme, name), scheme, params,
                                           dict(physical_params=value)))
    # TD: Gamma(shape, scale) nuclear lengths with shape 10f and scale g/10, f and g in {1/3, 1, 3}
    factors = (1 / 3.0, 1.0, 3.0)
    for i, g in enumerate(factors):
        for scheme in ('constant', 'fvc'):
            for params in arms['td'][scheme]:
                for f in factors:
                    conditions.append(('td', 'heterogeneity', '%s_scale%d' % (scheme, i), scheme, params,
                                       dict(heterogeneities=[[2, 2], [f * 10, g / 10]])))
    # LV: initial counts uniform on a range of the given width centred at 500
    lv_arms = [('constant', [[32]]), ('constant', [[64], [128]]),
               ('fvc', [[2, None, 1], [4, None, 1]]), ('fvc', [[8, None, 1], [16, None, 1]])]
    for i, (scheme, params_list) in enumerate(lv_arms):
        for params in params_list:
            for width in (1, 250, 500, 750):
                conditions.append(('lv', 'heterogeneity', '%s_ic_range%d' % (scheme, i % 2), scheme, params,
                                   dict(heterogeneities=[width])))
    # LV baseline condition (TD's lies in its heterogeneity grid)
    for params in ([32], [64], [128]):
        conditions.append(('lv', 'baseline', 'constant_baseline', 'constant', params, {}))
    for n0 in (2, 4, 8, 16):
        conditions.append(('lv', 'baseline', 'fvc_baseline', 'fvc', [n0, None, 1.], {}))
    return conditions


def robustness_fits(seeds=10):
    fits = []
    for i, (model, family, sweep, scheme, params, settings) in enumerate(robustness_conditions()):
        for seed in range(seeds):
            fits.append(synthetic_fit('robustness', model, family, sweep, scheme, params, seed, seed + 1, seed,
                                      settings, '%s_%03d_s%02d' % (family, i, seed),
                                      arm='%s_n%d' % (scheme, params[0])))
    return fits


def nsweep_fits(sizes=(512, 2048, 8192, 32768), seeds=3, fixed=512):
    fits = []
    for n in sizes:
        arms = [('full_batch', 'constant', [n]), ('fvc_n0_2', 'fvc', [2, None, 1.])]
        if fixed < n:
            arms.append(('fixed_n%d' % fixed, 'constant', [fixed]))
        for arm, scheme, params in arms:
            for seed in range(seeds):
                fits.append(synthetic_fit('nsweep', 'td', 'nsweep', '%s_N%d' % (arm, n), scheme, params, seed,
                                          seed + 1, seed, dict(sample_size=n),
                                          'nsweep_N%06d_%s_s%02d' % (n, arm, seed), arm=arm))
    return fits


def ecoli_fits():
    return [dict(campaign='ecoli', model='ecoli', scheme=scheme, size=size, seed=seed,
                 tag='ecoli_%s_n%04d_s%d' % (scheme, size, seed))
            for scheme, size in ECOLI_ARMS for seed in ECOLI_SEEDS]


CAMPAIGNS = dict(main=main_fits, robustness=robustness_fits, nsweep=nsweep_fits, ecoli=ecoli_fits)


#### Synthetic data
def make_dataset(model, seed, settings):
    """The observed dataset of one generating condition and seed."""
    rng = np.random.default_rng(seed)
    theta = [float(v) for v in settings['physical_params']]
    n = int(settings['sample_size'])
    if model == 'td':
        bp, gp = settings['heterogeneities']
        lengths = np.ones(n) if gp is None else rng.gamma(gp[0], gp[1], size=n)
        sites = .5 * lengths if bp is None else rng.beta(bp[0], bp[1], size=n) * lengths
        observations = td.simulate(theta[0], float(settings['kminus']), theta[1], theta[2],
                                   T, DT, sites, lengths, seed=seed)
        return dict(data=[np.asarray(d) for d in observations], sites=sites, lengths=lengths)
    width = int(settings['heterogeneities'][0])
    ic = rng.choice(width, (n, 2)) + 500 - int((width - 1) / 2)
    _, observations = lv.tau_leaping(ic, T, *theta, float(settings['d']), dt=DT, seed=seed)
    return dict(data=observations, ic=ic)


#### Models, as in experiments.transcriptional_dynamics_step and experiments.lotka_volterra_step
class Chunked:
    """Simulate a batch larger than `chunk` cells in chunks, each with a seed drawn from the proposal's seed. Bounds memory at large N."""

    def __init__(self, model, chunk=SIM_CHUNK):
        self.model = model
        self.chunk = int(chunk)

    def __call__(self, params, batch_indices, seed=None):
        idx = np.asarray(batch_indices)
        if idx.shape[0] <= self.chunk:
            return self.model(params, idx, seed=seed)
        seeded = seed is not None and int(seed) >= 0
        rng = np.random.default_rng(int(seed) if seeded else None)
        sims, obs = None, []
        for start in range(0, idx.shape[0], self.chunk):
            sub_seed = int(rng.integers(0, np.iinfo(np.int32).max)) if seeded else seed
            sim_part, obs_part = self.model(params, idx[start:start + self.chunk], seed=sub_seed)
            if sims is None:
                sims = sim_part
            else:
                for cell in sim_part:
                    sims.append(cell)
            obs.extend(list(obs_part))
        return sims, obs


def td_problem(dataset, settings):
    data, sites, lengths = dataset['data'], dataset['sites'], dataset['lengths']
    kminus = float(settings['kminus'])

    def model(p, Bi, seed=None):
        z = sites[Bi,]
        l = lengths[Bi,]
        obs_data = [data[i] for i in Bi]
        sim = td.simulate(p[0], kminus, p[1], p[2], T, DT, z, l, seed if seed is not None else -1)
        return sim, obs_data

    def stats_func(x, y, sf=E.stats_func_td):
        return sf(x), sf(y)
    return data, Chunked(model), stats_func


def lv_problem(dataset, settings):
    ic = dataset['ic']
    d = float(settings['d'])
    summaries = E.stats_func_lv
    data = summaries(dataset['data'])
    N_stats = data.shape[1]

    def model(p, Bi, num_reps=1, seed=None):
        alpha, beta, gamma = p
        s_sim = np.zeros((Bi.shape[0], N_stats))
        rep_rng = np.random.default_rng(seed) if seed is not None else None
        for _ in range(num_reps):
            rep_seed = int(rep_rng.integers(np.iinfo(np.int64).max)) if rep_rng is not None else -1
            _, sim = lv.tau_leaping(ic[Bi], T, alpha, beta, gamma, d, dt=DT, seed=rep_seed)
            s_sim += summaries(sim)
        s_sim /= num_reps
        s_obs = data[Bi]
        return s_sim, s_obs

    def stats_func(x, y):
        return x, y
    return data, model, stats_func


#### One synthetic fit
def number(x):
    if x is None:
        return None
    value = float(x)
    return value if np.isfinite(value) else None


def run_synthetic(fit, out=RUNS, cores=8, max_generations=MAX_GENERATIONS):
    """Run one TD or LV fit to its stopping rule and save its record. Returns the record's path."""
    model, settings = fit['model'], fit['settings']
    folder = os.path.join(out, fit['campaign'], model)
    particle_dir = os.path.join(folder, fit['tag'])
    os.makedirs(particle_dir, exist_ok=True)
    print('Generating %s data: seed=%d, parameters=%s, heterogeneities=%s'
          % (model, fit['data_seed'], settings['physical_params'], settings['heterogeneities']), flush=True)
    dataset = make_dataset(model, fit['data_seed'], settings)
    data, sim_model, stats_func = (td_problem if model == 'td' else lv_problem)(dataset, settings)
    truth = np.asarray(settings['physical_params'], dtype=float)
    sample_size = int(settings['sample_size'])
    wall_minutes = WALL_MINUTES[fit['campaign']]
    init_params = dict(data=data, model=sim_model, stats_func=stats_func,
                       prior=utils.uniform_prior(PRIOR[model], seed=fit['seed']),
                       num_particles=PARTICLES[model], cores=cores, alpha=ALPHA, seed=fit['seed'],
                       parallel_args=dict(max_nbytes='100M', timeout=99999))
    init = schemes.constant_init if fit['scheme'] == 'constant' else schemes.fvc_init
    loop = schemes.constant_loop if fit['scheme'] == 'constant' else schemes.fvc_loop

    rows = []
    start = time.monotonic()
    est = init(init_params, fit['params'])
    batch = int(est.batch_size)
    reason = None
    while reason is None:
        loop(est)
        p, w = np.asarray(est.posterior), np.asarray(est.weights)
        w = w / w.sum()
        previous = rows[-1] if rows else {}
        proposals = float(est.total_sims) - previous.get('proposals_cum', 0)
        _, intervals, widths = est.hdpr_marginal(alpha=.05)
        mean = w @ p
        reason = stop_reason(est.generation, est.acceptance_rate, est.snr, time.monotonic() - start,
                             wall_minutes, max_generations)
        v = number(est.v_total_est)
        rows.append(dict(
            generation=int(est.generation), batch_used=batch, batch_next=int(est.batch_size),
            tolerance=number(est.alpha_threshold), acceptance=number(est.acceptance_rate),
            snr=number(est.snr), ESS=float(est.ESS),
            proposals_gen=proposals, proposals_cum=float(est.total_sims),
            cells_gen=proposals * batch, cells_cum=previous.get('cells_cum', 0) + proposals * batch,
            wall_gen=float(est.step_time), wall_cum=float(est.total_time),
            log_hdpr_product=number(est.log_hdpr_product),
            hdp_widths=np.asarray(widths).tolist(), hdp_intervals=[np.asarray(i).tolist() for i in intervals],
            post_mean=mean.tolist(), post_sd=np.sqrt(w @ (p - mean) ** 2).tolist(), truth=truth.tolist(),
            v_total_est=v, v_total_ema=number(getattr(est, 'v_total', None)),
            c=number(getattr(est, 'c', None)), c_est=number(getattr(est, 'c_est_', None)),
            v_over_n=None if v is None else v / batch, v_over_N=None if v is None else v / sample_size))
        np.savez_compressed(os.path.join(particle_dir, 'particles_%03d.npz' % est.generation), posterior=p, weights=w)
        print('%s/%s gen=%d batch=%d->%d acc=%.4g SNR=%.4g seconds=%.1f'
              % (model, fit['tag'], est.generation, batch, est.batch_size, est.acceptance_rate, est.snr,
                 est.total_time), flush=True)
        batch = int(est.batch_size)

    path = os.path.join(folder, fit['tag'] + '.pkl')
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(dict(job=fit, rows=rows, stop_reason=reason), f)
    os.replace(path + '.tmp', path)
    print('Done %s/%s: %s' % (model, fit['tag'], reason), flush=True)
    return path


def run_fit(fit, out=RUNS, cores=8):
    if fit['campaign'] == 'ecoli':
        import application_kilic
        return application_kilic.run_fit(fit['scheme'], fit['size'], fit['seed'], os.path.join(out, 'ecoli'),
                                         cores=cores, time_limit=ECOLI_TIME_LIMIT)
    return run_synthetic(fit, out, cores)


#### Command line
def indices(spec, n):
    if spec == 'all':
        return list(range(n))
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    if any(i < 0 or i >= n for i in out):
        raise SystemExit('indices must lie in 0..%d' % (n - 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('list', 'run'))
    parser.add_argument('campaign', choices=tuple(CAMPAIGNS))
    parser.add_argument('fits', nargs='?', default='all', help="'all', an index, or ranges such as 0-9,12")
    parser.add_argument('--cores', type=int, default=8, help='worker processes per fit (default 8)')
    parser.add_argument('--out', default=RUNS, help='output folder (default: %(default)s)')
    args = parser.parse_args()
    fits = CAMPAIGNS[args.campaign]()
    chosen = indices(args.fits, len(fits))
    if args.command == 'list':
        for i in chosen:
            fit = fits[i]
            detail = {k: fit[k] for k in ('scheme', 'params', 'size', 'seed') if k in fit}
            if 'settings' in fit:
                detail.update({k: fit['settings'][k] for k in ('physical_params', 'heterogeneities', 'sample_size')})
            print(i, fit['tag'], json.dumps(detail))
        return
    for i in chosen:
        print('[%s %d/%d] %s' % (args.campaign, i, len(fits), fits[i]['tag']), flush=True)
        run_fit(fits[i], args.out, args.cores)


if __name__ == '__main__':
    main()
