"""
Gather the per-fit records written by `campaign.py` into the sweep pickles that the TD/LV
plotting scripts read. The format is that of `experiments.simulate_experiment`:

    results[param_tuple] = [replicate_1, replicate_2, ...]     (ordered by replicate)
    replicate            = [generation_1, generation_2, ...]   (dicts of estimator attributes)

with the attribute names used by `experiments.get_attr` (`total_time`, `total_sims`,
`batch_size`, `alpha_threshold`, `acceptance_rate`, `snr`, `v_total_est`, `c`,
`log_hdpr_product`, ...). Parameter tuples follow `experiments.to_key`: the scheme
parameters, then (robustness grids) the generating parameters and heterogeneities.

    python paper/collect_results.py [--runs paper/results/runs] [--out paper/results]

Writes, for model in td and lv:

    {model}_constant.pkl, {model}_fvc_lambda.pkl, {model}_fvc_n0.pkl, {model}_fvc_c.pkl
    {model}_constant_physical.pkl, {model}_fvc_physical.pkl
    {model}_constant_heterogeneity.pkl, {model}_fvc_heterogeneity.pkl

The adaptive baseline arm (n0 = 2, automatic c, lambda = 1) belongs to all three
hyperparameter sweeps, and the baseline generating condition to both robustness families.
The last generation of each replicate also carries its particles and weights.
"""
import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from campaign import DEFAULTS

# record row key -> attribute name used by the plotting helpers
RENAME = {
    'wall_cum': 'total_time',
    'cells_cum': 'total_sims',
    'batch_used': 'batch_size',
    'tolerance': 'alpha_threshold',
    'acceptance': 'acceptance_rate',
    'hdp_intervals': 'hdpr_intervals',
}
KEEP = ['generation', 'log_hdpr_product', 'v_total_est', 'snr', 'c']


def load_fit(path):
    """The fit's job and its generations; the last generation carries the particles."""
    with open(path, 'rb') as f:
        rec = pickle.load(f)
    gens = [dict({RENAME[k]: r[k] for k in RENAME if k in r}, **{k: r[k] for k in KEEP if k in r})
            for r in rec['rows']]
    npz = os.path.join(path[:-4], f"particles_{rec['rows'][-1]['generation']:03d}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        gens[-1]['posterior'] = z['posterior']
        gens[-1]['weights'] = z['weights']
    return rec['job'], gens


def _num(x):
    """Normalise numbers so that 1 and 1.0, or 0.020000000000000004 and 0.02, give one key."""
    if x is None:
        return None
    x = float(x)
    return int(x) if x == int(x) else float(f'{x:.6g}')


def _het(h):
    return tuple(_num(y) for y in h) if isinstance(h, (list, tuple)) else _num(h)


def key_main(job):
    return tuple(_num(x) for x in job['params'])


def key_robust(job):
    s = job['settings']
    return (tuple(_num(x) for x in s['scheme_params'])
            + tuple(_num(x) for x in s['physical_params'])
            + tuple(_het(h) for h in s['heterogeneities']))


def is_baseline(job):
    s, d = job['settings'], DEFAULTS[job['model']]
    return ([_num(x) for x in s['physical_params']] == [_num(x) for x in d['physical_params']]
            and [_het(h) for h in s['heterogeneities']] == [_het(h) for h in d['heterogeneities']])


def gather(folder):
    """[(job, generations)] for every record in `folder`, in replicate order."""
    fits = [load_fit(p) for p in glob.glob(os.path.join(folder, '*.pkl'))]
    return sorted(fits, key=lambda f: f[0]['rep'])


def write(out, stem, data):
    with open(os.path.join(out, stem + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def collect_main(runs, model, out):
    sweeps = defaultdict(lambda: defaultdict(list))   # file stem -> key -> replicates
    for job, gens in gather(os.path.join(runs, 'main', model)):
        key = key_main(job)
        if job['scheme'] == 'constant':
            sweeps[f'{model}_constant'][key].append(gens)
            continue
        n0, c, lam = job['params']
        baseline = (n0 == 2 and c is None and float(lam) == 1.0)
        if baseline or lam != 1.0:
            sweeps[f'{model}_fvc_lambda'][key].append(gens)
        if baseline or n0 != 2:
            sweeps[f'{model}_fvc_n0'][key].append(gens)
        if baseline or c is not None:
            sweeps[f'{model}_fvc_c'][key].append(gens)
    for stem, data in sweeps.items():
        write(out, stem, dict(sorted(data.items(),
                                     key=lambda kv: [(-1e9 if x is None else float(x)) for x in kv[0]])))
        print(stem, {k: len(v) for k, v in data.items()})


def collect_robust(runs, model, out):
    sweeps = defaultdict(lambda: defaultdict(list))
    for job, gens in gather(os.path.join(runs, 'robustness', model)):
        families = ['physical', 'heterogeneity'] if is_baseline(job) else [job['family']]
        for family in families:
            sweeps[f'{model}_{job["scheme"]}_{family}'][key_robust(job)].append(gens)
    for stem, data in sweeps.items():
        write(out, stem, dict(sorted(data.items(), key=lambda kv: str(kv[0]))))
        print(stem, len(data), 'conditions,', sorted({len(v) for v in data.values()}), 'replicates each')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default=os.path.join(HERE, 'results', 'runs'),
                   help='fit records written by campaign.py (default: %(default)s)')
    p.add_argument('--out', default=os.path.join(HERE, 'results'),
                   help='where the sweep pickles are written (default: %(default)s)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for model in ('td', 'lv'):
        collect_main(a.runs, model, a.out)
        collect_robust(a.runs, model, a.out)


if __name__ == '__main__':
    main()
