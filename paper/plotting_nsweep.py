"""
Dataset-size sweep for the TD model: the cost of reaching a target posterior width W against
the dataset size N, for three arms,

    full batch   n = N
    fixed n=512  constant minibatch (for N > 512)
    adaptive     n_0 = 2, automatic c

each fit under the paper's stopping rule. For every fit and every target width W we read off
the cumulative simulated cells and the wall time at the first generation that reaches W. A seed
that never reaches W contributes nothing, and a target that no seed of an arm reached is absent
from the figure.

    python paper/plotting_nsweep.py [--runs paper/results/runs] [--out paper/results/figures]

Reads the records in `<runs>/nsweep/td/` and writes `td_nsweep_cells_wall` (pdf and png):
A, cumulative simulated cells against N; B, wall time against N. The median cost per
(N, arm, W) is printed as a table.
"""
import argparse
import glob
import math
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style

TARGETS = (30.0, 3.0, 0.3, 0.03)            # target widths W
MIN_SEEDS = 2                               # below this a point is drawn hollow

FULL = 'full_batch'
FVC = 'fvc_n0_2'
FIXED = 'fixed_n512'
ARMS = (FULL, FIXED, FVC)
ARM_LABEL = {FULL: r'Full batch ($n=N$)', FIXED: r'Fixed $n=512$',
             FVC: paper_style.ADAPTIVE_NAME + r' ($n_0=2$, auto $c$)'}
ARM_TABLE = {FULL: 'full batch', FIXED: 'fixed n=512', FVC: 'adaptive'}

# the line style carries the arm, the marker and the shade carry the target width
ARM_LINESTYLE = {FULL: '-', FIXED: (0, (5, 2)), FVC: '-'}

W_MARKERS = ('o', 's', '^', 'D', 'v', 'P')
MS = 5.0


def w_marker(W):
    """Marker for a target width, in ladder order (loosest first)."""
    i = list(TARGETS).index(W) if W in TARGETS else 0
    return W_MARKERS[i % len(W_MARKERS)]


def arm_colors():
    """Full batch = darkest fixed-batch colour; adaptive = the adaptive blue."""
    ylord = paper_style.sweep_colors(paper_style.CONSTANT_CMAP, 3)
    return {FULL: paper_style.FULL_BATCH_COLOR, FIXED: ylord[1], FVC: paper_style.FVC_COLOR}


def shade(color, frac):
    """Mix `color` toward white; frac = 1 keeps it, frac = 0 is white."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - frac + frac * r, 1 - frac + frac * g, 1 - frac + frac * b)


def w_shades(base):
    """Lightest for the loosest target, darkest for the tightest one."""
    fracs = np.linspace(0.45, 1.0, len(TARGETS))
    return {w: shade(base, f) for w, f in zip(TARGETS, fracs)}


# ------------------------------------------------------------------ loading
def load_fits(folder):
    """One dict per fit: N, arm, seed, generations (sorted) and stop reason."""
    fits = []
    for path in sorted(glob.glob(os.path.join(folder, '*.pkl'))):
        with open(path, 'rb') as fh:
            rec = pickle.load(fh)
        job = rec['job']
        rows = sorted((r for r in rec['rows'] if r.get('log_hdpr_product') is not None),
                      key=lambda r: r['generation'])
        fits.append({'N': int(job['settings']['sample_size']), 'arm': job['arm'],
                     'seed': job['seed'], 'rows': rows, 'stop_reason': rec['stop_reason'],
                     'label': os.path.splitext(os.path.basename(path))[0]})
    return fits


# ------------------------------------------------------------------ crossing costs
def first_crossing(fit, W):
    """(cells_cum, wall_cum) at the first generation with width <= W."""
    logW = math.log(W)
    for row in fit['rows']:
        if row['log_hdpr_product'] <= logW:
            return float(row['cells_cum']), float(row['wall_cum'])
    return np.nan, np.nan


def build_table(fits):
    """costs[(N, arm, W)] = dict(cells=[...], wall=[...]) over seeds. A wall-capped fit is
    left out for the targets it did not reach."""
    costs = {}
    for fit in fits:
        for W in TARGETS:
            cells, wall = first_crossing(fit, W)
            entry = costs.setdefault((fit['N'], fit['arm'], W), {'cells': [], 'wall': []})
            if np.isnan(cells) and fit['stop_reason'] == 'wall_cap':
                continue
            entry['cells'].append(cells)
            entry['wall'].append(wall)
    return costs


def stat(values):
    """(median, min, max) ignoring NaN; all-NaN gives NaN."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(np.nanmedian(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))


def series(costs, Ns, arm, W, field):
    """Median / min / max of `field` across seeds, over the ordered list of N."""
    med, lo, hi = [], [], []
    for N in Ns:
        entry = costs.get((N, arm, W))
        m, a, b = stat(entry[field]) if entry else (np.nan, np.nan, np.nan)
        med.append(m); lo.append(a); hi.append(b)
    return np.array(med), np.array(lo), np.array(hi)


def counts(costs, Ns, arm, W):
    """How many seeds reached W, over the ordered list of N."""
    out = []
    for N in Ns:
        entry = costs.get((N, arm, W))
        vals = np.asarray(entry['cells'], dtype=float) if entry else np.array([])
        out.append(int(np.sum(~np.isnan(vals))))
    return np.array(out)


# ------------------------------------------------------------------ plotting
def _errorbar(ax, x, med, lo, hi, cnt, color, marker, ls):
    """Median with a min-max bar; a point resting on a single seed is drawn hollow."""
    ok = ~np.isnan(med)
    if not ok.any():
        return
    x = np.asarray(x, dtype=float)[ok]
    med, lo, hi, cnt = med[ok], lo[ok], hi[ok], np.asarray(cnt)[ok]
    yerr = np.vstack([np.maximum(med - lo, 0.0), np.maximum(hi - med, 0.0)])
    h = ax.errorbar(x, med, yerr=yerr, color=color, marker=marker, linestyle=ls,
                    markersize=MS, markeredgewidth=0.0, linewidth=1.2,
                    elinewidth=paper_style.ERR_LW, capsize=paper_style.ERR_CAP,
                    capthick=paper_style.ERR_LW, zorder=3)
    paper_style.restyle_errorbar(h)      # markeredgewidth=0 would otherwise erase the caps
    thin = cnt < MIN_SEEDS
    if thin.any():
        ax.plot(x[thin], med[thin], linestyle='none', marker=marker, markersize=MS,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=0.9,
                zorder=4)


def _cost_panel(ax, costs, Ns, field, ylab, shades):
    for arm in ARMS:
        for W in TARGETS:
            med, lo, hi = series(costs, Ns, arm, W, field)
            _errorbar(ax, Ns, med, lo, hi, counts(costs, Ns, arm, W),
                      shades[arm][W], w_marker(W), ARM_LINESTYLE[arm])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Dataset size, $N$ (cells)')
    ax.set_ylabel(ylab)
    paper_style.style_axes(ax)


def _slope_guide(ax, costs, Ns):
    r"""A short grey $\propto N$ segment in the empty upper left of the panel, above the
    full-batch curves at the two smallest N."""
    if len(Ns) < 2:
        return
    highs = [series(costs, Ns, arm, W, 'cells')[0][1] for arm in ARMS for W in TARGETS]
    highs = [v for v in highs if not np.isnan(v)]
    if not highs:
        return
    x = np.array([Ns[0], Ns[1]], dtype=float)
    y = 3.2 * max(highs) * (x / x[1])
    ax.plot(x, y, color='0.6', lw=0.9, ls='-', zorder=1)
    ax.annotate(r'$\propto N$', xy=(float(np.sqrt(x[0] * x[1])), float(np.sqrt(y[0] * y[1]))),
                xytext=(0, 7), textcoords='offset points', color='0.45',
                fontsize=paper_style.FS_LEGEND, ha='center', va='bottom')


def arm_handles():
    colors = arm_colors()
    return [Line2D([], [], color=colors[a], lw=1.6, linestyle=ARM_LINESTYLE[a],
                   label=ARM_LABEL[a]) for a in ARMS]


def w_handles():
    """Grey handles on the same light-to-dark ladder as the coloured curves."""
    greys = w_shades('black')
    return [Line2D([], [], color=greys[W], lw=1.2, marker=w_marker(W), linestyle='-',
                   markersize=MS, label=r'$W=%g$' % W) for W in TARGETS]


def make_cells_wall_figure(costs, Ns, stem):
    """Two panels, both cost against N: simulated cells (A) and wall time (B), with the two
    legends in a column to the right of the panels."""
    shades = {arm: w_shades(c) for arm, c in arm_colors().items()}

    fig, axes = plt.subplots(1, 2, figsize=(paper_style.TEXT_WIDTH, 2.5))
    axA, axB = axes
    _cost_panel(axA, costs, Ns, 'cells', 'Cumulative simulated\ncells', shades)
    _slope_guide(axA, costs, Ns)
    _cost_panel(axB, costs, Ns, 'wall', 'Wall time (s)', shades)

    fig.subplots_adjust(left=0.105, right=0.735, bottom=0.2, top=0.92, wspace=0.36)
    arms = arm_handles()
    for h in arms[2:]:                 # the adaptive label on two lines to fit the column
        h.set_label(h.get_label().replace(' (', '\n('))
    fig.legend(handles=arms, loc='upper left', bbox_to_anchor=(0.75, 0.93), frameon=True,
               framealpha=0.85, edgecolor='0.75', borderpad=0.35, handlelength=1.8,
               labelspacing=0.2, borderaxespad=0.0)
    fig.legend(handles=w_handles(), loc='lower left', bbox_to_anchor=(0.75, 0.2),
               frameon=True, framealpha=0.85, edgecolor='0.75', title='Target width',
               borderpad=0.35, handlelength=1.6, labelspacing=0.2, ncol=1,
               borderaxespad=0.0)

    for ax, letter in zip(axes, 'AB'):
        paper_style.panel_label(ax, letter, dx=-0.30, dy=1.02)

    paper_style.save(fig, stem)
    plt.close(fig)


# ------------------------------------------------------------------ numbers
def numbers_table(costs, Ns):
    lines = ['| N | W | arm | seeds | median cells | median wall (s) | full / adaptive (cells) |',
             '|---|---|-----|-------|--------------|-----------------|-------------------------|']
    for N in Ns:
        for W in TARGETS:
            full = series(costs, [N], FULL, W, 'cells')[0][0]
            fvc = series(costs, [N], FVC, W, 'cells')[0][0]
            for arm in ARMS:
                entry = costs.get((N, arm, W))
                if entry is None:
                    continue
                mc, mw = stat(entry['cells'])[0], stat(entry['wall'])[0]
                ratio = '%.2f' % (full / fvc) if arm == FVC and not np.isnan(full / fvc) else ''
                cells_s, wall_s = (('not reached',) * 2 if np.isnan(mc)
                                   else ('%.4g' % mc, '%.4g' % mw))
                lines.append('| %d | %g | %s | %d | %s | %s | %s |' % (
                    N, W, ARM_TABLE[arm], counts(costs, [N], arm, W)[0], cells_s, wall_s, ratio))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(HERE, 'results', 'runs'),
                    help='fit records written by campaign.py (default: %(default)s)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'figures'),
                    help='output directory (default: %(default)s)')
    args = ap.parse_args(argv)

    paper_style.apply()
    os.makedirs(args.out, exist_ok=True)
    fits = load_fits(os.path.join(args.runs, 'nsweep', 'td'))
    if not fits:
        print('No records under %s' % os.path.join(args.runs, 'nsweep', 'td'))
        return 1
    Ns = sorted({f['N'] for f in fits})
    costs = build_table(fits)
    make_cells_wall_figure(costs, Ns, os.path.join(args.out, 'td_nsweep_cells_wall'))
    print(numbers_table(costs, Ns))
    return 0


if __name__ == '__main__':
    sys.exit(main())
