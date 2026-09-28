'''
Posteriors and convergence curves for the TD and LV models, comparing three fixed minibatch
sizes (the largest the full batch n = N) with the adaptive rule (n_0 = 2, automatic c,
lambda = 1), three seeds each.

    python paper/plotting_posteriors.py [--results paper/results] [--out paper/results/figures]

Reads `{td,lv}_constant.pkl` and `{td,lv}_fvc_c.pkl` written by `collect_results.py` and writes
(pdf and png)

    td_posteriors_corner, lv_posteriors_corner    marginal densities on the diagonal and 95%
                                                  HDR contours for each pair of parameters
    td_width_vs_cost, lv_width_vs_cost            width, tolerance, acceptance rate and SNR
                                                  against cumulative simulation cost

Posterior panels pool the stopping-generation particles of the three seeds of an arm (each
seed's weights normalised, then divided by the number of seeds). Curve panels show one thin
line per seed and the generation-wise median over the seeds.
'''
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import experiments as E
import paper_style

get_attr, get_quantiles = E.get_attr, E.get_quantiles


# ------------------------------------------------------------------ configuration
FIXED_LABELS = [r'Fixed $n=%d$', r'Fixed $n=%d$', r'Full batch $n=N=%d$']
FVC_KEY = (2, None, 1)
FVC_LABEL = paper_style.ADAPTIVE_NAME + r' ($n_0=2$, auto $c$)'

MODELS = {
    'td': dict(
        fixed_keys=[(32,), (128,), (512,)],
        cost_label='Cumulative simulated cells',
        params=[
            dict(name=r'$k_+$', prior=(1.0, 100.0), truth=15.0, scale=1.0,
                 prior_txt=r'prior $\mathcal{U}(1,\,100)$'),
            dict(name=r'$r_{\mathrm{burst}}$', prior=(1.0, 100.0), truth=10.0, scale=1.0,
                 prior_txt=r'prior $\mathcal{U}(1,\,100)$'),
            dict(name=r'$D$', prior=(0.0, 1.0), truth=0.1, scale=1.0,
                 prior_txt=r'prior $\mathcal{U}(0,\,1)$'),
        ],
    ),
    'lv': dict(
        fixed_keys=[(8,), (32,), (128,)],
        cost_label='Cumulative simulated trajectories',
        params=[
            dict(name=r'$\alpha$', prior=(0.0, 50.0), truth=5.0, scale=1.0,
                 prior_txt=r'prior $\mathcal{U}(0,\,50)$'),
            dict(name=r'$\beta$  ($\times 10^{-2}$)', prior=(0.0, 0.2), truth=0.02,
                 scale=100.0, prior_txt=r'prior $\mathcal{U}(0,\,0.2)$'),
            dict(name=r'$\gamma$', prior=(0.0, 50.0), truth=5.0, scale=1.0,
                 prior_txt=r'prior $\mathcal{U}(0,\,50)$'),
        ],
    ),
}


def load(results_dir, model):
    """{label: [replicate][generation dict]} for the four arms, in plotting order, and their colours."""
    const = E.load(os.path.join(results_dir, f'{model}_constant.pkl'))
    fvc = E.load(os.path.join(results_dir, f'{model}_fvc_c.pkl'))
    fixed_colors, fvc_color = paper_style.method_colors(3)
    arms, colors = {}, {}
    for key, tmpl, col in zip(MODELS[model]['fixed_keys'], FIXED_LABELS, fixed_colors):
        label = tmpl % key[0]
        arms[label] = const[key]
        colors[label] = col
    arms[FVC_LABEL] = fvc[FVC_KEY]
    colors[FVC_LABEL] = fvc_color
    return arms, colors


# ------------------------------------------------------------------ weighted KDE
def weighted_kde(x, w, lo, hi, grid):
    """Gaussian KDE of the weighted sample (x, w), reflected at the prior bounds so that no
    mass leaks outside the support. Scott's rule on the weighted effective sample size."""
    w = np.asarray(w, float)
    w = w / w.sum()
    x = np.asarray(x, float)
    neff = 1.0 / np.sum(w ** 2)
    mu = np.sum(w * x)
    sd = np.sqrt(max(np.sum(w * (x - mu) ** 2), 1e-300))
    h = sd * neff ** (-0.2)
    dens = np.zeros_like(grid)
    for centres in (x, 2 * lo - x, 2 * hi - x):
        z = (grid[:, None] - centres[None, :]) / h
        dens += np.exp(-0.5 * z ** 2) @ w
    return dens / (h * np.sqrt(2 * np.pi)), h


def weighted_kde2(x, y, w, gx, gy):
    """Weighted Gaussian KDE on a 2-D grid (Scott bandwidth per axis, no reflection)."""
    w = np.asarray(w, float)
    w = w / w.sum()
    neff = 1.0 / np.sum(w ** 2)
    h = neff ** (-1.0 / 6.0)
    sx = np.sqrt(np.sum(w * (x - np.sum(w * x)) ** 2)) * h
    sy = np.sqrt(np.sum(w * (y - np.sum(w * y)) ** 2)) * h
    zx = (gx[None, :] - x[:, None]) / sx
    zy = (gy[None, :] - y[:, None]) / sy
    dens = np.einsum('n,ni,nj->ij', w, np.exp(-0.5 * zy ** 2), np.exp(-0.5 * zx ** 2))
    return dens / (2 * np.pi * sx * sy)


def hdr_level(dens, mass=0.95):
    """Density value whose super-level set holds `mass` of the (grid-normalised) density."""
    d = np.sort(dens.ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return d[np.searchsorted(c, mass)]


def pooled_sample(replicates, j, scale):
    """Pool the stopping-generation particle clouds of every seed of one arm."""
    xs, ws = [], []
    for rep in replicates:
        g = rep[-1]
        xs.append(np.asarray(g['posterior'])[:, j] * scale)
        w = np.asarray(g['weights'], float)
        ws.append(w / w.sum() / len(replicates))
    return np.concatenate(xs), np.concatenate(ws)


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    return np.interp(q, np.cumsum(w) / np.sum(w), x)


def _round_out(lo, hi):
    """Round an interval outward to a tidy grid (quarter of the leading decade)."""
    step = 10.0 ** np.floor(np.log10(hi - lo)) / 4.0
    return np.floor(lo / step) * step, np.ceil(hi / step) * step


# ------------------------------------------------------------------ corner figure
def posterior_corner_figure(results_dir, model, stem, mass=0.95, n_grid=90):
    """Marginals on the diagonal and, below it, the pairwise marginals as one contour per
    method enclosing `mass` of its particles. The prior band appears on the diagonal."""
    cfg = MODELS[model]
    arms, colors = load(results_dir, model)
    p = len(cfg['params'])
    fig, axes = plt.subplots(p, p, figsize=(paper_style.TEXT_WIDTH, paper_style.TEXT_WIDTH * 0.85))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.085, hspace=0.1, wspace=0.1)

    samples = [{lab: pooled_sample(rep, j, par['scale']) for lab, rep in arms.items()}
               for j, par in enumerate(cfg['params'])]
    # each parameter's range covers every arm's bulk plus three bandwidths, within the prior
    ranges = []
    for j, par in enumerate(cfg['params']):
        lo_p, hi_p = par['prior'][0] * par['scale'], par['prior'][1] * par['scale']
        lo_d, hi_d = np.inf, -np.inf
        for x, w in samples[j].values():
            _, h = weighted_kde(x, w, lo_p, hi_p, np.array([lo_p, hi_p]))
            lo_d = min(lo_d, weighted_quantile(x, w, 0.001) - 3 * h)
            hi_d = max(hi_d, weighted_quantile(x, w, 0.999) + 3 * h)
        xlo, xhi = _round_out(lo_d, hi_d)
        ranges.append((max(xlo, lo_p), min(xhi, hi_p)))

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            par_j, par_i = cfg['params'][j], cfg['params'][i]
            xlo, xhi = ranges[j]
            if i == j:
                lo_p, hi_p = par_j['prior'][0] * par_j['scale'], par_j['prior'][1] * par_j['scale']
                grid = np.linspace(xlo, xhi, 1000)
                prior_dens = 1.0 / (hi_p - lo_p)
                ax.fill_between([xlo, xhi], 0, prior_dens, facecolor=paper_style.PRIOR_FILL,
                                linewidth=0, zorder=0)
                ax.axvline(par_j['truth'] * par_j['scale'], color=paper_style.TRUTH_COLOR,
                           ls='--', lw=1.0, zorder=2)
                peak = prior_dens
                for k, (lab, (x, w)) in enumerate(samples[j].items()):
                    dens, _ = weighted_kde(x, w, lo_p, hi_p, grid)
                    fvc = lab == FVC_LABEL
                    ax.plot(grid, dens, color=colors[lab], lw=paper_style.LW_DENSITY,
                            zorder=6 if fvc else 3 + k, solid_capstyle='round')
                    peak = max(peak, dens.max())
                ax.set_ylim(0, 1.12 * peak)
                ax.set_yticks([])
                ax.yaxis.set_minor_locator(NullLocator())
                ax.text(0.97, 0.95, par_j['prior_txt'], transform=ax.transAxes, ha='right',
                        va='top', fontsize=paper_style.FS_TICK, color='0.35')
            else:
                ylo, yhi = ranges[i]
                gx = np.linspace(xlo, xhi, n_grid)
                gy = np.linspace(ylo, yhi, n_grid)
                ax.axvline(par_j['truth'] * par_j['scale'], color=paper_style.TRUTH_COLOR,
                           ls='--', lw=0.8, zorder=2)
                ax.axhline(par_i['truth'] * par_i['scale'], color=paper_style.TRUTH_COLOR,
                           ls='--', lw=0.8, zorder=2)
                for k, lab in enumerate(arms):
                    x, w = samples[j][lab]
                    y, _ = samples[i][lab]
                    dens = weighted_kde2(x, y, w, gx, gy)
                    fvc = lab == FVC_LABEL
                    ax.contour(gx, gy, dens, levels=[hdr_level(dens, mass)],
                               colors=[colors[lab]], linewidths=paper_style.LW_DENSITY,
                               zorder=6 if fvc else 3 + k)
                ax.set_ylim(ylo, yhi)
                if j > 0:
                    ax.tick_params(axis='y', labelleft=False)
                else:
                    ax.set_ylabel(par_i['name'], labelpad=2)
            ax.set_xlim(xlo, xhi)
            paper_style.style_axes(ax)
            if i < p - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(par_j['name'], labelpad=2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            if i != j:
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            # a tick label at the right edge of one column would run into the first label of
            # the next column, so that one label is dropped
            if j < p - 1:
                ticks = [t for t in ax.get_xticks() if xlo <= t <= xhi]
                if ticks and ticks[-1] > xlo + 0.92 * (xhi - xlo):
                    ax.xaxis.set_major_locator(FixedLocator(ticks[:-1]))

    handles = [Patch(facecolor=paper_style.PRIOR_FILL, edgecolor='none', label='Prior'),
               Line2D([], [], color=paper_style.TRUTH_COLOR, ls='--', lw=1.0,
                      label='Generating value')]
    handles += [Line2D([], [], color=colors[lab], lw=paper_style.LW_DENSITY, label=lab)
                for lab in arms]
    handles.append(Line2D([], [], color='0.4', lw=paper_style.LW_DENSITY,
                          label=r'%d\%% HDR contours' % round(100 * mass)))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.975), ncol=1,
               frameon=True, handlelength=1.9, labelspacing=0.45)
    paper_style.save(fig, stem)
    plt.close(fig)


# ------------------------------------------------------------------ width and diagnostics
PANELS = [
    ('log_hdpr_product', np.exp, paper_style.Y_LABEL_WIDTH, None),
    ('alpha_threshold', None, paper_style.Y_LABEL_TOL.replace(', ', ',\n'), None),
    ('acceptance_rate', None, paper_style.Y_LABEL_ACC.replace(' ', '\n'), 0.01),
    ('snr', None, 'Signal-to-noise\n' + r'ratio, $\mathrm{SNR}_t$', 1.0),
]


def width_figure(results_dir, model, stem):
    """Width, tolerance, acceptance rate and SNR against cumulative cost; dashed lines mark
    the stopping thresholds."""
    cfg = MODELS[model]
    arms, colors = load(results_dir, model)

    cost_all = get_attr(arms, 'total_sims', trunc=False)
    cost_med = get_quantiles(get_attr(arms, 'total_sims', trunc=True))

    fig = plt.figure(figsize=(paper_style.TEXT_WIDTH, 6.3))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.0, 1.0, 1.0], hspace=0.17)
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0] if axes else None))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.965, bottom=0.075)

    xmin = min(min(min(s) for s in v) for v in cost_all.values())
    xmax = max(max(max(s) for s in v) for v in cost_all.values())

    for ax, (attr, fn, ylab, ref), letter in zip(axes, PANELS, 'ABCD'):
        raw = get_attr(arms, attr, trunc=False)
        med = get_quantiles(get_attr(arms, attr, trunc=True))
        for lab in arms:
            col = colors[lab]
            fvc = lab == FVC_LABEL
            for xs, ys in zip(cost_all[lab], raw[lab]):
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
                if fn is not None:
                    ys = fn(ys)
                ax.plot(xs, ys, color=col, lw=paper_style.LW_SEED, alpha=0.65, zorder=3)
                ax.plot(xs[-1], ys[-1], marker='o', mfc='none', mec=col, mew=1.0,
                        ms=paper_style.STOP_MS, ls='none', zorder=7, clip_on=False)
            xm = np.asarray(cost_med[lab][1], float)
            ym = np.asarray(med[lab][1], float)
            if fn is not None:
                ym = fn(ym)
            ax.plot(xm, ym, color=col, lw=paper_style.LW_MEDIAN_FVC if fvc else paper_style.LW_MEDIAN,
                    zorder=6 if fvc else 4, solid_capstyle='round')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(ylab, labelpad=4)
        ax.set_xlim(0.55 * xmin, 1.9 * xmax)
        if ref is not None:
            ax.axhline(ref, color='0.35', ls='--', lw=0.9, zorder=2)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, ref / 1.7), ymax)
            ax.text(0.72 * xmin, ref * 1.25, f'{ref:g}', ha='left', va='bottom',
                    fontsize=paper_style.FS_TICK, color='0.45')
        paper_style.style_axes(ax)
        paper_style.panel_label(ax, letter, dx=(0.005 - 0.14) / 0.845, dy=1.0)

    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel(cfg['cost_label'], labelpad=3)
    fig.align_ylabels(axes)

    handles = [Line2D([], [], color=colors[lab],
                      lw=paper_style.LW_MEDIAN_FVC if lab == FVC_LABEL else paper_style.LW_MEDIAN,
                      label=lab) for lab in arms]
    handles.append(Line2D([], [], color='0.35', marker='o', mfc='none', mew=1.0,
                          ms=paper_style.STOP_MS, ls='none', label=paper_style.STOP_LABEL))
    axes[0].legend(handles=handles, loc='lower left', frameon=True, handlelength=1.9,
                   borderaxespad=0.6)
    paper_style.save(fig, stem)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(HERE, 'results'),
                    help='sweep pickles written by collect_results.py (default: %(default)s)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'figures'),
                    help='output directory (default: %(default)s)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paper_style.apply()
    for model in ('td', 'lv'):
        width_figure(a.results, model, os.path.join(a.out, f'{model}_width_vs_cost'))
        posterior_corner_figure(a.results, model, os.path.join(a.out, f'{model}_posteriors_corner'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
