'''
What the posterior width W is: a corner plot of one adaptive posterior for the transcriptional
dynamics model (n_0 = 2, automatic c, lambda = 1; first seed, stopping generation), with each
marginal's 95% highest-density interval highlighted and the product of interval lengths
reported as W.

    python paper/plotting_hdr_pairplot.py [--results paper/results] [--out paper/results/figures]

Reads `td_fvc_c.pkl` written by `collect_results.py` and writes `hdr_pairplot` (pdf and png).
The interval end points are the ones stored with the run (`smc_abc.hdpr`), so the W shown is
the W of the other figures.
'''
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import experiments as E
import paper_style

KEY = (2, None, 1)                       # adaptive baseline arm
SEED = 0
NAMES = [r'$k_+$', r'$r_{\mathrm{burst}}$', r'$D$']
HDR_FILL = paper_style.SUBOPTIMAL_FILL   # the highlighted interval
HDR_EDGE = paper_style.SUBOPTIMAL_EDGE
POINT = paper_style.FVC_COLOR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(HERE, 'results'),
                    help='sweep pickles written by collect_results.py (default: %(default)s)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'figures'),
                    help='output directory (default: %(default)s)')
    a = ap.parse_args()
    paper_style.apply()
    os.makedirs(a.out, exist_ok=True)
    res = E.load(os.path.join(a.results, 'td_fvc_c.pkl'))
    g = res[KEY][SEED][-1]
    X = np.asarray(g['posterior'], float)
    w = np.asarray(g['weights'], float)
    w = w / w.sum()
    ivs = [np.asarray(iv, float).reshape(-1, 2) for iv in g['hdpr_intervals']]
    lens = [float(np.sum(iv[:, 1] - iv[:, 0])) for iv in ivs]
    W = float(np.prod(lens))
    p = X.shape[1]

    fig, axes = plt.subplots(p, p, figsize=(paper_style.TEXT_WIDTH * 0.83, paper_style.TEXT_WIDTH * 0.83))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.09, hspace=0.12, wspace=0.12)
    lims = [(X[:, j].min(), X[:, j].max()) for j in range(p)]
    lims = [(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)) for lo, hi in lims]

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(X[:, j], bins=22, weights=w, color=POINT, alpha=0.85, lw=0)
                for lo, hi in ivs[j]:
                    ax.axvspan(lo, hi, facecolor=HDR_FILL, edgecolor='none', zorder=0)
                lo, hi = ivs[j][:, 0].min(), ivs[j][:, 1].max()
                y = ax.get_ylim()[1]
                ax.annotate('', xy=(lo, y * 1.02), xytext=(hi, y * 1.02),
                            arrowprops=dict(arrowstyle='|-|,widthA=0.25,widthB=0.25',
                                            color=HDR_EDGE, lw=0.9, shrinkA=0, shrinkB=0),
                            annotation_clip=False)
                ax.text(0.5 * (lo + hi), y * 1.06, r'$\ell_%d = %.3g$' % (j + 1, lens[j]),
                        ha='center', va='bottom', fontsize=paper_style.FS_LEGEND,
                        color=paper_style.SUBOPTIMAL_TEXT)
                ax.set_ylim(0, y * 1.28)
                ax.set_yticks([])
            else:
                ax.scatter(X[:, j], X[:, i], s=3.5, color=POINT, alpha=0.6, lw=0, zorder=3)
                xa, xb = ivs[j][:, 0].min(), ivs[j][:, 1].max()
                ya, yb = ivs[i][:, 0].min(), ivs[i][:, 1].max()
                ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, facecolor=HDR_FILL,
                                       edgecolor=HDR_EDGE, lw=0.9, zorder=1))
                ax.set_ylim(*lims[i])
            ax.set_xlim(*lims[j])
            paper_style.style_axes(ax)
            if i == j:
                ax.tick_params(axis='y', which='both', left=False, right=False)
            if i < p - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(NAMES[j])
            if j > 0 or i == 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(NAMES[i])
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    fig.text(0.62, 0.90,
             'Posterior width\n'
             r'$W = \prod_{i=1}^{%d} \ell_i = %.3g$' % (p, W),
             ha='left', va='top', fontsize=paper_style.FS_LABEL + 1)
    fig.text(0.62, 0.80,
             'Shaded: marginal 95\\% HDR\n'
             'intervals, of length $\\ell_i$;\n'
             'boxes: their products for\n'
             'pairs of parameters.',
             ha='left', va='top', fontsize=paper_style.FS_LEGEND + 0.5, color='0.3')
    paper_style.save(fig, os.path.join(a.out, 'hdr_pairplot'))
    plt.close(fig)
    print('interval lengths', lens, 'W', W)


if __name__ == '__main__':
    main()
