"""
The shared look of the paper's figures: rcParams, colours, line weights, axis labels and a few
drawing helpers. Every plotting script calls `apply()` once.

Figures are drawn at their printed width (TEXT_WIDTH, the manuscript's text width) and included
at that width, so the font sizes below are the printed sizes. Text is set with LaTeX, as in the
`pareto_frontier` and `time_series` helpers of `experiments.py`.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

TEXT_WIDTH = 6.5          # inches

FS_LABEL = 11.0           # axis labels, at body-text size
FS_TITLE = 11.0
FS_TICK = 8.5
FS_LEGEND = 9.5           # legends and annotations
FS_PANEL = 14.0           # panel letters

RC = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEGEND,
    'legend.title_fontsize': FS_LEGEND,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '0.75',
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.4,
    'errorbar.capsize': 2.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.facecolor': 'white',
}


def apply():
    mpl.rcParams.update(RC)


# ---------------------------------------------------------------- colours
CONSTANT_CMAP = 'YlOrRd'      # fixed minibatch sizes n_t
ADAPTIVE_CMAP = 'GnBu'        # adaptive hyperparameter sweeps
AUTO_C_COLOR = 'magenta'      # adaptive rule with automatic c
AUTO_C_MARKER = '*'
SUBOPTIMAL_FILL = '#F8DFB8'   # region already covered by the fixed-batch arms
SUBOPTIMAL_EDGE = '#E3A65C'
SUBOPTIMAL_TEXT = '#B8730A'
PRIOR_FILL = '0.88'           # posterior figures: grey prior band
TRUTH_COLOR = 'black'         # dashed generating value

# The two methods that appear in every curve and posterior figure keep one colour each: the
# adaptive rule a medium blue, the full batch (n = N) the darkest fixed-batch colour.
FVC_COLOR = '#2171B5'
FULL_BATCH_COLOR = plt.get_cmap(CONSTANT_CMAP)(0.90)

# Error bars
ERR_LW = 0.8                  # bar and cap line width
ERR_CAP = 2.0                 # capsize (points)

# Accent colours shared with the schematic figures
ACCENT = {
    'purple': '#7B5FD6',
    'blue': '#1F3CFF',
    'green': '#3C9D6B',
    'orange': '#E8961E',
}

# Line weights
LW_SEED = 0.7                 # one thin line per seed
LW_MEDIAN = 1.6               # generation-wise median over seeds
LW_MEDIAN_FVC = 2.2           # the adaptive median, drawn on top
LW_DENSITY = 1.5              # posterior marginals
LW_SWEEP = 0.9                # generation curves of the hyperparameter sweeps
STOP_MS = 5.0                 # open circle at each seed's stopping generation

# Axis and legend wording
Y_LABEL_WIDTH = r'Posterior width, $W$'
Y_LABEL_WIDTH_2L = 'Posterior width, $W$\n' + r'(product of 95\% HDR widths)'
X_LABEL_WALL = r'Wall time (s)'
X_LABEL_GEN = r'Generation, $t$'
Y_LABEL_BATCH = r'Minibatch size, $n_t$'
Y_LABEL_NOISE = r'Comparison noise, $v_t/n_t$'
Y_LABEL_TOL = r'ABC tolerance, $\epsilon_t$'
Y_LABEL_ACC = r'Acceptance rate'
STOP_LABEL = 'Stopping generation'
ADAPTIVE_NAME = 'Adaptive'
FS_SUBOPTIMAL = FS_LABEL


def sweep_colors(cmap, n, lo=0.35, hi=0.90):
    """n colours sampled from `cmap` between lo and hi, as `experiments.pareto_frontier` does."""
    cm = plt.get_cmap(cmap)
    return [cm(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]


def method_colors(n_fixed):
    """Colours for `n_fixed` fixed batch sizes in increasing order (the last one the full
    batch) and for the adaptive rule."""
    fixed = sweep_colors(CONSTANT_CMAP, n_fixed)
    fixed[-1] = FULL_BATCH_COLOR
    return fixed, FVC_COLOR


# ---------------------------------------------------------------- axes helpers
def style_axes(ax):
    """Inward major and minor ticks on all four sides, all spines visible."""
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   length=3, width=0.6)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   length=1.5, width=0.45)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if ax.get_xscale() == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    for s in ax.spines.values():
        s.set_visible(True)


def panel_label(ax, letter, dx=-0.12, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='normal', va='bottom', ha='left')


def suboptimal_text(ax, x, y, ha='center', va='center'):
    """The 'Suboptimal' label, identical in every Pareto panel."""
    ax.text(x, y, 'Suboptimal', transform=ax.transAxes, ha=ha, va=va,
            color=SUBOPTIMAL_TEXT, fontsize=FS_SUBOPTIMAL, zorder=2)


def suboptimal_wedge(ax, xs, ys, label=True, x_frac=0.55, y_frac=0.42):
    """
    The region a fixed-batch ladder already covers: everything above and to the right of the
    polyline through the fixed-batch medians (xs, ys in data units, in order of n_t). Light-orange
    fill, a darker boundary running from the top of the axes down the polyline and out to the
    right edge, and optionally the label centred in the region. Call after the axis limits are final.
    """
    from matplotlib.patches import Polygon
    # Only the non-dominated arms bound the region: read left to right, keep a point when
    # nothing cheaper already gives a narrower posterior.
    pts, best = [], np.inf
    for x, y in sorted(zip(np.asarray(xs, float), np.asarray(ys, float))):
        if y < best:
            pts.append((x, y))
            best = y
    x_hi, y_hi = ax.get_xlim()[1], ax.get_ylim()[1]
    bnd = [(pts[0][0], y_hi)] + pts + [(x_hi, pts[-1][1])]
    ax.add_patch(Polygon(bnd + [(x_hi, y_hi)], closed=True, zorder=0,
                         facecolor=SUBOPTIMAL_FILL, edgecolor='none'))
    ax.plot([p[0] for p in bnd], [p[1] for p in bnd], color=SUBOPTIMAL_EDGE, lw=0.9,
            solid_joinstyle='round', zorder=0.5, clip_on=True)
    if label:
        # in axes fraction: x a fixed way across the region, y a fixed way up from the boundary
        to_ax = lambda p: ax.transAxes.inverted().transform(ax.transData.transform(p))
        fb = np.array([to_ax(p) for p in bnd])
        fx = fb[1, 0] + x_frac * (1.0 - fb[1, 0])
        fy_b = float(np.interp(fx, fb[1:, 0], fb[1:, 1]))
        suboptimal_text(ax, fx, fy_b + y_frac * (1.0 - fy_b))


def restyle_errorbar(container, ms=None, marker=None, color=None):
    """One look for every errorbar: shared bar and cap widths, optional marker restyle."""
    line, caps, bars = container.lines
    if line is not None:
        if color is not None:
            line.set_color(color)
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)
        if ms is not None:
            line.set_markersize(ms)
        if marker is not None:
            line.set_marker(marker)
    for c in caps:
        c.set_markersize(2.0 * ERR_CAP)
        c.set_markeredgewidth(ERR_LW)
        if color is not None:
            c.set_color(color)
    for b in bars:
        b.set_linewidth(ERR_LW)
        if color is not None:
            b.set_color(color)


def save(fig, stem, formats=('pdf', 'png')):
    """Save `stem`.pdf and a 300 dpi `stem`.png, without bbox cropping so the printed size is
    exactly the figure size."""
    for ext in formats:
        fig.savefig(f'{stem}.{ext}', format=ext, dpi=300 if ext == 'png' else None, facecolor='white')
