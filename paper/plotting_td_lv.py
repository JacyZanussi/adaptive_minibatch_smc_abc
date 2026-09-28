'''
Pareto figures for the transcriptional dynamics (TD) and Lotka-Volterra (LV) models: wall time
at the stopping generation against the posterior width W, for fixed minibatches n_t and the
adaptive rule.

    python paper/plotting_td_lv.py [--results paper/results] [--out paper/results/figures]

Reads the sweep pickles written by `collect_results.py` and writes (pdf and png)

    td_hyperparameters, lv_hyperparameters            sweeps of lambda, n_0 and c, with the
                                                      minibatch size and comparison noise per generation
    td_physical_parameters, lv_physical_parameters    one panel per generating-parameter value
    td_heterogeneity, lv_heterogeneity                one panel per heterogeneity setting

The panels are drawn by `experiments.pareto_frontier` and `experiments.time_series` into axes
passed with `fig_ax`; the shaded suboptimal region, legends, axis limits and panel letters are
added here.
'''
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import experiments as E
import paper_style

# `pareto_frontier` loads its two inputs with `load()`. Let an already-loaded results dict pass
# through, so that a grid panel can hand it a subset of a sweep.
E.load = (lambda f, _load=E.load: f if isinstance(f, dict) else _load(f))

X_LABEL = paper_style.X_LABEL_WALL
Y_LABEL = paper_style.Y_LABEL_WIDTH_2L
Y_LABEL_1L = paper_style.Y_LABEL_WIDTH
FS_LABEL, FS_TICK, FS_TITLE = paper_style.FS_LABEL, paper_style.FS_TICK, paper_style.FS_TITLE


# --------------------------------------------------------------------------- #
#  small helpers                                                               #
# --------------------------------------------------------------------------- #
def hdr_volume(results):
    """Posterior width W at each replicate's own stopping generation. Passed to
    `pareto_frontier` as a callable y_attr so that the axis carries widths, not logs."""
    raw = E.get_attr(results, 'log_hdpr_product', trunc=False, slice=-1)
    return {k: np.exp(np.asarray(v, dtype=float)) for k, v in raw.items()}
hdr_volume.__name__ = 'hdr_volume'


def noise(results):
    """Comparison noise v_t / n_t over generations."""
    v = E.get_attr(results, 'v_total_est', trunc=True)
    n = E.get_attr(results, 'batch_size', trunc=True)
    return {k: np.asarray(v[k], dtype=float) / np.asarray(n[k], dtype=float)
            for k in results}
noise.__name__ = 'noise'


def stop_quantiles(results, attr):
    """Quartiles across replicates of `attr` at each replicate's own last generation."""
    raw = E.get_attr(results, attr, trunc=False, slice=-1)
    return E.get_quantiles({k: np.asarray(v, dtype=float) for k, v in raw.items()})


def mean_c(results):
    raw = E.get_attr(results, 'c', trunc=False, slice=-1)
    return {k: float(np.mean(v)) for k, v in raw.items()}


def restyle(ax, label=FS_LABEL, tick=FS_TICK, title=FS_TITLE):
    ax.xaxis.label.set_size(label)
    ax.yaxis.label.set_size(label)
    ax.title.set_size(title)
    ax.tick_params(axis='both', which='major', labelsize=tick)


def eb_style(h, color=None, ms=None, marker=None, lw=None):
    """Restyle one errorbar container: shared bar and cap widths, plus the marker, colour and
    connecting-line width asked for."""
    paper_style.restyle_errorbar(h, ms=ms, marker=marker, color=color)
    if lw is not None and h.lines[0] is not None:
        h.lines[0].set_linewidth(lw)


def shift_x(h, dx):
    """Move an errorbar container along x (`time_series` indexes generations from 0)."""
    line, caps, bars = h.lines
    if line is not None:
        line.set_xdata(np.asarray(line.get_xdata(), dtype=float) + dx)
    for c in caps:
        c.set_xdata(np.asarray(c.get_xdata(), dtype=float) + dx)
    for b in bars:
        segs = [np.array(s, dtype=float) for s in b.get_segments()]
        for s in segs:
            s[:, 0] += dx
        b.set_segments(segs)


def clear_legends(ax):
    for child in list(ax.get_children()):
        if isinstance(child, Legend):
            child.remove()


# Legend text stays at paper_style.FS_LEGEND; the boxes are made to fit by shrinking the
# handle, the padding and the marker, never the type.
LEG_KW = dict(framealpha=0.92, edgecolor='0.75', handlelength=0.8,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.16,
              borderaxespad=0.3, markerscale=0.65, columnspacing=0.5)


def legend_clash(ax, legs, xq, yq):
    """True when a legend box covers a plotted point or the end of its quartile bars."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = 4.0 * fig.dpi / 72.0        # a marker radius of clearance
    boxes = [l.get_window_extent(r).expanded(1.0, 1.0).padded(pad) for l in legs]
    for qx, qy in zip(xq, yq):
        for px, py in ((qx[1], qy[1]), (qx[0], qy[1]), (qx[2], qy[1]),
                       (qx[1], qy[0]), (qx[1], qy[2])):
            dx, dy = ax.transData.transform((px, py))
            if any(b.contains(dx, dy) for b in boxes):
                return True
    return False


def one_legend(ax, handles, labels, title, loc, fs, ncol=1):
    clear_legends(ax)
    leg = ax.legend(handles, labels, title=title, loc=loc, ncol=ncol,
                    fontsize=fs, title_fontsize=fs, **LEG_KW)
    leg.set_zorder(20)
    return leg


def strip_legend(fig, groups, y, fs, x=0.5):
    """One framed single-row legend for the entries that every panel of a figure shares.
    `groups` = [(title, handles, labels), ...]; each title is an entry with a blank handle,
    so the row reads 'Constant n_t: 32 64 ...  Adaptive n_0: 2 8 ...'."""
    H, L = [], []
    for g, (title, hs, ls) in enumerate(groups):
        H.append(Line2D([], [], linestyle='none', marker=''))
        L.append((r'\quad ' if g else '') + title)
        H += list(hs)
        L += list(ls)
    leg = fig.legend(H, L, loc='upper center', bbox_to_anchor=(x, y), ncol=len(H),
                     fontsize=fs, frameon=True, framealpha=0.92, edgecolor='0.75',
                     handlelength=0.9, handletextpad=0.35, columnspacing=0.9,
                     borderpad=0.35, borderaxespad=0.0, markerscale=0.9)
    return leg


def suboptimal_wedge(ax, xs, ys, label=True, **frac):
    """Shade the region the fixed-minibatch arms already cover (see paper_style), in place of
    the grey median-to-median line that `pareto_frontier` draws."""
    for ln in ax.get_lines():
        if ln.get_color() == 'gray' and ln.get_alpha() == 0.5:
            ln.set_visible(False)
    paper_style.suboptimal_wedge(ax, xs, ys, label=label, **frac)


def _sci_tick(v, _=None):
    k = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10.0 ** k
    if abs(m - 1.0) < 1e-6:
        return rf'$10^{{{k}}}$'
    return rf'${m:g}\times 10^{{{k}}}$'


def tidy_log_yticks(ax, max_ticks=3):
    """Decade labels only; minor ticks stay unlabelled. A panel that spans less than two
    decades would otherwise carry a single label, so it gets two or three round values."""
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    n_dec = sum(1 for k in range(-12, 12) if lo <= 10.0 ** k <= hi)
    if n_dec < 2:
        cands = [m * 10.0 ** k for k in range(-12, 12) for m in (1, 2, 5)]
        inside = [c for c in cands if lo * 1.03 <= c <= hi / 1.03]
        if len(inside) > max_ticks:
            idx = np.linspace(0, len(inside) - 1, max_ticks).round().astype(int)
            inside = [inside[i] for i in sorted(set(idx))]
        if inside:
            ax.yaxis.set_major_locator(mticker.FixedLocator(inside))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_sci_tick))


def const_label(v, N):
    return rf'${v:g} = N$' if v == N else rf'${v:g}$'


def approx_label(c):
    """The automatic-c arm is labelled by the mean c it chose, e.g. '$\\approx 0.14$'."""
    return r'$\approx ' + f'{c:.2f}' + '$'


def save(fig, stem):
    paper_style.save(fig, stem)
    plt.close(fig)
    print('Saved:', stem + '.{pdf,png}')


# --------------------------------------------------------------------------- #
#  one Pareto panel                                                            #
# --------------------------------------------------------------------------- #
def pareto_panel(ax, ref, novel, key_idx, legend_title, N, fs, title=None, wedge_label=True,
                 markersize=4.0, y_head=2.2, x_pad=0.06, wedge_frac=None, legend=False):
    """Fixed-batch (`ref`) and adaptive (`novel`) arms at their stopping generations: median and
    quartiles of wall time and W. With `legend`, the panel keys its own adaptive sweep."""
    ref_res = E.load(ref)
    novel_res = E.load(novel)

    fig, ax, rh, nh, ah = E.pareto_frontier(
        ref_res, novel_res, fig_ax=(ax.figure, ax),
        x_attr='total_time', y_attr=hdr_volume,
        legend_key_index_ref=0, legend_key_index=key_idx,
        ref_cmap=paper_style.CONSTANT_CMAP, novel_cmap=paper_style.ADAPTIVE_CMAP,
        auto_c_color=paper_style.AUTO_C_COLOR, auto_c_marker=paper_style.AUTO_C_MARKER)

    for h in rh:                       # fixed batches: squares, adaptive: circles,
        eb_style(h, ms=markersize, marker='s')          # automatic c: magenta star
    for h in nh:
        eb_style(h, ms=markersize, marker='o')
    for h in ah:
        eb_style(h, ms=markersize + 3.5, marker=paper_style.AUTO_C_MARKER)

    # ---- axes: widths on a log axis, cost linear from zero -------------------
    ax.set_yscale('log')
    xq = stop_quantiles(ref_res, 'total_time'), stop_quantiles(novel_res, 'total_time')
    yq = hdr_volume(ref_res), hdr_volume(novel_res)
    yq = [E.get_quantiles(d) for d in yq]
    x_hi = max(q[2] for d in xq for q in d.values())
    y_lo = min(q[0] for d in yq for q in d.values())
    y_hi = max(q[2] for d in yq for q in d.values())
    ax.set_xlim(0.0, x_hi * (1.0 + x_pad))
    ax.set_ylim(y_lo / 1.35, y_hi * y_head)

    ref_x = {k: q[1] for k, q in xq[0].items()}
    ref_y = {k: q[1] for k, q in yq[0].items()}

    # ---- legends -------------------------------------------------------------
    rl = [const_label(k[0], N) for k in ref_x]
    cbar = mean_c(novel_res)
    nkeys = [k for k in novel_res if k[key_idx] is not None]
    akeys = [k for k in novel_res if k[key_idx] is None]
    nl = [rf'${k[key_idx]:g}$' for k in nkeys]
    al = [approx_label(cbar[k]) for k in akeys]
    # the entries shared by every panel are drawn once per figure by the caller
    ax.legend_handles_ = (rh, rl, nh + ah, nl + al)
    clear_legends(ax)
    legs = []
    if legend:
        legs = [one_legend(ax, nh + ah, nl + al, 'Adaptive $' + legend_title + '$',
                           'upper right', fs)]

    # When the legend box would cover a point, open up the top of the (log) width axis until
    # nothing is hidden. The wedge is drawn afterwards so that it reaches the final top edge.
    all_xq = list(xq[0].values()) + list(xq[1].values())
    all_yq = list(yq[0].values()) + list(yq[1].values())
    for _ in range(6):
        if not legs or not legend_clash(ax, legs, all_xq, all_yq):
            break
        ax.set_ylim(y_lo / 1.35, ax.get_ylim()[1] * 1.8)
    if legs and legend_clash(ax, legs, all_xq, all_yq):
        print(f'  ! legend still covers data in panel {title}')

    suboptimal_wedge(ax, list(ref_x.values()), [ref_y[k] for k in ref_x], label=wedge_label,
                     **(wedge_frac or {}))

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel('')
    if title is not None:
        ax.set_title(title)
    tidy_log_yticks(ax)
    restyle(ax)
    return ax


# --------------------------------------------------------------------------- #
#  hyperparameter composites                                                   #
# --------------------------------------------------------------------------- #
def hyperparameter_composite(results_dir, model, N, stem):
    """Row A: Pareto panels for the lambda, n_0 and c sweeps. Row B: minibatch size per
    generation for the same sweeps. Row C: comparison noise per generation, with the fixed
    batches in the last panel."""
    p = lambda name: os.path.join(results_dir, name + '.pkl')
    const = p(f'{model}_constant')
    sweeps = [
        (p(f'{model}_fvc_lambda'), 2, r'\lambda'),
        (p(f'{model}_fvc_n0'),     0, r'n_0'),
        (p(f'{model}_fvc_c'),      1, r'c'),
    ]
    fs = paper_style.FS_LEGEND

    # The fixed-batch ladder, identical in every panel, gets one legend row across the top of
    # the figure; each row-A panel keys its own adaptive sweep, which also keys rows B and C
    # below it (same colours, same column).
    fig = plt.figure(figsize=(paper_style.TEXT_WIDTH, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.90, 0.90], hspace=0.38,
                          left=0.13, right=0.985, top=0.91, bottom=0.07)
    axA = gs[0].subgridspec(1, 3, wspace=0.30).subplots()
    axB = gs[1].subgridspec(1, 3, wspace=0.30).subplots()
    axC = gs[2].subgridspec(1, 4, wspace=0.42).subplots()

    # ---- A: Pareto frontiers -------------------------------------------------
    for ax, (fpath, key_idx, ltitle) in zip(axA, sweeps):
        pareto_panel(ax, const, fpath, key_idx, ltitle, N, fs, y_head=4.5,
                     legend=True, wedge_frac=dict(x_frac=0.62, y_frac=0.26))
    axA[0].set_ylabel(Y_LABEL)
    restyle(axA[0])
    rh, rl, _, _ = axA[0].legend_handles_
    strip_legend(fig, [('Constant $n_t$:', rh, rl)], 0.995, fs)

    # ---- B and C: generation curves -----------------------------------------
    def curves(ax, fpath, key_idx, ltitle, attr, cmap, log_y, N_label=None,
               markersize=2.4, marker='o'):
        res = E.load(fpath)
        keys = list(res.keys())
        fig_, ax_, handles = E.time_series(
            fpath, attr_list=[attr], attr_ylabels=[r'\;'], attr_transform=['id'],
            legend_key_index=key_idx, legend_title='$' + ltitle + '$',
            errorbar_cmap=cmap, fig_ax=(ax.figure, ax))
        sweep_i = [i for i, k in enumerate(keys) if k[key_idx] is not None]
        auto_i = [i for i, k in enumerate(keys) if k[key_idx] is None]
        cols = paper_style.sweep_colors(cmap, len(sweep_i))
        # thin connecting lines: the markers carry the generations
        for c, i in zip(cols, sweep_i):
            eb_style(handles[i], color=c, ms=markersize, marker=marker, lw=paper_style.LW_SWEEP)
        for i in auto_i:
            eb_style(handles[i], color=paper_style.AUTO_C_COLOR,
                     ms=markersize + 2.6, marker=paper_style.AUTO_C_MARKER,
                     lw=paper_style.LW_SWEEP + 0.2)
            handles[i].lines[0].set_zorder(6)
        for h in handles:
            shift_x(h, 1.0)                     # generations are numbered from 1
        clear_legends(ax)                       # keyed by the row-A legend of the same column
        if log_y:
            ax.set_yscale('log')
        ax.relim()
        ax.autoscale_view()
        x_max = max(np.max(h.lines[0].get_xdata()) for h in handles)
        ax.set_xlim(0.4, x_max + 0.6)
        if log_y:
            tidy_log_yticks(ax)
        ax.set_xlabel(paper_style.X_LABEL_GEN)
        ax.set_ylabel('')
        restyle(ax)

    for ax, (fpath, key_idx, ltitle) in zip(axB, sweeps):
        curves(ax, fpath, key_idx, ltitle, 'batch_size', paper_style.ADAPTIVE_CMAP, False)
    axB[0].set_ylabel(paper_style.Y_LABEL_BATCH)

    panelsC = sweeps + [(const, 0, r'n_t')]
    for ax, (fpath, key_idx, ltitle) in zip(axC, panelsC):
        is_const = (fpath == const)
        curves(ax, fpath, key_idx, ltitle, noise,
               paper_style.CONSTANT_CMAP if is_const else paper_style.ADAPTIVE_CMAP,
               True, N_label=(N if is_const else None), markersize=2.0,
               marker=('s' if is_const else 'o'))
    axC[0].set_ylabel(paper_style.Y_LABEL_NOISE.replace(', ', ',\n'))
    for ax in axC:                    # narrow panels: label every fourth generation
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    for ax in list(axA) + list(axB) + list(axC):
        restyle(ax)
    axA[0].yaxis.label.set_size(FS_LABEL)
    # panel letters at the left edge of the figure, clear of the two-line y label
    for ax, letter in ((axA[0], 'A'), (axB[0], 'B'), (axC[0], 'C')):
        pos = ax.get_position()
        paper_style.panel_label(ax, letter, dx=(0.008 - pos.x0) / pos.width, dy=1.02)
    save(fig, stem)


# --------------------------------------------------------------------------- #
#  robustness grids                                                            #
# --------------------------------------------------------------------------- #
def match(key, batch_sizes, slots):
    """Keep a key whose minibatch entry is one we plot and whose generating-parameter and
    heterogeneity slots all equal the wanted values."""
    if key[0] not in batch_sizes:
        return False
    for idx, want in slots:
        have = key[idx]
        if isinstance(want, tuple):
            if not (isinstance(have, tuple) and len(have) == len(want)
                    and np.allclose(have, want)):
                return False
        elif not np.isclose(float(have), float(want)):
            return False
    return True


def pareto_grid(ref_all, novel_all, cells, stem, suptitle, nrows, ncols, figsize,
                n_ts, n0s, N, fs=paper_style.FS_LEGEND, y_head=3.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(cells):]:
        ax.set_visible(False)
    for ax, (title, ref_slots, novel_slots) in zip(axes, cells):
        # sorted by minibatch size, so that the colour ramp and the legend follow n
        ref = {k: ref_all[k] for k in sorted(
            (k for k in ref_all if match(k, n_ts, ref_slots)), key=lambda k: float(k[0]))}
        nov = {k: novel_all[k] for k in sorted(
            (k for k in novel_all if match(k, n0s, novel_slots)), key=lambda k: float(k[0]))}
        assert len(ref) == len(n_ts) and len(nov) == len(n0s), (title, len(ref), len(nov))
        pareto_panel(ax, ref, nov, 0, r'n_0', N, fs, title=title,
                     wedge_label=False, markersize=3.2, y_head=y_head)
        ax.set_ylabel('')
    for i, ax in enumerate(axes[:len(cells)]):
        if i % ncols == 0:
            ax.set_ylabel(Y_LABEL_1L)
        restyle(ax)
    # The two legends are identical in all panels, so they are drawn once: as a single row
    # under the suptitle, or, when the grid has an empty cell, side by side in that cell.
    H = fig.get_size_inches()[1]
    st = fig.suptitle(suptitle, fontsize=paper_style.FS_TITLE + 1, y=1.0 - 0.06 / H, va='top')
    st.set_in_layout(False)
    rh, rl, nh, nl = axes[0].legend_handles_
    if len(cells) < len(axes):
        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.30 / H), h_pad=0.6)
        W = fig.get_size_inches()[0]
        last = axes[len(cells) - 1].get_position()
        y_mid = 0.5 * (last.y0 + last.y1)
        box = dict(fontsize=fs, title_fontsize=fs, framealpha=0.92, edgecolor='0.75',
                   handlelength=1.2, handletextpad=0.4, borderpad=0.4, labelspacing=0.25,
                   borderaxespad=0.0, loc='center left')
        lc = fig.legend(rh, rl, title='Constant $n_t$',
                        bbox_to_anchor=(last.x1 + 0.3 / W, y_mid), **box)
        fig.canvas.draw()
        x_next = lc.get_window_extent().transformed(fig.transFigure.inverted()).x1
        fig.legend(nh, nl, title='Adaptive $n_0$',
                   bbox_to_anchor=(x_next + 0.12 / W, y_mid), **box)
    else:
        strip_legend(fig, [('Constant $n_t$:', rh, rl), ('Adaptive $n_0$:', nh, nl)],
                     1.0 - 0.31 / H, fs)
        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.50 / H), h_pad=0.6)
    fig.align_ylabels(axes[:len(cells)])
    save(fig, stem)


#   TD fixed-batch key: (n_t, k+, r_burst, D, (2,2), (shape, scale))
#   TD adaptive key:    (n_0, None, 1, k+, r_burst, D, (2,2), (shape, scale))
TD_PHYS_BASE, TD_HET_BASE = (15, 10, 0.1), (10, 0.1)
TD_NT, TD_N0 = (128, 256, 512), (2, 8, 32)
#   LV fixed-batch key: (n_t, alpha, beta, gamma, R)
#   LV adaptive key:    (n_0, None, 1, alpha, beta, gamma, R)
LV_PHYS_BASE, LV_R_BASE = (5, 0.02, 5), 1000
LV_NT, LV_N0, LV_N0_HET = (32, 64, 128), (2, 4, 8), (2, 4, 8, 16)


def phys_cells(names, base, het_slot, het_base, ref_off=1, novel_off=3):
    cells = []
    for latex, off, values in names:
        for v in values:
            phys = list(base)
            phys[off] = v
            ref = [(ref_off + i, phys[i]) for i in range(3)] + [(het_slot[0], het_base)]
            nov = [(novel_off + i, phys[i]) for i in range(3)] + [(het_slot[1], het_base)]
            cells.append((rf'${latex} = {v:g}$', ref, nov))
    return cells


def td_het_cells():
    shapes = [(10 / 3, r'\frac{10}{3}'), (10, '10'), (30, '30')]
    scales = [(1 / 30, r'\frac{1}{30}'), (1 / 10, r'\frac{1}{10}'), (3 / 10, r'\frac{3}{10}')]
    cells = []
    for sh, shl in shapes:
        for sc, scl in scales:
            het = (sh, sc)
            ref = [(1 + i, TD_PHYS_BASE[i]) for i in range(3)] + [(5, het)]
            nov = [(3 + i, TD_PHYS_BASE[i]) for i in range(3)] + [(7, het)]
            cells.append((rf'$({shl},\ {scl})$', ref, nov))
    return cells


def lv_het_cells():
    cells = []
    for R in (1, 250, 500, 750, 1000):
        ref = [(1 + i, LV_PHYS_BASE[i]) for i in range(3)] + [(4, R)]
        nov = [(3 + i, LV_PHYS_BASE[i]) for i in range(3)] + [(6, R)]
        cells.append((rf'$R = {R}$', ref, nov))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(HERE, 'results'),
                    help='sweep pickles written by collect_results.py (default: %(default)s)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'figures'),
                    help='output directory (default: %(default)s)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load = lambda name: E.load(os.path.join(a.results, name + '.pkl'))
    out = lambda stem: os.path.join(a.out, stem)

    paper_style.apply()

    hyperparameter_composite(a.results, 'td', 512, out('td_hyperparameters'))
    hyperparameter_composite(a.results, 'lv', 128, out('lv_hyperparameters'))
    pareto_grid(
        load('td_constant_physical'), load('td_fvc_physical'),
        phys_cells([(r'k_+', 0, [3, 15, 75]),
                    (r'r_\mathrm{burst}', 1, [2, 10, 50]),
                    (r'D', 2, [0.02, 0.1, 0.5])], TD_PHYS_BASE, (5, 7), TD_HET_BASE),
        out('td_physical_parameters'),
        r'Baseline parameters: $(k_+,\ r_\mathrm{burst},\ D) = (15,\ 10,\ 0.1)$',
        3, 3, (paper_style.TEXT_WIDTH, 6.2), TD_NT, TD_N0, 512)
    pareto_grid(
        load('lv_constant_physical'), load('lv_fvc_physical'),
        phys_cells([(r'\alpha', 0, [1, 5, 25]),
                    (r'\beta', 1, [0.004, 0.02, 0.1]),
                    (r'\gamma', 2, [1, 5, 25])], LV_PHYS_BASE, (4, 6), LV_R_BASE),
        out('lv_physical_parameters'),
        r'Baseline parameters: $(\alpha,\ \beta,\ \gamma) = (5,\ 0.02,\ 5)$',
        3, 3, (paper_style.TEXT_WIDTH, 6.2), LV_NT, LV_N0, 128)
    pareto_grid(
        load('td_constant_heterogeneity'), load('td_fvc_heterogeneity'),
        td_het_cells(), out('td_heterogeneity'),
        r'Baseline cell-size heterogeneity: $(\mathrm{shape},\ \mathrm{scale}) = '
        r'\left(10,\ \frac{1}{10}\right)$',
        3, 3, (paper_style.TEXT_WIDTH, 6.2), TD_NT, TD_N0, 512)
    pareto_grid(
        load('lv_constant_heterogeneity'), load('lv_fvc_heterogeneity'),
        lv_het_cells(), out('lv_heterogeneity'),
        r'Baseline initial-condition range: $R = 1000$',
        2, 3, (paper_style.TEXT_WIDTH, 4.1), LV_NT, LV_N0_HET, 128)


if __name__ == '__main__':
    main()
