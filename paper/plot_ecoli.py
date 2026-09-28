"""
E. coli figures: fixed minibatches and the adaptive rule on the Kilic et al. (2023) Fig. 4
mature-RNA counts, compared with the full-likelihood posterior.

    python paper/plot_ecoli.py [--runs paper/results/runs]
                               [--reference paper/results/likelihood_reference.pkl]
                               [--out paper/results/figures] [--only main|w1|estimates|all]

Writes (pdf and png)

    ecoli_main                A, the observation model (simulated cells, each counted at its own
                              fixation time) and the cells per fixation time; B, accuracy against
                              wall time at the stopping generation; C, marginal posteriors
    ecoli_w1_vs_cost          accuracy, tolerance, acceptance rate and SNR against cumulative
                              simulated cells
    ecoli_estimates_vs_cost   posterior mean of each rate against cumulative simulated cells

Inputs
    fit records   `<runs>/ecoli/ecoli_{scheme}_n{size:04d}_s{seed}.pkl` written by
                  `kilic/application_kilic.py`, each `(generations, posterior, weights, meta)`;
                  `generations[g]` carries `total_time` (s), `sim_cells`, `alpha_threshold`,
                  `acceptance`, `snr`, and `posterior` (rates) and `weights`
    reference     the full-likelihood posterior written by `kilic/likelihood_reference.py`
                  (`samples_log10`, `map_log10`)
    Kilic chain   the Source Data of Kilic et al. (2023) Fig. 4, for their posterior means

Accuracy is the mean over the four rates of the marginal Wasserstein-1 distance, in decades,
between the weighted ABC particles (log10 rates) and the reference draws. The stopping
generation is the first generation > 5 with acceptance < 0.01 or SNR < 1 within the fit's
wall-time limit; every marker, curve end point and panel-C particle cloud is taken there.
"""

import argparse
import io
import os
import pickle
import sys
import zipfile
from collections import OrderedDict
from itertools import combinations

import numpy as np
from scipy.stats import wasserstein_distance

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.transforms import blended_transform_factory   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style                                     # noqa: E402

DEF_RUNS = os.path.join(HERE, "results", "runs")
DEF_REFERENCE = os.path.join(HERE, "results", "likelihood_reference.pkl")
DEF_KILIC = os.path.join(REPO, "datasets", "kilic2023_fig4", "kilic2023_fig4_source_data.zip")
DEF_OUT = os.path.join(HERE, "results", "figures")

PARAM_NAMES = ["K_12", "K_21", "B_1", "B_2"]
PARAM_TEX = [r"$K_{1,2}$ ($10^{%d}$ s$^{-1}$)", r"$K_{2,1}$ ($10^{%d}$ s$^{-1}$)",
             r"$B_1$ ($10^{%d}$ s$^{-1}$)", r"$B_2$ ($10^{%d}$ s$^{-1}$)"]
FLOOR = 1e-300
FULL_BATCH_N = 3833
SEEDS = (100, 101, 102, 103, 104)
CONST_SIZES = (64, 256, 1024, 3833)
FVC_SIZES = (16, 64, 256)

STOP_ACCEPTANCE, STOP_SNR, STOP_MIN_GEN = 0.01, 1.0, 5

# ---- colours: panel B's ladders are the palette for all three figures ----------------------
CONST_COLORS = OrderedDict(zip(CONST_SIZES, paper_style.sweep_colors(paper_style.CONSTANT_CMAP, 4)))
FVC_COLORS = OrderedDict(zip(FVC_SIZES, paper_style.sweep_colors(paper_style.ADAPTIVE_CMAP, 3)))
FVC_MAIN = 64                       # the adaptive arm drawn as curves and in panel C
# Wherever one adaptive arm and the full batch are compared as curves they carry the paper-wide
# adaptive blue and full-batch red, whatever their rung on panel B's ladders.
FULL_COLOR = paper_style.FULL_BATCH_COLOR
FVC_COLOR = paper_style.FVC_COLOR

WANG_COLOR = paper_style.ACCENT['purple']
KILIC_COLOR = '0.30'
REF_COLOR = paper_style.ACCENT['green']     # the likelihood reference and its mean
REF_FILL = '#BFE0CC'
PRIOR_FILL = '0.93'

STATE_COLORS = (paper_style.ACCENT['blue'], paper_style.ACCENT['orange'])   # promoter OFF, ON
TRAJ_OBS = paper_style.ACCENT['purple']   # the fixation-and-count event, and the cells per time
TRAJ_STEM = TRAJ_OBS
TRAJ_PARAMS = dict(K_12=0.00214551, K_21=0.00976828, B_1=0.00056906, B_2=0.09734202)   # near the likelihood reference
TRAJ_T_MAX = 1200.0
# one simulated cell per drawn fixation time (seeds chosen so the traces separate visually)
TRAJ_SAMPLES = ((120.0, 3), (240.0, 11), (600.0, 8), (1200.0, 16))
GAMMA_FIXED = 0.00533


# =============================================================================================
# Reference posterior and W1 scoring
# =============================================================================================
def load_reference(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    X = np.asarray(raw["samples_log10"], dtype=float)
    flat = X.reshape(-1, X.shape[-1])
    return dict(samples=flat, map_log10=np.asarray(raw["map_log10"], dtype=float),
                path=path, n_draws=int(flat.shape[0]))


def norm_weights(weights, n):
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n or not np.isfinite(w).all() or w.sum() <= 0:
        return np.full(n, 1.0 / n)
    return w / w.sum()


def to_log10(posterior):
    return np.log10(np.clip(np.asarray(posterior, dtype=float), FLOOR, None))


def mean_marginal_w1(posterior, weights, ref):
    """Mean over the four rates of the marginal W1 (decades) to the reference posterior."""
    P = to_log10(posterior)
    w = norm_weights(weights, P.shape[0])
    R = ref["samples"]
    return float(np.mean([wasserstein_distance(P[:, j], R[:, j], u_weights=w)
                          for j in range(P.shape[1])]))


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    xs, ws = x[order], np.asarray(w, dtype=float)[order]
    cdf = np.cumsum(ws) - 0.5 * ws
    cdf /= ws.sum()
    return np.interp(np.atleast_1d(q), cdf, xs)


def weighted_sd(x, w):
    w = np.asarray(w, dtype=float) / np.sum(w)
    m = float(w @ x)
    return float(np.sqrt(max(w @ (x - m) ** 2, 0.0)))


def kde(x, w, grid, bw, block=4000):
    """Weighted Gaussian KDE at a fixed bandwidth."""
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    w = w / w.sum()
    out = np.zeros(grid.size)
    for a in range(0, x.size, block):
        xb, wb = x[a:a + block], w[a:a + block]
        d = (grid[None, :] - xb[:, None]) / bw
        out += wb @ np.exp(-0.5 * d ** 2)
    return out / (bw * np.sqrt(2.0 * np.pi))


# =============================================================================================
# Kilic et al. (2023) posterior means
# =============================================================================================
def kilic_fig4_means(path, burn_fraction=0.3):
    """
    Posterior means of the MCMC chain distributed as Source Data for Kilic et al. (2023) Fig. 4
    (`Figure4/panel_c/results/{rates,loads}.csv`; see `Figure4/readme.txt` in the archive).
    The chain samples up to six promoter states with the degradation rate free. As in the
    authors' `visualize_results.m`, the first 30% of iterations are discarded; the remaining
    iterations with exactly two active states are averaged, taking the OFF state as the one
    with the lower production rate. Returns (K_12, K_21, B_1, B_2) in s^-1.
    """
    with zipfile.ZipFile(path) as z:
        rates = np.loadtxt(io.BytesIO(z.read("Figure4/panel_c/results/rates.csv")), delimiter=',')
        loads = np.loadtxt(io.BytesIO(z.read("Figure4/panel_c/results/loads.csv")), delimiter=',')
    L, n = loads.shape
    pairs = list(combinations(range(L), 2))
    idx = pairs + [(b, a) for a, b in pairs]            # row order of the transition rates
    beta = rates[len(idx):len(idx) + L]
    active = loads.astype(bool)
    keep = (active.sum(0) == 2) & (np.arange(n) >= int(burn_fraction * n))
    out = []
    for j in np.where(keep)[0]:
        a, b = np.where(active[:, j])[0]
        if beta[a, j] > beta[b, j]:
            a, b = b, a
        out.append([rates[idx.index((a, b)), j], rates[idx.index((b, a)), j],
                    beta[a, j], beta[b, j]])
    return tuple(float(v) for v in np.mean(out, axis=0))


# =============================================================================================
# The fits
# =============================================================================================
def run_path(runs_dir, scheme, size, seed):
    return os.path.join(runs_dir, "ecoli", "ecoli_%s_n%04d_s%d.pkl" % (scheme, size, seed))


def load_run_rows(path, ref):
    """One row per generation within the wall-time limit, with the accuracy already scored."""
    with open(path, "rb") as f:
        gens, _post, _w, meta = pickle.load(f)
    tlim = meta.get("time_limit", None)
    tlim = float(tlim) if tlim is not None else np.inf
    rows = []
    for g in sorted(gens):
        r = gens[g]
        tot = float(r["total_time"])
        if tot > tlim:
            continue
        rows.append(dict(
            generation=int(g), total_time=tot, cells=float(r["sim_cells"]),
            alpha_threshold=float(r["alpha_threshold"]), acceptance=float(r["acceptance"]),
            snr=float(r["snr"]),
            w1=mean_marginal_w1(r["posterior"], r["weights"], ref),
            posterior=np.asarray(r["posterior"], dtype=float),
            weights=np.asarray(r["weights"], dtype=float)))
    return rows, meta


def stop_index(rows):
    """Index of the generation where the stopping rule fires, or None."""
    for i, r in enumerate(rows):
        if r["generation"] <= STOP_MIN_GEN:
            continue
        if r["acceptance"] < STOP_ACCEPTANCE or r["snr"] < STOP_SNR:
            return i
    return None


def arm_specs():
    """Every arm, in ladder order: fixed batches first, then the adaptive sweep."""
    out = []
    for n in CONST_SIZES:
        full = n >= FULL_BATCH_N
        out.append(dict(key=('const', n), kind='constant', scheme='constant', size=n,
                        color=CONST_COLORS[n],
                        label=(r"Full batch $n = N = %d$" % n) if full
                              else (r"Fixed $n = %d$" % n),
                        legend=(r"$%d = N$" % n) if full else (r"$%d$" % n)))
    for n in FVC_SIZES:
        out.append(dict(key=('fvc', n), kind='fvc', scheme='fvc', size=n, color=FVC_COLORS[n],
                        label=paper_style.ADAPTIVE_NAME + r" ($n_0 = %d$)" % n,
                        legend=r"$%d$" % n))
    return out


def load_fits(runs_dir, ref, seeds=SEEDS):
    """{arm key: dict(spec, runs=[rows...], stops=[row...])}, everything scored once."""
    camp = OrderedDict()
    for spec in arm_specs():
        runs, stops, metas = [], [], []
        for s in seeds:
            p = run_path(runs_dir, spec['scheme'], spec['size'], s)
            if not os.path.exists(p):
                print("  missing: %s" % os.path.basename(p))
                continue
            rows, meta = load_run_rows(p, ref)
            i = stop_index(rows)
            if i is None:
                print("  no stop: %s" % os.path.basename(p))
                continue
            runs.append(rows[:i + 1])          # every curve ends at its own stop
            stops.append(rows[i])
            metas.append(meta)
        rec = dict(spec)
        rec.update(runs=runs, stops=stops, metas=metas, n_fired=len(stops))
        camp[spec['key']] = rec
        if stops:
            print("  %-24s n=%d  gen %d  wall %.0f s  cells %.3e  W1 %.4f"
                  % (spec['label'].replace('$', ''), len(stops),
                     int(np.median([r['generation'] for r in stops])),
                     np.median([r['total_time'] for r in stops]),
                     np.median([r['cells'] for r in stops]),
                     np.median([r['w1'] for r in stops])))
    return camp


def med_range(stops, field):
    v = np.array([r[field] for r in stops], dtype=float)
    return float(np.median(v)), float(v.min()), float(v.max())


def band_curve(runs, xkey, ykey, min_seeds=None):
    """
    The arm's median curve over its seeds, one point per generation: x is the median cost of
    that generation over the seeds that reached it, y the median value, and the band their
    min-max. Generations reached by fewer than `min_seeds` runs (half of them, rounded up, by
    default) are dropped, so the curve ends where the median run ends and the marker at the
    stopping generation sits on it.
    """
    if min_seeds is None:
        min_seeds = int(np.ceil(len(runs) / 2.0))
    by_gen = {}
    for rs in runs:
        for r in rs:
            by_gen.setdefault(int(r['generation']), []).append(r)
    gens = sorted(g for g, v in by_gen.items() if len(v) >= min_seeds)
    if len(gens) < 2:
        return None, None, None, None
    gx = np.array([np.median([r[xkey] for r in by_gen[g]]) for g in gens])
    stack = [np.array([float(r[ykey]) for r in by_gen[g]]) for g in gens]
    med = np.array([np.median(v) for v in stack])
    lo = np.array([v.min() for v in stack])
    hi = np.array([v.max() for v in stack])
    return gx, med, lo, hi


def pooled_cloud(stops):
    """The stopping-generation particles of every seed of one arm, each seed with total weight 1/n."""
    P, W = [], []
    for r in stops:
        w = norm_weights(r["weights"], r["posterior"].shape[0])
        P.append(to_log10(r["posterior"]))
        W.append(w)
    return np.vstack(P), np.concatenate(W) / len(P)


# =============================================================================================
# Panel A: the observation model (telegraph model, exact simulation, fixed seeds)
# =============================================================================================
def simulate_trajectory(params, gamma, t_max, seed):
    rng = np.random.default_rng(int(seed))
    beta = (float(params["B_1"]), float(params["B_2"]))
    ksw = (float(params["K_12"]), float(params["K_21"]))
    t, m, s = 0.0, 0, 0
    ts, ms, ss = [0.0], [0], [0]
    while True:
        r_sw, r_pr, r_dg = ksw[s], beta[s], float(gamma) * m
        total = r_sw + r_pr + r_dg
        if total <= 0.0:
            break
        t = t + rng.exponential(1.0 / total)
        if t >= t_max:
            break
        u = rng.random() * total
        if u < r_sw:
            s = 1 - s
        elif u < r_sw + r_pr:
            m += 1
        else:
            m -= 1
        ts.append(t); ms.append(m); ss.append(s)
    ts.append(float(t_max)); ms.append(m); ss.append(s)
    return np.asarray(ts), np.asarray(ms, dtype=float), np.asarray(ss, dtype=float)


def panel_A(axes, meta, gamma=GAMMA_FIXED, samples=TRAJ_SAMPLES):
    """
    One small panel per simulated cell (top rows): each cell runs only until its own fixation
    time and is counted there; the trace is coloured by the promoter state. Bottom row: how the
    dataset is spread over its fixation times (cells per time).
    """
    ax_st = axes[-1]
    ax_trs = axes[:-1]
    drawn = []
    for t_end, seed in samples[:len(ax_trs)]:
        ts, ms, ss = simulate_trajectory(TRAJ_PARAMS, gamma, float(t_end), seed)
        drawn.append((float(t_end), ts, ms, ss))
    top = max(4.0, max(float(d[2].max()) for d in drawn)) * 1.02

    for ax, (t_end, ts, ms, ss) in zip(ax_trs, drawn):
        ax.axhline(0.0, color='0.80', lw=0.5, zorder=1)
        segs, cols = [], []
        for i in range(ts.size - 1):
            c = STATE_COLORS[int(ss[i] >= 0.5)]
            segs.append([(ts[i], ms[i]), (ts[i + 1], ms[i])])
            segs.append([(ts[i + 1], ms[i]), (ts[i + 1], ms[i + 1])])
            cols += [c, c]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.9, zorder=4))
        ax.plot([t_end], [ms[-1]], ls='none', marker='o', ms=5.5, mfc=TRAJ_OBS,
                mec='white', mew=0.6, color=TRAJ_OBS, zorder=6)
        ax.set_ylim(-0.06 * top, top * 1.15)
        ax.set_yticks([0, int(round(top / 10.0)) * 5 * 2 if top >= 10 else 5])
        ax.tick_params(axis='x', labelbottom=False)
    ax_trs[len(ax_trs) // 2].set_ylabel("Mature mRNA", labelpad=3)

    obs_t = np.asarray(meta['obs_t'], dtype=float)
    n_cells = np.asarray(meta['n_cells'], dtype=float)
    ax_st.bar(obs_t, n_cells, width=13.0, color=TRAJ_STEM, lw=0, zorder=3)
    ax_st.set_ylim(0.0, 1.4 * n_cells.max())
    ax_st.set_yticks([0, 200, 400])
    ax_st.set_ylabel("Cells", labelpad=3)
    ax_st.text(0.98, 0.9, r"$N = %d$ cells, %d fixation times"
               % (int(n_cells.sum()), obs_t.size),
               transform=ax_st.transAxes, ha='right', va='top', fontsize=paper_style.FS_LEGEND,
               color='0.3')
    ax_st.set_xlabel("Time since induction (s)", labelpad=1.5)

    for ax in axes:
        ax.set_xlim(-30.0, TRAJ_T_MAX + 45.0)
        ax.set_xticks([0, 300, 600, 900, 1200])
        paper_style.style_axes(ax)
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    handles = [Line2D([], [], color=STATE_COLORS[0], lw=0.9, label="promoter OFF"),
               Line2D([], [], color=STATE_COLORS[1], lw=0.9, label="promoter ON"),
               Line2D([], [], color=TRAJ_OBS, ls='none', marker='o', ms=5.0, mfc=TRAJ_OBS,
                      mec='white', mew=0.6, label="cell fixed and counted")]
    return handles


# =============================================================================================
# Panel B: accuracy against wall time at the stop
# =============================================================================================
def nondominated(x, y):
    """The fixed-batch points that nothing cheaper already matches on accuracy, left to right."""
    pts = sorted(zip(np.asarray(x, float), np.asarray(y, float)))
    keep, best = [], np.inf
    for xi, yi in pts:
        if yi < best:
            keep.append((xi, yi))
            best = yi
    return keep


PARETO_MS = 4.0


def panel_B(ax, camp):
    rows = [r for r in camp.values() if r['n_fired']]
    const = [r for r in rows if r['kind'] == 'constant']
    fvc = [r for r in rows if r['kind'] == 'fvc']

    xs, ys = [], []
    for r in rows:
        mx, lox, hix = med_range(r['stops'], 'total_time')
        my, loy, hiy = med_range(r['stops'], 'w1')
        r['_xy'] = (mx, my)
        xs += [lox, hix]
        ys += [loy, hiy]
        # squares: fixed batches, circles: adaptive
        r['_h'] = ax.errorbar([mx], [my],
                              xerr=np.array([[mx - lox], [hix - mx]]),
                              yerr=np.array([[my - loy], [hiy - my]]),
                              color=r['color'], ecolor=r['color'],
                              elinewidth=paper_style.ERR_LW, capsize=paper_style.ERR_CAP,
                              capthick=paper_style.ERR_LW,
                              marker='o' if r['kind'] == 'fvc' else 's',
                              markersize=PARETO_MS, markerfacecolor=r['color'],
                              markeredgecolor=r['color'], linestyle='none',
                              zorder=6 if r['kind'] == 'fvc' else 5)

    ax.set_xlim(0.0, 1.03 * max(xs))
    ax.set_ylim(0.0, 1.08 * max(ys))
    ax.set_xlabel(paper_style.X_LABEL_WALL, labelpad=2)
    ax.set_ylabel("Mean marginal $W_1$ to the\nfull-likelihood posterior (decades)", labelpad=3)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    paper_style.style_axes(ax)

    const.sort(key=lambda r: r['size'])
    pts = nondominated([r['_xy'][0] for r in const], [r['_xy'][1] for r in const])
    paper_style.suboptimal_wedge(ax, [p[0] for p in pts], [p[1] for p in pts], label=True)

    fvc.sort(key=lambda r: r['size'])
    leg_kw = dict(frameon=True, fancybox=False, labelspacing=0.3, borderpad=0.45,
                  handletextpad=0.35, handlelength=1.0, framealpha=0.92)
    leg1 = ax.legend([r['_h'] for r in const], [r['legend'] for r in const],
                     title="Constant\n$n_t$", loc='upper right',
                     bbox_to_anchor=(0.755, 0.995), **leg_kw)
    leg1.set_zorder(20)
    ax.add_artist(leg1)
    leg2 = ax.legend([r['_h'] for r in fvc], [r['legend'] for r in fvc],
                     title="Adaptive\n$n_0$", loc='upper right',
                     bbox_to_anchor=(0.999, 0.995), **leg_kw)
    leg2.set_zorder(20)
    for lg in (leg1, leg2):
        lg.get_frame().set_edgecolor('0.75')
        lg.get_frame().set_linewidth(0.6)


# =============================================================================================
# Panel C: the marginal posteriors
# =============================================================================================
BW_FACTOR = 0.32             # bandwidth of the likelihood reference, times its sd
BW_FACTOR_ABC = 0.55         # the ABC clouds are 5,000 weighted particles: smoothed more
KDE_GRID = 512
PRIOR_MU, PRIOR_SD = -1.0, 0.5
PRIOR_HEIGHT = 0.17          # the rescaled prior's height, as a fraction of the panel's peak
MARK_LEVEL = 0.93            # reference marks sit on one row above the densities
MARK_MS = 6.0


def reference_marks(meta, kilic_means, fvc_mean, ref_mean):
    """The four reference points, as rates (s^-1)."""
    wang = [float(v) for v in np.asarray(meta['wang2019'], dtype=float)[:4]]
    return OrderedDict([
        ('wang', dict(v=wang, color=WANG_COLOR, marker='v', label="Wang et al. 2019")),
        ('kilic', dict(v=list(kilic_means), color=KILIC_COLOR, marker='D',
                       label="Kilic et al. 2023")),
        ('ref', dict(v=[float(v) for v in ref_mean], color=REF_COLOR, marker='*',
                     label="Full-likelihood posterior mean")),
        ('fvc', dict(v=[float(v) for v in fvc_mean], color=FVC_COLOR, marker='o',
                     label=paper_style.ADAPTIVE_NAME + " posterior mean")),
    ])


def _unit_exponent(hi):
    """Power of ten used to scale one panel's rate axis, e.g. -3 for rates around 1e-3."""
    return int(np.floor(np.log10(hi)))


def panel_C(axes, camp, ref, marks, pad=0.05):
    """
    Marginals on a linear rate scale. The clouds and the reference are stored in log10 and
    mapped back to rates here; each panel is scaled by a power of ten (given in the axis label)
    so that the tick labels stay short.
    """
    fvc = camp[('fvc', FVC_MAIN)]
    full = camp[('const', FULL_BATCH_N)]
    Pf, Wf = pooled_cloud(fvc['stops'])
    Pb, Wb = pooled_cloud(full['stops'])
    Pf, Pb, R = 10.0 ** Pf, 10.0 ** Pb, 10.0 ** ref['samples']
    windows = []
    for j in range(4):
        lo = [weighted_quantile(Pf[:, j], Wf, 0.005)[0], weighted_quantile(Pb[:, j], Wb, 0.005)[0],
              float(np.quantile(R[:, j], 0.001))]
        hi = [weighted_quantile(Pf[:, j], Wf, 0.995)[0], weighted_quantile(Pb[:, j], Wb, 0.995)[0],
              float(np.quantile(R[:, j], 0.999))]
        for m in marks.values():
            lo.append(m['v'][j])
            hi.append(m['v'][j])
        a, b = min(lo), max(hi)
        p = pad * (b - a)
        windows.append((max(a - p, 0.0), b + p))

    handles = None
    for j, ax in enumerate(axes):
        z0, z1 = windows[j]
        e = _unit_exponent(z1)
        u = 10.0 ** e
        grid = np.linspace(z0, z1, KDE_GRID)
        sd_ref = float(R[:, j].std(ddof=1))
        sd_abc = min(weighted_sd(Pf[:, j], Wf), weighted_sd(Pb[:, j], Wb))
        dref = kde(R[:, j], np.ones(R.shape[0]), grid, BW_FACTOR * sd_ref)
        bw_abc = max(BW_FACTOR_ABC * sd_abc, BW_FACTOR * sd_ref)
        dfvc = kde(Pf[:, j], Wf, grid, bw_abc)
        dful = kde(Pb[:, j], Wb, grid, bw_abc)
        ymax = max(dref.max(), dfvc.max(), dful.max())

        # the prior (log-normal in the rate) is nearly flat over this window, so it is rescaled
        # to a fixed fraction of the panel
        g = np.clip(grid, 1e-300, None)
        dpri = np.exp(-0.5 * ((np.log10(g) - PRIOR_MU) / PRIOR_SD) ** 2) / g
        dpri = dpri * (PRIOR_HEIGHT * ymax / max(dpri.max(), 1e-300))

        gs = grid / u
        h_pri = ax.fill_between(gs, 0.0, dpri, facecolor=PRIOR_FILL, edgecolor='none',
                                zorder=0)
        h_ref = ax.fill_between(gs, 0.0, dref, facecolor=REF_FILL, edgecolor='none',
                                zorder=1)
        h_ful, = ax.plot(gs, dful, color=FULL_COLOR, lw=paper_style.LW_DENSITY, zorder=5)
        h_fvc, = ax.plot(gs, dfvc, color=FVC_COLOR, lw=paper_style.LW_DENSITY, zorder=6)

        ax.set_xlim(z0 / u, z1 / u)
        ax.set_ylim(0.0, 1.28 * ymax)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        # reference marks: one row above the densities, each with a thin guide line down
        # through the densities
        for key, m in marks.items():
            v = m['v'][j] / u
            if not (z0 / u <= v <= z1 / u):
                continue
            ax.axvline(v, color=m['color'], lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=1.5)
            ax.plot([v], [MARK_LEVEL], transform=tr, marker=m['marker'], ls='none',
                    ms=MARK_MS + (2.0 if m['marker'] == '*' else 0.0), color=m['color'],
                    mfc=m['color'], mec='white', mew=0.5, clip_on=False, zorder=8)

        ax.set_yticks([])
        ax.set_xlabel(PARAM_TEX[j] % e, labelpad=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune='both'))
        paper_style.style_axes(ax)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        if j == 0:
            ax.set_ylabel("Density", labelpad=4)
        if handles is None:
            handles = [(h_pri, "Prior (rescaled)"),
                       (h_ref, "Full likelihood"),
                       (h_fvc, paper_style.ADAPTIVE_NAME + r" ($n_0 = %d$)" % FVC_MAIN),
                       (h_ful, r"Full batch $n = N = %d$" % FULL_BATCH_N)]
    mark_handles = [Line2D([], [], color=m['color'], marker=m['marker'], ls='none',
                           ms=MARK_MS + (2.0 if m['marker'] == '*' else 0.0), mfc=m['color'],
                           mec='white', mew=0.5, label=m['label']) for m in marks.values()]
    return handles, mark_handles, windows


# =============================================================================================
# ecoli_main
# =============================================================================================
# Layout in inches from the bottom of the figure: two-row marker legend, panel C, one-row curve
# legend, one-row promoter legend, panels A and B.
MAIN_MARK_Y = 0.22            # centre of the reference-mark legend rows
MAIN_C_BOT = 0.82             # panel C axes
MAIN_C_H = 1.30
MAIN_ROW = 0.22               # pitch between the two legend rows above panel C
MAIN_AB_GAP = 0.45            # promoter legend row centre -> bottom of panels A/B
MAIN_AB_H = 3.09              # panels A/B
MAIN_TOP = 0.19               # room for the A/B letters
MAIN_H = (MAIN_C_BOT + MAIN_C_H + 0.15 + MAIN_ROW + MAIN_AB_GAP + MAIN_AB_H + MAIN_TOP)


def figure_main(camp, ref, kilic_means, out_stem):
    H = MAIN_H
    fy = lambda inch: inch / H                 # inches from the bottom -> figure fraction
    c_bot, c_top = MAIN_C_BOT, MAIN_C_BOT + MAIN_C_H
    curve_y = c_top + 0.15                     # panel C's curve legend row
    prom_y = curve_y + MAIN_ROW                # panel A's promoter legend row
    fig = plt.figure(figsize=(paper_style.TEXT_WIDTH, H))
    n_tr = 4                                   # simulated cells drawn in panel A
    gs_top = fig.add_gridspec(n_tr + 1, 2, left=0.085, right=0.985, top=fy(H - MAIN_TOP),
                              bottom=fy(prom_y + MAIN_AB_GAP),
                              width_ratios=[1.0, 1.35], height_ratios=[1.0] * n_tr + [1.6],
                              hspace=0.14, wspace=0.31)
    axesA = [fig.add_subplot(gs_top[0, 0])]
    axesA += [fig.add_subplot(gs_top[i, 0], sharex=axesA[0]) for i in range(1, n_tr + 1)]
    axB = fig.add_subplot(gs_top[:, 1])

    hA = panel_A(axesA, camp[('fvc', FVC_MAIN)]['metas'][0])
    panel_B(axB, camp)
    paper_style.panel_label(axesA[0], "A", dx=-0.21, dy=1.02)
    paper_style.panel_label(axB, "B", dx=-0.155, dy=1.005)

    fig.legend(handles=hA, loc='center left', bbox_to_anchor=(0.005, fy(prom_y)), ncol=3,
               frameon=False, fontsize=paper_style.FS_LEGEND, handlelength=1.5, handletextpad=0.5,
               columnspacing=1.2, borderaxespad=0.0)

    stops = camp[('fvc', FVC_MAIN)]['stops']
    fvc_mean = np.average(np.vstack([np.asarray(r['posterior'], dtype=float) for r in stops]),
                          axis=0,
                          weights=np.concatenate([norm_weights(r['weights'], 1000) / 5
                                                  for r in stops]))
    ref_mean = (10.0 ** ref['samples']).mean(axis=0)
    marks = reference_marks(camp[('fvc', FVC_MAIN)]['metas'][0], kilic_means, fvc_mean, ref_mean)

    gs_bot = fig.add_gridspec(1, 4, left=0.075, right=0.985, top=fy(c_top), bottom=fy(c_bot),
                              wspace=0.24)
    axesC = [fig.add_subplot(gs_bot[0, j]) for j in range(4)]
    hC, hMarks, windows = panel_C(axesC, camp, ref, marks)
    # the letter sits at the left end of panel C's curve-legend row
    paper_style.panel_label(axesC[0], "C", dx=-0.21, dy=1.0 + 0.07 / MAIN_C_H)

    fig.legend([h for h, _ in hC], [t for _, t in hC], loc='center left',
               bbox_to_anchor=(0.06, fy(curve_y)), ncol=4, frameon=False,
               fontsize=paper_style.FS_LEGEND, handlelength=1.4, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0.0)
    fig.legend(handles=hMarks, loc='center', bbox_to_anchor=(0.5, fy(MAIN_MARK_Y)), ncol=2,
               frameon=False, fontsize=paper_style.FS_LEGEND, handlelength=0.6,
               handletextpad=0.3, columnspacing=0.6, borderaxespad=0.0)

    paper_style.save(fig, out_stem)
    plt.close(fig)
    return windows


# =============================================================================================
# ecoli_w1_vs_cost
# =============================================================================================
def _two_lines(label):
    """Break a shared 'Name, $symbol$' label after its first word."""
    head, _, tail = label.partition(' ')
    return head + '\n' + tail


W1_H = 6.5
W1_PANELS = (('w1', "Mean marginal $W_1$ to the\nfull-likelihood posterior (decades)", None),
             ('alpha_threshold', _two_lines(paper_style.Y_LABEL_TOL), None),
             ('acceptance', _two_lines(paper_style.Y_LABEL_ACC), STOP_ACCEPTANCE),
             ('snr', r"Signal-to-noise" + "\n" + r"ratio, $\mathrm{SNR}_t$", STOP_SNR))
CURVE_ARMS = (('const', 64), ('const', 256), ('const', 1024), ('const', FULL_BATCH_N),
              ('fvc', FVC_MAIN))


def curve_color(rec):
    """Fixed sizes keep panel B's ladder colours; the full batch and the adaptive arm the paper-wide ones."""
    if rec['kind'] == 'fvc':
        return FVC_COLOR
    return FULL_COLOR if rec['size'] >= FULL_BATCH_N else rec['color']


def figure_w1(camp, out_stem):
    H = W1_H
    fig = plt.figure(figsize=(paper_style.TEXT_WIDTH, H))
    gs = fig.add_gridspec(4, 1, left=0.135, right=0.985, top=1.0 - 0.16 / H, bottom=0.44 / H,
                          height_ratios=[2.2, 1.15, 1.15, 1.15], hspace=0.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    arms = [camp[k] for k in CURVE_ARMS]

    # one thin line per seed, ending at that seed's stopping generation (open circle), and a
    # thick generation-wise median over the seeds drawn to the arm's median stopping generation
    for ax, (field, ylab, dashed) in zip(axes, W1_PANELS):
        for rec in arms:
            fvc = rec['kind'] == 'fvc'
            col = curve_color(rec)
            for rows in rec['runs']:
                xs = np.array([r['cells'] for r in rows], dtype=float)
                ys = np.array([r[field] for r in rows], dtype=float)
                ax.plot(xs, ys, color=col, lw=paper_style.LW_SEED, alpha=0.65, zorder=3)
                ax.plot(xs[-1], ys[-1], marker='o', mfc='none', mec=col, mew=1.0,
                        ms=paper_style.STOP_MS, ls='none', zorder=7, clip_on=False)
            gx, med, lo, hi = band_curve(rec['runs'], 'cells', field)
            if gx is None:
                continue
            ax.plot(gx, med, color=col, lw=paper_style.LW_MEDIAN_FVC if fvc else paper_style.LW_MEDIAN,
                    zorder=6 if fvc else 4, solid_capstyle='round')
        if dashed is not None:
            ax.axhline(dashed, color='0.35', ls='--', lw=0.8, zorder=1)
            ax.text(0.012, dashed, "%g" % dashed, transform=blended_transform_factory(
                ax.transAxes, ax.transData), fontsize=paper_style.FS_LEGEND, color='0.35',
                va='bottom', ha='left')
        ax.set_xscale('log')
        if field != 'w1':                 # the W1 distance stays linear (it spans one decade)
            ax.set_yscale('log')
        ax.set_ylabel(ylab, labelpad=3, fontsize=paper_style.FS_LABEL)
        paper_style.style_axes(ax)
        # room for the stop markers at the bottom of the acceptance and SNR panels
        vals = [r[field] for rec in camp.values() for rs in rec['runs'] for r in rs]
        vals += [dashed] if dashed is not None else []
        a, b = np.log10(min(vals)), np.log10(max(vals))
        ax.set_ylim(10.0 ** (a - 0.07 * (b - a)), 10.0 ** (b + 0.07 * (b - a)))

    xlo = min(min(r[0]['cells'] for r in rec['runs']) for rec in camp.values() if rec['runs'])
    xhi = max(max(r[-1]['cells'] for r in rec['runs']) for rec in camp.values() if rec['runs'])
    for ax in axes:
        ax.set_xlim(10 ** (np.log10(xlo) - 0.08), 10 ** (np.log10(xhi) + 0.08))
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                                         numticks=30))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    for ax in axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)
    axes[-1].set_xlabel("Cumulative simulated cells", labelpad=2)

    # accuracy and tolerance span barely more than a decade each, so they get named ticks
    axes[0].set_ylim(0.0, 1.08 * max(r['w1'] for rec in camp.values() for rs in rec['runs']
                                     for r in rs))
    axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    ticks = [0.2, 0.5, 1.0, 2.0]
    axes[1].yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    axes[1].yaxis.set_major_formatter(mpl.ticker.FixedFormatter([("%g" % t) for t in ticks]))
    axes[1].yaxis.set_minor_locator(mpl.ticker.LogLocator(
        base=10.0, subs=tuple(np.arange(2, 10) / 10.0), numticks=40))
    for ax in axes:
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    handles = [Line2D([], [], color=curve_color(rec),
                      lw=paper_style.LW_MEDIAN_FVC if rec['kind'] == 'fvc' else paper_style.LW_MEDIAN,
                      label=rec['label']) for rec in arms]
    handles.append(Line2D([], [], color='0.35', marker='o', mfc='none', mew=1.0,
                          ms=paper_style.STOP_MS, ls='none', label=paper_style.STOP_LABEL))
    leg = axes[0].legend(handles=handles, loc='lower left', frameon=True, fancybox=False,
                         fontsize=paper_style.FS_LEGEND, labelspacing=0.32, borderpad=0.5,
                         handlelength=1.6, handletextpad=0.6)
    leg.get_frame().set_edgecolor('0.75')
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(20)

    for ax, letter in zip(axes, "ABCD"):
        # letters at the figure's left edge, clear of the two-line y labels
        paper_style.panel_label(ax, letter, dx=-0.152, dy=0.97 if ax is axes[0] else 0.88)
    fig.align_ylabels(axes)

    paper_style.save(fig, out_stem)
    plt.close(fig)


# =============================================================================================
# ecoli_estimates_vs_cost
# =============================================================================================
EST_H = 4.7
EST_EXPONENTS = (-3, -2, -3, -1)        # the powers of ten that scale each rate axis
EST_TOL = 0.10               # "within 10% of the likelihood value", for the printed summary
EST_REF_PANEL = 2            # panel C (B_1): a strip is opened under its band for the legend
EST_FOOT_IN = 0.40           # height of that strip, inches
EST_LEFT, EST_GAP_W = 0.72, 0.74        # inches: left margin, gap between the two columns
EST_TOP, EST_BOT, EST_GAP_H = 0.22, 0.50, 0.22


def _mean_runs(rec):
    """Per generation, the weighted posterior mean of each rate, keyed 'm0'..'m3', and the cost."""
    out = []
    for rows in rec['runs']:
        rs = []
        for r in rows:
            P = np.asarray(r['posterior'], dtype=float)
            m = norm_weights(r['weights'], P.shape[0]) @ P
            d = dict(generation=r['generation'], cells=r['cells'])
            d.update(('m%d' % j, float(m[j])) for j in range(P.shape[1]))
            rs.append(d)
        out.append(rs)
    return out


def figure_estimates(camp, ref, out_stem):
    H, W = EST_H, paper_style.TEXT_WIDTH
    R = 10.0 ** ref['samples']                            # likelihood draws as rates
    ref_val = R.mean(axis=0)
    # central 95% credible interval of the likelihood draws
    ref_lo, ref_hi = np.percentile(R, 2.5, axis=0), np.percentile(R, 97.5, axis=0)

    ax_w = (W * 0.985 - EST_LEFT - EST_GAP_W) / 2.0
    ax_h = (H - EST_TOP - EST_BOT - EST_GAP_H) / 2.0
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, left=EST_LEFT / W, right=0.985, top=1.0 - EST_TOP / H,
                          bottom=EST_BOT / H, hspace=EST_GAP_H / ax_h, wspace=EST_GAP_W / ax_w)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    arms = [camp[k] for k in CURVE_ARMS]
    sruns = {rec['key']: _mean_runs(rec) for rec in arms}

    xlo = min(min(r[0]['cells'] for r in rec['runs']) for rec in camp.values() if rec['runs'])
    xhi = max(max(r[-1]['cells'] for r in rec['runs']) for rec in camp.values() if rec['runs'])
    summary = []
    for j, ax in enumerate(axes):
        key = 'm%d' % j
        u = 10.0 ** EST_EXPONENTS[j]
        ax.axhspan(ref_lo[j] / u, ref_hi[j] / u, facecolor=REF_FILL, edgecolor='none', zorder=0)
        ax.axhline(ref_val[j] / u, color=REF_COLOR, lw=1.1, zorder=1)
        for rec in arms:
            fvc = rec['kind'] == 'fvc'
            col = curve_color(rec)
            runs = sruns[rec['key']]
            for rows in runs:
                xs = np.array([r['cells'] for r in rows], dtype=float)
                ys = np.array([r[key] for r in rows], dtype=float) / u
                ax.plot(xs, ys, color=col, lw=paper_style.LW_SEED, alpha=0.65, zorder=3)
                ax.plot(xs[-1], ys[-1], marker='o', mfc='none', mec=col, mew=1.0,
                        ms=paper_style.STOP_MS, ls='none', zorder=7, clip_on=True)
            gx, med, lo, hi = band_curve(runs, 'cells', key)
            if gx is None:
                continue
            ax.plot(gx, med / u, color=col,
                    lw=paper_style.LW_MEDIAN_FVC if fvc else paper_style.LW_MEDIAN,
                    zorder=6 if fvc else 4, solid_capstyle='round')
            # printed summary: the cost from which the median stays within the tolerance of
            # the likelihood value through the arm's last median generation (None: never)
            rel = np.abs(med - ref_val[j]) / ref_val[j]
            settle = []
            for tol in (EST_TOL, 2 * EST_TOL):
                bad = np.nonzero(rel > tol)[0]
                k = 0 if bad.size == 0 else bad[-1] + 1
                settle.append(gx[k] if k < gx.size else None)
            summary.append((PARAM_NAMES[j], rec['label'], settle, rel[-1], gx[-1]))
        # log y: the early generations sit near the prior, one to two decades above the
        # posterior, and a log axis keeps them and the convergence readable together
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10 ** (np.log10(xlo) - 0.08), 10 ** (np.log10(xhi) + 0.08))
        vals = [ref_lo[j] / u, ref_hi[j] / u]
        vals += [r[key] / u for rows in (rw for rec in arms for rw in sruns[rec['key']])
                 for r in rows]
        vals = [v for v in vals if v > 0]
        a, b = np.log10(min(vals)), np.log10(max(vals))
        top = b + 0.06 * (b - a)
        bot = a - 0.06 * (b - a)
        if j == EST_REF_PANEL:
            # room under the band for the reference legend: no estimate falls below the band
            rr = EST_FOOT_IN / ax_h
            bot = min(bot, (np.log10(ref_lo[j] / u) - rr * top) / (1.0 - rr))
        ax.set_ylim(10.0 ** bot, 10.0 ** top)
        name = PARAM_TEX[j].split(' ', 1)
        ax.set_ylabel("%s, %s\n%s" % ("Posterior mean", name[0], name[1] % EST_EXPONENTS[j]),
                      labelpad=3)
        paper_style.style_axes(ax)
        # plain-number ticks at 1-10-100, 1-3-10 or 1-2-5 depending on the span
        span = top - bot
        subs = (1.0,) if span > 2.2 else ((1.0, 3.0) if span > 1.2 else (1.0, 2.0, 5.0))
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=subs))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                                         numticks=30))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if j < 2:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel("Cumulative simulated cells", labelpad=2)

    # legends inside the panels, in the regions the median curves leave empty: the arms and the
    # stop circle in panel A above the band, the reference in the strip opened under panel C's
    # band. The full-batch entry omits N so that the column fits that region.
    arm_h = [Line2D([], [], color=curve_color(rec),
                    lw=paper_style.LW_MEDIAN_FVC if rec['kind'] == 'fvc' else paper_style.LW_MEDIAN,
                    label=(r"Full batch $n = N$" if rec['key'] == ('const', FULL_BATCH_N)
                           else rec['label'])) for rec in arms]
    arm_h.append(Line2D([], [], color='0.35', marker='o', mfc='none', mew=1.0,
                        ms=paper_style.STOP_MS, ls='none', label=paper_style.STOP_LABEL))
    ref_h = [Line2D([], [], color=REF_COLOR, lw=1.1, label="Full-likelihood posterior mean"),
             Patch(facecolor=REF_FILL, edgecolor='none',
                   label=r"Full-likelihood 95\% credible interval")]
    leg_kw = dict(frameon=False, fontsize=paper_style.FS_LEGEND, handlelength=1.1,
                  handletextpad=0.45, labelspacing=0.25, borderaxespad=0.3)
    y0, y1 = (np.log10(v) for v in axes[0].get_ylim())
    band_top = (np.log10(ref_hi[0] / 10.0 ** EST_EXPONENTS[0]) - y0) / (y1 - y0)
    leg = axes[0].legend(handles=arm_h, loc='lower left', bbox_to_anchor=(0.0, band_top),
                         **dict(leg_kw, borderaxespad=0.2))
    leg.set_zorder(20)
    leg = axes[EST_REF_PANEL].legend(handles=ref_h, loc='lower left', **leg_kw)
    leg.set_zorder(20)

    for ax, letter in zip(axes, "ABCD"):
        # above the top-left corner, over the y-label column
        paper_style.panel_label(ax, letter, dx=-0.70 / ax_w, dy=1.0 + 0.02 / ax_h)
    fig.align_ylabels([axes[0], axes[2]])
    fig.align_ylabels([axes[1], axes[3]])

    paper_style.save(fig, out_stem)
    plt.close(fig)

    print("   median posterior mean: cumulative cells from which it stays within %d%% / %d%% of "
          "the likelihood value; relative error at the last median generation"
          % (int(100 * EST_TOL), int(200 * EST_TOL)))
    fmt = lambda v: ("%.2e" % v) if v is not None else "never"
    for name, lab_, settle, rel_end, x_end in summary:
        print("   %-5s %-26s %-10s %-10s end %6.1f%% (at %.2e cells)"
              % (name, lab_.replace('$', ''), fmt(settle[0]), fmt(settle[1]),
                 100 * rel_end, x_end))


# =============================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default=DEF_RUNS, help="fit records (default: %(default)s)")
    p.add_argument('--reference', default=DEF_REFERENCE,
                   help="full-likelihood reference posterior (default: %(default)s)")
    p.add_argument('--kilic', default=DEF_KILIC,
                   help="Kilic et al. (2023) Fig. 4 Source Data archive (default: %(default)s)")
    p.add_argument('--out', default=DEF_OUT, help="output directory (default: %(default)s)")
    p.add_argument('--only', choices=('main', 'w1', 'estimates', 'all'), default='all')
    a = p.parse_args(argv)

    paper_style.apply()
    os.makedirs(a.out, exist_ok=True)
    ref = load_reference(a.reference)
    print("reference: %s (%d draws)" % (a.reference, ref['n_draws']))
    camp = load_fits(a.runs, ref)

    if a.only in ('main', 'all'):
        kilic_means = kilic_fig4_means(a.kilic)
        print("Kilic et al. (2023) Fig. 4 posterior means:", ", ".join("%.3g" % v for v in kilic_means))
        stem = os.path.join(a.out, "ecoli_main")
        figure_main(camp, ref, kilic_means, stem)
        print("wrote %s.{pdf,png}" % stem)
    if a.only in ('w1', 'all'):
        stem = os.path.join(a.out, "ecoli_w1_vs_cost")
        figure_w1(camp, stem)
        print("wrote %s.{pdf,png}" % stem)
    if a.only in ('estimates', 'all'):
        stem = os.path.join(a.out, "ecoli_estimates_vs_cost")
        figure_estimates(camp, ref, stem)
        print("wrote %s.{pdf,png}" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
