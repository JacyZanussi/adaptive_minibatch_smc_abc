"""
Loaders for the E. coli lacZ smFISH data (Wang, Zhang, Xu & Golding, Nat. Microbiol. 2019).

- `load_kilic_fig4`: mature mRNA counts at 17 fixation times, read from the Source Data of
  Kilic et al. (2023) Fig. 4 (`datasets/kilic2023_fig4/`). These are the data the paper fits.
- `load_wang`: total or nascent counts from `datasets/wang2019/data-fig2{g,f}.mat`, used by the
  validation in `likelihood_reference.py`.

Both return `(obs_t, counts)`: a float array of fixation times in seconds and a list of 1-D
integer arrays, one per time point, with the NaN padding removed. Every cell is a separate fixed
cell, so each time point has a different number of cells.
"""
import io
import os
import zipfile

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "datasets")

FIG4_ZIP = os.path.join(DATA_DIR, "kilic2023_fig4", "kilic2023_fig4_source_data.zip")
FIG4_TIMES_MEMBER = "Figure4/panel_c/data/data_times.csv"
FIG4_COUNTS_MEMBER = "Figure4/panel_c/data/rna_counts.csv"
# Cells per time point in the Fig. 4 counts, checked on loading
FIG4_JK = np.array([139, 312, 317, 389, 202, 165, 182, 198, 130, 310, 83, 306, 261, 273, 173, 232, 300])

WANG_FILES = {"slow": "data-fig2g.mat", "fast": "data-fig2f.mat"}

# Two-state estimates of Wang et al. (2019), slow growth, ordered [K_12, K_21, B_1, B_2, delta] (1/s).
# The degradation rate delta is the value the fits hold fixed.
WANG2019_SLOW = np.array([0.00533, 0.03, 0.0, 0.166, 0.00533])


def load_wang(condition="slow", n_drop_last=3, which="data_tot"):
    """
    Counts from the Wang et al. `.mat` files: `which` is "data_tot" (total mRNA per cell) or
    "data_nas" (nascent). `n_drop_last` drops trailing time points (Kilic et al. drop three).
    """
    path = os.path.join(DATA_DIR, "wang2019", WANG_FILES[condition])
    d = sio.loadmat(path, simplify_cells=True)
    ts = np.asarray(d["ts"], dtype=float).ravel()
    counts = []
    for x in d[which]:
        x = np.asarray(x, dtype=float).ravel()
        counts.append(np.round(x[~np.isnan(x)]).astype(int))
    if n_drop_last:
        ts = ts[:-n_drop_last]
        counts = counts[:-n_drop_last]
    return ts, counts


def load_kilic_fig4(zip_path=None, n_drop_last=0, t_max=None, check=True):
    """
    Mature mRNA counts of the Fig. 4 Source Data of Kilic et al. (2023), all 17 times 0-1200 s.

    Parameters
    ----------
    zip_path : str or None      the Source Data archive (default: the copy in `datasets/`)
    n_drop_last : int           drop this many trailing time points
    t_max : float or None       keep only time points with t <= t_max
    check : bool                verify the number of cells per time point against `FIG4_JK`
    """
    zip_path = FIG4_ZIP if zip_path is None else zip_path
    with zipfile.ZipFile(zip_path) as z:
        ts = np.loadtxt(io.BytesIO(z.read(FIG4_TIMES_MEMBER)), delimiter=",")
        table = np.loadtxt(io.BytesIO(z.read(FIG4_COUNTS_MEMBER)), delimiter=",")
    ts = np.asarray(ts, dtype=float).ravel()
    table = np.atleast_2d(np.asarray(table, dtype=float))
    if table.shape[1] != ts.size:
        raise ValueError(f"{FIG4_COUNTS_MEMBER}: {table.shape[1]} columns for {ts.size} times")
    counts = []
    for j in range(ts.size):
        x = table[:, j]
        counts.append(np.round(x[~np.isnan(x)]).astype(int))
    if check:
        got = np.array([c.size for c in counts])
        if got.size != FIG4_JK.size or not np.array_equal(got, FIG4_JK):
            raise ValueError(f"cells per time point {got.tolist()} != expected {FIG4_JK.tolist()}")
    if n_drop_last:
        ts = ts[:-n_drop_last]
        counts = counts[:-n_drop_last]
    if t_max is not None:
        keep = ts <= float(t_max)
        ts = ts[keep]
        counts = [c for c, k in zip(counts, keep) if k]
    return ts, counts


if __name__ == "__main__":
    ts, counts = load_kilic_fig4()
    print("Kilic et al. Fig. 4, mature counts")
    print("  t    :", ts.astype(int))
    print("  cells:", np.array([c.size for c in counts]))
    print("  mean :", np.round([c.mean() for c in counts], 3))
    for cond in WANG_FILES:
        ts, counts = load_wang(cond)
        print(f"Wang et al. {cond} growth, total counts")
        print("  t    :", ts.astype(int))
        print("  cells:", np.array([c.size for c in counts]))
