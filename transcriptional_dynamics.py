"""
njit-compatible stochastic transcriptional dynamics simulator.

Key changes from the class-based version:
- All logic is in standalone @njit functions (njit doesn't support classes)
- bndry_func must be a @njit function; it receives `lengths` as an explicit
  argument instead of capturing it from an outer scope (closures aren't
  supported by njit)
- Variable-length per-snapshot output uses numba.typed.List
- np.bincount is replaced by a manual loop (not in njit)
- np.repeat for 2D arrays is replaced by a manual loop (njit only supports 1D repeat)
- np.random.geometric / exponential / uniform / normal / poisson are all
  supported natively in njit
"""

import numpy as np
import numba as nb
from numba import njit
from numba.typed import List


# ---------------------------------------------------------------------------
# Boundary function — must be @njit so it can be passed to other @njit funcs.
# `lengths` is passed explicitly because njit functions can't close over
# regular Python/numpy arrays.
# ---------------------------------------------------------------------------

@njit
def bndry_func_njit(x, ind, lengths):
    """
    Return boolean array: True where particle is inside its nucleus.
    x      : 1-D float array of positions
    ind    : 1-D int array — snapshot index for each particle
    lengths: 1-D float array — nucleus length for each snapshot
    """
    out = np.empty(x.shape[0], dtype=nb.boolean)
    for i in range(x.shape[0]):
        out[i] = (x[i] > 0.0) and (x[i] < lengths[ind[i]])
    return out


# ---------------------------------------------------------------------------
# Population dynamics
# ---------------------------------------------------------------------------

@njit
def population_dynamics(z, kplus, kminus, rburst, T):
    """
    Simulate birth/death for a bursty expression model.

    Returns
    -------
    Tbirth_sd      : birth times for particles that survived degradation
    Nbirths_sd     : per-snapshot count of surviving particles (length = sample_size)
    class_index_sd : snapshot label for each surviving particle
    """
    sample_size = z.shape[0]

    # --- birth events per snapshot (Poisson) ---
    Nbirthevents = np.random.poisson(kplus * T, sample_size)   # shape (sample_size,)
    total_events = int(np.sum(Nbirthevents))

    # --- burst sizes (Geometric) for each event ---
    Nbirths_per_event = np.empty(total_events, dtype=np.int64)
    for i in range(total_events):
        Nbirths_per_event[i] = np.random.geometric(1.0 / rburst)

    # Build per-event snapshot label and expand to per-birth
    # (replaces np.repeat + np.arange, which njit handles only for 1-D)
    total_births = int(np.sum(Nbirths_per_event))

    Tbirthevents     = np.empty(total_events,  dtype=np.float64)
    event_class      = np.empty(total_events,  dtype=np.int64)
    idx = 0
    for snap in range(sample_size):
        for _ in range(Nbirthevents[snap]):
            Tbirthevents[idx] = np.random.uniform(0.0, T)
            event_class[idx]  = snap
            idx += 1

    # Expand event arrays to per-birth arrays
    Tbirth       = np.empty(total_births, dtype=np.float64)
    class_index  = np.empty(total_births, dtype=np.int64)
    b = 0
    for e in range(total_events):
        nb_e = Nbirths_per_event[e]
        for _ in range(nb_e):
            Tbirth[b]      = Tbirthevents[e]
            class_index[b] = event_class[e]
            b += 1

    # --- degradation ---
    survived_count = 0
    survived_mask  = np.empty(total_births, dtype=nb.boolean)
    for i in range(total_births):
        tdecay = np.random.exponential(1.0 / kminus)
        survived_mask[i] = (Tbirth[i] + tdecay) > T
        if survived_mask[i]:
            survived_count += 1

    # Collect survivors
    Tbirth_sd      = np.empty(survived_count, dtype=np.float64)
    class_index_sd = np.empty(survived_count, dtype=np.int64)
    s = 0
    for i in range(total_births):
        if survived_mask[i]:
            Tbirth_sd[s]      = Tbirth[i]
            class_index_sd[s] = class_index[i]
            s += 1

    # Per-snapshot surviving birth counts (replaces np.bincount)
    Nbirths_sd = np.zeros(sample_size, dtype=np.int64)
    for i in range(survived_count):
        Nbirths_sd[class_index_sd[i]] += 1

    return Tbirth_sd, Nbirths_sd, class_index_sd


# ---------------------------------------------------------------------------
# Spatial dynamics
# ---------------------------------------------------------------------------

@njit
def spatial_dynamics(Tbirth_sd, Nbirths_sd, class_index_sd,
                     D, T, dt, z, lengths):
    """
    Diffuse surviving particles and apply boundary conditions.

    Returns a numba.typed.List of 1-D float arrays (one per snapshot).
    `lengths` is passed explicitly for use inside the @njit boundary check.
    """
    sample_size = z.shape[0]

    # Return empty typed list if no particles survived
    result = List()
    for _ in range(sample_size):
        result.append(np.empty(0, dtype=np.float64))

    sz_sd = Tbirth_sd.shape[0]
    if sz_sd == 0:
        return result

    # Initialise positions at transcription-site locations
    # (replaces np.repeat on 2-D array — z is 1-D here)
    pos  = np.empty(sz_sd, dtype=np.float64)
    b = 0
    for snap in range(sample_size):
        for _ in range(Nbirths_sd[snap]):
            pos[b] = z[snap]
            b += 1

    time = Tbirth_sd.copy()
    survived_boundary = bndry_func_njit(pos, class_index_sd, lengths)

    v = np.sqrt(2.0 * D * dt)
    n_steps = int(np.ceil((T - np.min(time)) / dt))

    # active[i] == True while particle i is still being propagated
    active = np.empty(sz_sd, dtype=nb.boolean)
    for i in range(sz_sd):
        active[i] = survived_boundary[i] and (time[i] < T)

    for _ in range(n_steps):
        for i in range(sz_sd):
            if not active[i]:
                continue
            time[i] += dt
            pos[i]  += v * np.random.normal(0.0, 1.0)
            survived_boundary[i] = (pos[i] > 0.0) and (pos[i] < lengths[class_index_sd[i]])
            active[i] = survived_boundary[i] and (time[i] < T)

    # Collect final positions grouped by snapshot
    # First pass: count survivors per snapshot
    counts = np.zeros(sample_size, dtype=np.int64)
    for i in range(sz_sd):
        if survived_boundary[i] and time[i] >= T:
            counts[class_index_sd[i]] += 1

    # Second pass: fill output arrays
    out = List()
    for c in range(sample_size):
        out.append(np.empty(counts[c], dtype=np.float64))

    fill_idx = np.zeros(sample_size, dtype=np.int64)
    for i in range(sz_sd):
        if survived_boundary[i] and time[i] >= T:
            c = class_index_sd[i]
            out[c][fill_idx[c]] = pos[i]
            fill_idx[c] += 1

    return out


# ---------------------------------------------------------------------------
# Top-level simulate functions (thin wrappers — can stay as plain Python or
# be @njit themselves; kept as plain Python for flexibility)
# ---------------------------------------------------------------------------

@njit
def simulate(kplus, kminus, rburst, D, T, dt, z, lengths, seed=-1):
    """Full spatial + population simulation."""
    if seed >= 0:
        np.random.seed(seed)
    Tbirth_sd, Nbirths_sd, class_index_sd = population_dynamics(
        z, kplus, kminus, rburst, T
    )
    return spatial_dynamics(
        Tbirth_sd, Nbirths_sd, class_index_sd, D, T, dt, z, lengths
    )


@njit
def simulate_nospace(kplus, kminus, rburst, T, z):
    """Population-only simulation (no diffusion). Returns per-snapshot counts."""
    _, counts, _ = population_dynamics(z, kplus, kminus, rburst, T)
    return counts
