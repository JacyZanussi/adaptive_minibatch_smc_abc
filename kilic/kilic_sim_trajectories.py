import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)


def simulate_gene_expression_plotting(params, G, K, t_max, sample_times, max_steps=10000):
    """
    Numba-compatible stochastic gene expression simulation (Gillespie SSA).
    params: [K_12, K_21, ..., Beta_1,...,Beta_G, Delta]
    Returns:
        t_out : (K, len(sample_times)) array
        m_out : (K, len(sample_times)) array
    """
    n_switch = G * (G - 1)
    K_rates = np.zeros((G, G))
    idx = 0
    for i in range(G):
        for j in range(G):
            if i != j:
                K_rates[i, j] = params[idx]
                idx += 1

    beta = np.zeros(G)
    for i in range(G):
        beta[i] = params[n_switch + i]

    delta = params[-1]

    n_samples = sample_times.shape[0]
    t_out = np.full((K, n_samples), np.nan)
    m_out = np.full((K, n_samples), np.nan)
    s_out = np.full((K, n_samples), np.nan)

    for cell in range(K):
        t = 0.0
        m = 0
        s = 0  # initial promoter state
        sample_idx = 0
        t_sample = sample_times[sample_idx]

        for step in range(max_steps):
            if t >= t_max:
                break

            # Reaction rates
            total_rate = 0.0
            rates = np.zeros(G + 2)
            for j in range(G):
                if j != s:
                    rates[j] = K_rates[s, j]
                    total_rate += rates[j]
                else:
                    rates[j] = 0.0

            rates[G] = beta[s]              # production
            rates[G + 1] = delta * m        # degradation
            total_rate += rates[G] + rates[G + 1]

            if total_rate <= 0.0:
                break

            # Draw next reaction time
            t_last = t
            m_last = m
            s_last = s
            t += np.random.exponential(1.0 / total_rate)

            # Pick reaction
            r = np.random.rand() * total_rate
            cum = 0.0
            reaction = -1
            for i in range(G + 2):
                cum += rates[i]
                if r < cum:
                    reaction = i
                    break

            # Apply reaction
            if reaction < G:
                s = reaction
            elif reaction == G:
                m += 1
            else:
                if m > 0:
                    m -= 1

            # Record samples when crossing times
            while sample_idx < n_samples and t >= t_sample:
                t_out[cell, sample_idx] = t_last
                m_out[cell, sample_idx] = m_last
                s_out[cell, sample_idx] = s_last
                sample_idx += 1
                if sample_idx < n_samples:
                    t_sample = sample_times[sample_idx]

        # Fill remaining samples with NaN
        while sample_idx < n_samples:
            t_out[cell, sample_idx] = np.nan
            m_out[cell, sample_idx] = np.nan
            s_out[cell, sample_idx] = np.nan
            sample_idx += 1

    return t_out, m_out, s_out

# ── LaTeX + font setup ────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'lines.linewidth': 1.4,
})

true_params = np.array([0.00533, 0.03, 0 , 0.166  , 0.00533])
t_max = 1801.0
sample_times = np.arange(0,1801,1)
obs_t = np.array([0,15,30,45,60,75,90,105,120,150,180,240,300,360,480,600,1200,1800])
G = 2
K = 1

t_out, m_out, s_out = simulate_gene_expression_plotting(true_params, G, K, t_max, sample_times, max_steps=100000)

#plt.scatter(sample_times, m_out[0])
offset = 0.2 * np.max(m_out)
plt.figure(figsize=(9,4))
plt.step(sample_times, m_out[0], color = 'steelblue', where = 'post', label = 'mRNA Count')
plt.step(sample_times, (offset-1)*s_out[0] - offset, color = 'orange',label = 'State',linewidth = 1)
plt.scatter(obs_t,m_out[0,obs_t],color = 'firebrick',zorder = 4,label = 'Observed')
ytickobs = np.zeros(obs_t.shape) - offset - 1
plt.scatter(obs_t,ytickobs,color = 'firebrick',zorder = 4,marker='^')
plt.axhline(-offset, color='gray', linewidth=0.5, alpha=0.75,zorder = -1)
plt.axhline(0, color='gray', linewidth=0.5, alpha=0.75,zorder = -1)

plt.legend()
plt.xlabel("Time")
plt.xticks(np.arange(0,1801,200))
plt.xlim(-20,1820)
plt.tight_layout()


plt.savefig("kilic/model_timeseries.svg", format = 'svg', dpi = 600, bbox_inches="tight", transparent=True)
plt.show()





















'''




# =========================
# time series plot
# =========================
@njit
def simulate_gene_expression_plotting(
    params, G, K, t_max, sample_times, max_steps=10000
):
    n_switch = G * (G - 1)
    K_rates = np.zeros((G, G))
    idx = 0

    for i in range(G):
        for j in range(G):
            if i != j:
                K_rates[i, j] = params[idx]
                idx += 1

    beta = params[n_switch:n_switch + G]
    delta = params[-1]

    n_samples = sample_times.shape[0]
    t_out = np.full((K, n_samples), np.nan)
    m_out = np.full((K, n_samples), np.nan)
    s_out = np.full((K, n_samples), np.nan)

    for cell in range(K):
        t, m, s = 0.0, 0, 0
        sample_idx = 0

        for _ in range(max_steps):
            if t >= t_max:
                break

            rates = np.zeros(G + 2)
            for j in range(G):
                if j != s:
                    rates[j] = K_rates[s, j]

            rates[G] = beta[s]
            rates[G + 1] = delta * m
            total_rate = rates.sum()

            if total_rate <= 0:
                break

            t_last, m_last, s_last = t, m, s
            t += np.random.exponential(1.0 / total_rate)

            r = np.random.rand() * total_rate
            cum = 0.0
            for i in range(G + 2):
                cum += rates[i]
                if r < cum:
                    if i < G:
                        s = i
                    elif i == G:
                        m += 1
                    elif m > 0:
                        m -= 1
                    break

            while sample_idx < n_samples and t >= sample_times[sample_idx]:
                t_out[cell, sample_idx] = t_last
                m_out[cell, sample_idx] = m_last
                s_out[cell, sample_idx] = s_last
                sample_idx += 1

    return t_out, m_out, s_out







# =========================
# SSA PATH (single cell)
# =========================

@njit
def simulate_path(params, G, t_max, max_steps=100000):
    n_switch = G * (G - 1)

    K_rates = np.zeros((G, G))
    idx = 0
    for i in range(G):
        for j in range(G):
            if i != j:
                K_rates[i, j] = params[idx]
                idx += 1

    beta = params[n_switch:n_switch + G]
    delta = params[-1]

    t = 0.0
    m = 0
    s = 0  # start in state 0

    t_path = np.zeros(max_steps)
    m_path = np.zeros(max_steps)

    step = 0
    while t < t_max and step < max_steps:
        total_rate = 0.0
        rates = np.zeros(G + 2)

        for j in range(G):
            if j != s:
                rates[j] = K_rates[s, j]
                total_rate += rates[j]

        rates[G] = beta[s]
        rates[G + 1] = delta * m
        total_rate += rates[G] + rates[G + 1]

        if total_rate == 0.0:
            break

        t += np.random.exponential(1.0 / total_rate)

        r = np.random.rand() * total_rate
        cum = 0.0
        reaction = -1
        for i in range(G + 2):
            cum += rates[i]
            if r < cum:
                reaction = i
                break

        if reaction < G:
            s = reaction
        elif reaction == G:
            m += 1
        else:
            if m > 0:
                m -= 1

        t_path[step] = t
        m_path[step] = m
        step += 1

    return t_path[:step], m_path[:step]


# =========================
# SNAPSHOT SSA (K cells)
# =========================

@njit
def simulate_snapshots(params, G, K, t_max, sample_times, max_steps=10000):
    n_switch = G * (G - 1)

    K_rates = np.zeros((G, G))
    idx = 0
    for i in range(G):
        for j in range(G):
            if i != j:
                K_rates[i, j] = params[idx]
                idx += 1

    beta = params[n_switch:n_switch + G]
    delta = params[-1]

    n_samples = sample_times.shape[0]
    m_out = np.zeros((K, n_samples))

    for cell in range(K):
        t = 0.0
        m = 0
        s = 0
        sample_idx = 0

        for step in range(max_steps):
            if t >= t_max or sample_idx >= n_samples:
                break

            total_rate = 0.0
            rates = np.zeros(G + 2)

            for j in range(G):
                if j != s:
                    rates[j] = K_rates[s, j]
                    total_rate += rates[j]

            rates[G] = beta[s]
            rates[G + 1] = delta * m
            total_rate += rates[G] + rates[G + 1]

            if total_rate == 0.0:
                break

            t_last = t
            m_last = m
            t += np.random.exponential(1.0 / total_rate)

            r = np.random.rand() * total_rate
            cum = 0.0
            reaction = -1
            for i in range(G + 2):
                cum += rates[i]
                if r < cum:
                    reaction = i
                    break

            if reaction < G:
                s = reaction
            elif reaction == G:
                m += 1
            else:
                if m > 0:
                    m -= 1

            while sample_idx < n_samples and t >= sample_times[sample_idx]:
                m_out[cell, sample_idx] = m_last
                sample_idx += 1

    return m_out


# =========================
# PLOTTING
# =========================

# ── LaTeX + font setup ────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'lines.linewidth': 1.4,
})


def plot_and_save(params, G=2, K=200, t_max=50.0, n_samples=25):
    sample_times = np.linspace(0, t_max, n_samples)

    # ---- Single-cell trajectory
    #t_path, m_path = simulate_path(params, G, t_max)

    #plt.figure(figsize=(6, 3))
    #plt.step(t_path, m_path, where="post", linewidth=1.2)
    #plt.xlabel("Time")
    #plt.ylabel("mRNA count")
    #plt.title("Single-cell SSA trajectory")
    #plt.tight_layout()
    #plt.savefig("ssa_single_cell.svg", dpi=600)
    #plt.close()

    # ---- Population snapshots
    #m_snap = simulate_snapshots(params, G, K, t_max, sample_times)

    #mean_m = m_snap.mean(axis=0)
    #var_m = m_snap.var(axis=0)

    #fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    #ax[0].plot(sample_times, mean_m, marker="o")
    #ax[0].set_ylabel("Mean mRNA")

    #ax[1].plot(sample_times, var_m, marker="o")
    #ax[1].set_ylabel("Variance")
    #ax[1].set_xlabel("Time")

    #fig.suptitle("Snapshot statistics across cells")
    #fig.tight_layout()
    #fig.savefig("ssa_population_snapshots.svg", dpi=600)
    #plt.close()


    true_params = np.array([0.00533, 0.03, 0 , 0.166  , 0.00533])
    t_max = 1801.0
    sample_times = np.arange(0,1801,1)
    # Obtained from Wang data
    obs_t = [0,15,30,45,60,75,90,105,120,150,180,240,300,360,480,600,1200,1800]
    G = 2
    K = 1

    t_out, m_out, s_out = simulate_gene_expression_plotting(true_params, G, K, t_max, sample_times, max_steps=100000)


    #plt.scatter(sample_times, m_out[0])
    offset = 0.2 * np.max(m_out)
    plt.figure(figsize=(9,4))
    plt.step(sample_times, m_out[0], color = 'steelblue', where = 'post', label = 'mRNA Count')
    plt.step(sample_times, (offset-1)*s_out[0] - offset, color = 'orange',label = 'State',linewidth = 1)
    plt.scatter(obs_t,m_out[0,obs_t],color = 'firebrick',zorder = 4,label = 'Observed')
    ytickobs = np.zeros(obs_t.shape) - offset - 1
    plt.scatter(obs_t,ytickobs,color = 'firebrick',zorder = 4,marker='^')
    plt.axhline(-offset, color='gray', linewidth=0.5, alpha=0.75,zorder = -1)
    plt.axhline(0, color='gray', linewidth=0.5, alpha=0.75,zorder = -1)

    plt.legend()
    plt.xlabel("Time")
    plt.xticks(np.arange(0,1801,200))
    plt.xlim(-20,1820)
    plt.tight_layout()

    #plt.savefig("figures_ecoli/model_timeseries.svg", format = 'svg', dpi = 600, bbox_inches="tight", transparent=True)
    plt.show()

# =========================
# RUN
# =========================

if __name__ == "__main__":
    # Example parameters (2-state model)
    # [K_01, K_10, beta_0, beta_1, delta]
    # Parameters from Wang
    params = np.array([0.00533, 0.03, 0 , 0.166  , 0.00533])
    plot_and_save(params)
'''
