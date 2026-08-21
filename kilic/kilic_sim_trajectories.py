"""
Standalone figure script: simulates and plots a single Gillespie trajectory (mRNA count +
promoter state over time) for the illustrative Kilic model figure. This duplicates the SSA
logic in `kilic_model.simulate` without numba, and additionally records the promoter state
`s_out`, so a single trajectory can be plotted step-by-step; it is not used for inference.
"""
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)


def simulate_gene_expression_plotting(params, G, K, t_max, sample_times, max_steps=10000):
    """
    Plain-Python stochastic gene expression simulation (Gillespie SSA), also returning the
    promoter state trace (unlike `kilic_model.simulate`).

    params: [K_12, K_21, ..., Beta_1,...,Beta_G, Delta]
    Returns:
        t_out, m_out, s_out : each (K, len(sample_times)) arrays (time of last reaction,
        mRNA count, and promoter state at each sample time)
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


FONT_SIZE_AXES_LABEL = 16
FONT_SIZE_AXES_TITLE = 16
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_LEGEND = 10
FONT_SIZE_LEGEND_TITLE = 10
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'CMU Serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'lines.linewidth': 1.4,
    'svg.fonttype' : 'none',
    'svg.hashsalt' : '42',
    'axes.labelsize': FONT_SIZE_AXES_LABEL,
    'axes.titlesize': FONT_SIZE_AXES_TITLE,
    'xtick.labelsize': FONT_SIZE_XTICK,
    'ytick.labelsize': FONT_SIZE_YTICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'legend.title_fontsize': FONT_SIZE_LEGEND_TITLE,
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


plt.savefig("kilic/model_timeseries.pdf", format = 'pdf', dpi = 600, bbox_inches="tight", transparent=True)
#plt.show()

