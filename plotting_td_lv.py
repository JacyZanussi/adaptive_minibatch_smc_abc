'''
Plot results for Transcriptional Dynamics and Lotka Volterra simulations

'''
import sys
import os
import experiments as E
import numpy as np 

#I only do this so I don't have to replot everything every time.
if len(sys.argv) > 1:
    section = int(sys.argv[1])
else:
    section = 'all'

RESULTS_DIR = 'results'
def p(name):
    return os.path.join(RESULTS_DIR, name + '.pkl')


### Figure 2 Pareto Frontiers
OUTPUT_DIR = RESULTS_DIR + '/figure2_plots'

## A fairly self-contaiend yet complicated function for obtaining epsilon thresholds at different levels of SNR
lv_threshold_res = E.load('results/lv_lv_threshold.pkl')
td_threshold_res = E.load('results/td_td_threshold.pkl')
def get_epsilon_thresholds(results, ph_stop_fn = lambda x: x['snr'] < 2):
    eps_thresholds = []
    for k in results.keys():
        #eps_thresholds[k] = []
        res_ph = E.ph_stop(results, ph_stop_fn)
        eps = E.get_attr(res_ph,'alpha_threshold',trunc=False,slice=-1)
        eps_thresholds = np.quantile(eps[k],q=0.8)
    return eps_thresholds
def eps_stop(x,eps):
    return x['alpha_threshold'] < eps or x['acceptance_rate'] < 0.02

lv_eps = get_epsilon_thresholds(lv_threshold_res)
td_eps = get_epsilon_thresholds(td_threshold_res)

lv_stop = lambda x: eps_stop(x,lv_eps)
td_stop = lambda x: eps_stop(x,td_eps)

comparisons = [
    (p('td_constant'),p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda',td_stop),
    (p('td_constant'),p('td_fvc_n0'), 0, r'n_0', 'td_n0',td_stop),
    (p('td_constant'),p('td_fvc_c'), 1, r'c', 'td_c',td_stop),
    (p('lv_constant'),p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda',lv_stop),
    (p('lv_constant'),p('lv_fvc_n0'), 0, r'n_0', 'lv_n0',lv_stop),
    (p('lv_constant'),p('lv_fvc_c'), 1, r'c', 'lv_c',lv_stop),
]

if section == 0 or section == 'all':
    for ref, novel, key_idx, ltitle, tag, phs in comparisons:
        print(f'Pareto: {tag}')
        E.pareto_frontier(
            ref, novel,
            legend_key_index=key_idx,
            legend_title = '$' + ltitle + '$',
            post_hoc_stop=phs,
            out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.svg')
        )

### Figure 2 Plots over generations
generation_plots = [
    (p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda',td_stop),
    (p('td_fvc_n0'), 0, r'n_0', 'td_n0',td_stop),
    (p('td_fvc_c'), 1, r'c', 'td_c',td_stop),
    (p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda',lv_stop),
    (p('lv_fvc_n0'), 0, r'n_0', 'lv_n0',lv_stop),
    (p('lv_fvc_c'), 1, r'c', 'lv_c',lv_stop),
]

attr_list = ['batch_size','noise']
ylabel_list = [r'\mathrm{Batch\ Size}\ (n)', r'\mathrm{(\frac{v_t}{n_t})}']
ytrans_list = ['id', 'log']

if section == 1 or section == 'all':
    time_series_args = {
    'attr_list':attr_list,
    'attr_ylabels':ylabel_list,
    'attr_transform':ytrans_list
    }
    for fpath, key_idx, ltitle, tag, phs in generation_plots:
        print(f'Time series: {tag}')
        E.time_series(
            fpath,
            legend_key_index=key_idx,
            legend_title='$' + ltitle + '$',
            post_hoc_stop=phs,
            out_handle=os.path.join(OUTPUT_DIR, f'{tag}'),
            **time_series_args
        )



### Figure 3: Paretor Frontiers 
OUTPUT_DIR = RESULTS_DIR + '/figure3_plots'

kplus_eps = get_epsilon_thresholds(E.load('results/td_constant_kplus_threshold.pkl'))
rburst_eps = get_epsilon_thresholds(E.load('results/td_constant_rburst_threshold.pkl'))
diffusivity_eps = get_epsilon_thresholds(E.load('results/td_constant_diffusivity_threshold.pkl'))
gamma_beta_eps = get_epsilon_thresholds(E.load('results/td_constant_gamma_beta_threshold.pkl'))

snr = 3
alpha_eps = get_epsilon_thresholds(E.load('results/lv_constant_alpha_threshold.pkl'),lambda x: x['snr'] < snr)
beta_eps = get_epsilon_thresholds(E.load('results/lv_constant_beta_threshold.pkl'),lambda x: x['snr'] < snr)
gamma_eps = get_epsilon_thresholds(E.load('results/lv_constant_gamma_threshold.pkl'),lambda x: x['snr'] < snr)
ic_range_eps = get_epsilon_thresholds(E.load('results/lv_constant_ic_range_threshold.pkl'),lambda x: x['snr'] < snr)


comparisons = [
    (p('td_constant_kplus'),p('td_fvc_kplus'), 3, r'k_+', 'td_kplus',lambda x: eps_stop(x,kplus_eps)),
    (p('td_constant_rburst'),p('td_fvc_rburst'), 4, r'r_{\mathrm{burst}}', 'td_rburst',lambda x: eps_stop(x,rburst_eps)),
    (p('td_constant_diffusivity'),p('td_fvc_diffusivity'), 5, r'D', 'td_diffusivity',lambda x: eps_stop(x,diffusivity_eps)),
    (p('lv_constant_alpha'),p('lv_fvc_alpha'), 3, r'\alpha', 'lv_alpha',lambda x: eps_stop(x,alpha_eps)),
    (p('lv_constant_beta'),p('lv_fvc_beta'), 4, r'\beta', 'lv_beta',lambda x: eps_stop(x,beta_eps)),
    (p('lv_constant_gamma'),p('lv_fvc_gamma'), 5, r'\gamma', 'lv_gamma',lambda x: eps_stop(x,gamma_eps)),
]

if section == 2 or section == 'all':
    for ref, novel, key_idx, ltitle, tag, phs in comparisons:
        print(f'Pareto: {tag}')
        E.pareto_frontier(
            ref, novel,
            legend_key_index=key_idx,
            legend_title = '$' + ltitle + '$',
            post_hoc_stop=phs,
            out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.svg')
        )

### Figure 3: Pareto Frontiers for Heterogeneities
comparisons = [
    (p('td_constant_gamma_beta'),p('td_fvc_gamma_beta'), 3, r's', 'td_gamma_beta',lambda x: eps_stop(x,gamma_beta_eps)),
    (p('lv_constant_ic_range'),p('lv_fvc_ic_range'), 3, r'w_{\mathrm{ic}}', 'lv_ic_range',lambda x: eps_stop(x,ic_range_eps)),
]

if section == 3 or section == 'all':
    for ref, novel, key_idx, ltitle, tag, phs in comparisons:
        print(f'Pareto: {tag}')
        E.pareto_frontier(
            ref, novel,
            legend_key_index=key_idx,
            legend_title = '$' + ltitle + '$',
            post_hoc_stop=phs,
            out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.svg')
        )


if section == 4 or section == 'all':
    E.pareto_frontier_scatter(
        'results/td_constant.pkl',
        'results/td_fvc_c.pkl',
        x_attr='total_time',
        y_attr='log_hdpr_product',
        out_filename='results/pareto_scatter.svg'
    )

if section == 5 or section == 'all':
    E.pareto_frontier_schemes_vs_perturbations(
        scheme_results_list=[
            {'results':'results/td_constant_gamma_beta.pkl','label':'Constant'},
            {'results':'results/td_fvc_gamma_beta.pkl','label':'Adaptive'}
        ],
        heterogeneity_value=2,
        heterogeneity_index=0,
        perturbation_index=1,
        title=r'Heterogeneity: $\beta = 2$...',
        perturbation_label='something',
        out_filename='results/pareto_heterogeneity_scatter.svg'
    )