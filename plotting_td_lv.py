'''
Plot results for Transcriptional Dynamics and Lotka Volterra simulations

'''

import os
import experiments as E

RESULTS_DIR = 'results'
def p(name):
    return os.path.join(RESULTS_DIR, name + '.pkl')


### Figure 2 Pareto Frontiers
OUTPUT_DIR = RESULTS_DIR + '/figure2_plots'
comparisons = [
    (p('td_constant'),p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda'),
    (p('td_constant'),p('td_fvc_n0'), 0, r'n_0', 'td_n0'),
    (p('td_constant'),p('td_fvc_c'), 1, r'c', 'td_c'),
    (p('lv_constant'),p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda'),
    (p('lv_constant'),p('lv_fvc_n0'), 0, r'n_0', 'lv_n0'),
    (p('lv_constant'),p('lv_fvc_c'), 1, r'c', 'lv_c'),
]

### Figure 2 Plots over generations
generation_plots = [
    (p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda'),
    (p('td_fvc_n0'), 0, r'n_0', 'td_n0'),
    (p('td_fvc_c'), 1, r'c', 'td_c'),
    (p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda'),
    (p('lv_fvc_n0'), 0, r'n_0', 'lv_n0'),
    (p('lv_fvc_c'), 1, r'c', 'lv_c'),
]

time_series_args = {
    'attr_list':['batch_size'],
    'attr_ylabels':[r'\mathrm{Batch\ Size}\ (n)'],
    'attr_transform':['id']
}
for ref, novel, key_idx, ltitle, tag in comparisons:
    print(f'Pareto: {tag}')
    E.pareto_frontier(
        ref, novel,
        legend_key_index=key_idx,
        legend_title = '$' + ltitle + '$',
        out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.svg')
    )

for fpath, key_idx, ltitle, tag in generation_plots:
    print(f'Time series: {tag}')
    E.time_series(
        fpath,
        legend_key_index=key_idx,
        legend_title='$' + ltitle + '$',
        out_path = os.path.join(OUTPUT_DIR, f'{tag}_gen_series.svg'),
        **time_series_args
    )