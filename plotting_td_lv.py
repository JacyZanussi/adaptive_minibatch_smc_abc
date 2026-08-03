'''
Plot results for Transcriptional Dynamics and Lotka Volterra simulations

'''
import gc
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



##
td_stop = lambda x: (x['generation'] == 18) 
lv_stop = lambda x: (x['generation'] == 20)

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
        print(f'Pareto: {tag}.')
        E.pareto_frontier(
            ref, novel,
            legend_key_index=key_idx,
            legend_title = '$' + ltitle + '$',
            post_hoc_stop=phs,
            y_transform='log',
            out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.pdf')
        )
        gc.collect()


### Figure 2 Plots over generations
generation_plots = [
    (p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda',td_stop),
    (p('td_fvc_n0'), 0, r'n_0', 'td_n0',td_stop),
    (p('td_fvc_c'), 1, r'c', 'td_c',td_stop),
    (p('td_constant'),0, r'n_t' ,'td_constant',td_stop),
    (p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda',lv_stop),
    (p('lv_fvc_n0'), 0, r'n_0', 'lv_n0',lv_stop),
    (p('lv_fvc_c'), 1, r'c', 'lv_c',lv_stop),
    (p('lv_constant'),0, r'n_t', 'lv_constant',lv_stop)
]

attr_list = ['batch_size','noise','acceptance_rate','ESS','log_hdpr_product','alpha_threshold','total_sims']
ylabel_list = [r'\mathrm{Batch\ Size}\ (n)', r'\mathrm{(\frac{v_t}{n_t})}',
               r'\mathrm{(A)}',r'\mathrm{ESS}',r'\mathrm{(HDPR Vol)}', r'\mathrm{(\epsilon)}', r'\mathrm{Simulations}']
ytrans_list = ['id', 'log', 'log','id','log','log','log']


if section == 1 or section == 'all':
    time_series_args = {
    'attr_list':attr_list,
    'attr_ylabels':ylabel_list,
    'attr_transform':ytrans_list
    }
    for fpath, key_idx, ltitle, tag, phs in generation_plots:
        if 'constant' in fpath:
            cm = 'YlOrRd'
        else:
            cm = 'GnBu'
        print(f'Time series: {tag}.')
        E.time_series(
            fpath,
            legend_key_index=key_idx,
            legend_title='$' + ltitle + '$',
            post_hoc_stop=phs,
            errorbar_cmap = cm,
            out_handle=os.path.join(OUTPUT_DIR, f'{tag}'),
            **time_series_args
        )


comparisons = [
    (p('td_constant'),p('td_fvc_lambda'), 2, r'\lambda', 'td_lambda',td_stop,'total_sims',r'\mathrm{Simulations}'),
    (p('td_constant'),p('td_fvc_n0'), 0, r'n_0', 'td_n0',td_stop,'total_sims',r'\mathrm{Simulations}'),
    (p('td_constant'),p('td_fvc_c'), 1, r'c', 'td_c',td_stop,'total_sims',r'\mathrm{Simulations}'),
    (p('lv_constant'),p('lv_fvc_lambda'), 2, r'\lambda', 'lv_lambda',lv_stop,'total_sims',r'\mathrm{Simulations}'),
    (p('lv_constant'),p('lv_fvc_n0'), 0, r'n_0', 'lv_n0',lv_stop,'total_sims',r'\mathrm{Simulations}'),
    (p('lv_constant'),p('lv_fvc_c'), 1, r'c', 'lv_c',lv_stop,'total_sims',r'\mathrm{Simulations}'),
]

if section == 2 or section == 'all':
    for ref, novel, key_idx, ltitle, tag, phs, x_attr, x_label in comparisons:
        print(f'Pareto: {tag}.')
        E.pareto_frontier(
            ref, novel,
            legend_key_index=key_idx,
            legend_title = '$' + ltitle + '$',
            post_hoc_stop=phs,
            x_attr=x_attr,
            x_label=x_label,
            x_transform='log',
            out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto_simulations.pdf')
        )
        gc.collect()

# ### Pareto Frontiers for each heterogeneity
# This is 
OUTPUT_DIR = RESULTS_DIR + '/figure3_plots'


comparisons = [
    (p('td_constant_kplus'),p('td_fvc_kplus'), 3, r'$c$', 'td_kplus', 15, (16,16)),
    (p('td_constant_rburst'),p('td_fvc_rburst'), 4, r'$c$', 'td_rburst', 10, (11,18)),
    (p('td_constant_diffusivity'),p('td_fvc_diffusivity'), 5, r'$c$', 'td_diffusivity', 0.1,(13,8)),
    (p('lv_constant_alpha'),p('lv_fvc_alpha'), 3, r'$c$', 'lv_alpha', 5, (19,20)),
    (p('lv_constant_beta'),p('lv_fvc_beta'), 4, r'$c$', 'lv_beta', 0.02, (14,20)),
    (p('lv_constant_gamma'),p('lv_fvc_gamma'), 5, r'$c$', 'lv_gamma',5,(18,20)),
]

def get_key_list(data,val,idx):
    set = []
    for k in data.keys():
        if np.all(np.isclose(val,k[idx],)):
            set.append(k)
    return set

if section == 3 or section == 'all':
    for ref, novel, key_idx, ltitle, tag, param, final_gen in comparisons:
        #Up, then Down
        for i in range(2):
            direction = 'up' if i == 0 else 'down'
            factor = 5.0 if i == 0 else 1/5.0
            ref_data = E.load(ref)
            novel_data = E.load(novel)
            val = factor * param
            subset_ref = get_key_list(ref_data,val,key_idx-2)
            subset_novel = get_key_list(novel_data,val,key_idx)
            ref_sub = E.subset(ref_data,subset_ref)
            novel_sub = E.subset(novel_data,subset_novel)
            print([k for k in ref_sub.keys()])
            print([k for k in novel_sub.keys()])

            phs = lambda x: (x['generation'] == final_gen[i])
            print(f'Pareto: {tag}')
            E.pareto_frontier(
                ref_sub, novel_sub,
                legend_key_index=1,
                legend_title = '$' + ltitle + '$',
                post_hoc_stop=phs,
                auto_c_color='cool',
                y_transform='log',
                out_filename=os.path.join(OUTPUT_DIR, f'{tag}_{direction}_pareto.pdf')
            )

#### Figure 4 - Heterogeneities
# This is split up into two: TD, which has 9 perturbations of the parameters for the gamma distribution
# and LV, which has 4-5 perturbations that are sequential (ic range). So,...

def r(x):
    return round(x,4)

stop_dict = {
    (r(10/3),r(1/30)):10,(r(10),r(1/30)):13,(r(30),r(1/30)):17,
    (r(10/3),r(1/10)):14, (r(10),r(1/10)):17, (r(30),r(1/10)):16,
    (r(10/3),r(3/10)):17, (r(10),r(3/10)):17 , (r(30),r(3/10)):18
}


OUTPUT_DIR = RESULTS_DIR + '/figure4_plots'

if section == 4 or section == 'all':
    ref_file = p('td_constant_scale')
    novel_file = p('td_fvc_scale')
    ref = E.load(ref_file)
    novel = E.load(novel_file)
    
    #iterate over the factors of gamma
    factors = [1/3.0,1.0,3.0]
    for i in factors:
        ititle = r'$\frac{1}{3}$' if i < 1 else str(int(i))
        for j in factors:
            jtitle = r'$\frac{1}{3}$' if i < 1 else str(int(j))
            het = (10 * i, j / 10) #gamma with factors
            title = 'Shape Factor: ' + ititle + "\n" +  "Scale Factor: " + jtitle + '/10'
            val = get_key_list(ref,het,2)
            val_novel = get_key_list(novel,het,4)
            ref_ss = E.subset(ref,val)
            novel_ss = E.subset(novel,val_novel)
            het_rounded = (r(het[0]),r(het[1]))
            phs = lambda x : (x['generation'] >= stop_dict[het_rounded])
            
            left_tag = 'up_' if i > 2 else ('eq_' if i > 0.5 else 'down_')
            right_tag = 'up' if j > 2 else ('eq' if j > 0.5 else 'down')

            tag = left_tag + right_tag
            E.pareto_frontier(
                ref_ss, novel_ss,
                legend_key_index=1,
                legend_title = r'$c$',
                post_hoc_stop=phs,
                title = title,
                auto_c_color='cool',
                y_transform='log',
                out_filename=os.path.join(OUTPUT_DIR, f'td_scale_{tag}_pareto.pdf')
            )



OUTPUT_DIR = RESULTS_DIR + '/figure4_plots'

if section == 5 or section == 'all':
    ref_file = p('lv_constant_ic')
    novel_file = p('lv_fvc_ic')
    ref = E.load(ref_file)
    novel = E.load(novel_file)
    
    #iterate over the IC
    ic_list = [1,250,500,750]
    for het in ic_list:
        val = get_key_list(ref,het,1)
        val_novel = get_key_list(novel,het,3)
        ref_ss = E.subset(ref,val)
        novel_ss = E.subset(novel,val_novel)
        phs = lambda x : (x['generation'] >= 20)
        
        tag = str(int(het))
        E.pareto_frontier(
            ref_ss, novel_ss,
            legend_key_index=1,
            legend_title = r'$c$',
            post_hoc_stop=phs,
            auto_c_color='cool',
            y_transform='log',
            out_filename=os.path.join(OUTPUT_DIR, f'lv_ic_{tag}_pareto.pdf')
        )


#Plot Posteriors for both TD and LV
if section == 6 or section == 'all':
    pass


# ### Figure 3: Pareto Frontiers for Heterogeneities
# comparisons = [
#     (p('td_constant_gamma_beta'),p('td_fvc_gamma_beta'), 3, r's', 'td_gamma_beta',lambda x: eps_stop(x,gamma_beta_eps)),
#     (p('lv_constant_ic_range'),p('lv_fvc_ic_range'), 3, r'w_{\mathrm{ic}}', 'lv_ic_range',lambda x: eps_stop(x,ic_range_eps)),
# ]

# if section == 3 or section == 'all':
#     for ref, novel, key_idx, ltitle, tag, phs in comparisons:
#         print(f'Pareto: {tag}')
#         E.pareto_frontier(
#             ref, novel,
#             legend_key_index=key_idx,
#             legend_title = '$' + ltitle + '$',
#             post_hoc_stop=phs,
#             out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.pdf')
#         )


# if section == 4 or section == 'all':
#     E.pareto_frontier_scatter(
#         'results/td_constant.pkl',
#         'results/td_fvc_c.pkl',
#         x_attr='total_time',
#         y_attr='log_hdpr_product',
#         out_filename='results/pareto_scatter.pdf'
#     )


# #Physical parameters...
# comparisons = [
#         #transcriptional dynamics
#     (p('td_constant_kplus'),p('td_fvc_kplus'),1,[0,2,4],
#         r'$\left(\frac{k_+}{2},r_{\mathrm{burst}},D\right)$', r'c', 'td_kplus_half'),
#     (p('td_constant_kplus'),p('td_fvc_kplus'),1,[1,3,5],
#         r'$\left(2k_+,r_{\mathrm{burst}},D\right)$', r'c', 'td_kplus_double'),
#     (p('td_constant_rburst'),p('td_fvc_rburst'),1,[0,2,4],
#         r'$\left(k_+,\frac{r_{\mathrm{burst}}}{2},D\right)$', r'c', 'td_rburst_half'),
#     (p('td_constant_rburst'),p('td_fvc_rburst'),1,[1,3,5],
#         r'$\left(k_+,2r_{\mathrm{burst}},D\right)$', r'c', 'td_rburst_double'),
#     (p('td_constant_diffusivity'),p('td_fvc_diffusivity'),1,[0,2,4],
#         r'$\left(k_+,r_{\mathrm{burst}},\frac{D}{2}\right)$', r'c', 'td_diffusivity_half'),
#     (p('td_constant_diffusivity'),p('td_fvc_diffusivity'),1,[1,3,5],
#         r'$\left(k_+,r_{\mathrm{burst}},2D\right)$', r'c', 'td_diffusivity_double'),
#         # Lotka volterra below
#     (p('lv_constant_alpha'),p('lv_fvc_alpha'),1,[0,2,4],
#         r'$\left(\frac{\alpha}{2},\beta,\gamma\right)$', r'c', 'lv_alpha_half'),
#     (p('lv_constant_alpha'),p('lv_fvc_alpha'),1,[1,3,5],
#         r'$\left(2\alpha,\beta,\gamma\right)$', r'c', 'lv_alpha_double'),
#     (p('lv_constant_beta'),p('lv_fvc_beta'),1,[0,2,4],
#         r'$\left(\alpha,\frac{\beta}{2},\gamma\right)$', r'c', 'lv_beta_half'),
#     (p('lv_constant_beta'),p('lv_fvc_beta'),1,[1,3,5],
#         r'$\left(\alpha,2\beta,\gamma\right)$', r'c', 'lv_beta_double'),
#     (p('lv_constant_gamma'),p('lv_fvc_gamma'),1,[0,2,4],
#         r'$\left(\alpha,\beta,\frac{\gamma}{2}\right)$', r'c', 'lv_gamma_half'),
#     (p('lv_constant_gamma'),p('lv_fvc_gamma'),1,[1,3,5],
#         r'$\left(\alpha,\beta,2\gamma\right)$', r'c', 'lv_gamma_double')
# ]

# if section == 5 or section == 'all':
#     for ref, novel, key_idx, subset_idx,plot_title, ltitle, tag in comparisons:
#         E.pareto_frontier(
#             ref, novel,
#             legend_key_index=key_idx,
#             legend_title = '$' + ltitle + '$',
#             subset_idx=subset_idx,
#             subset_idx_ref=subset_idx,
#             plot_title=plot_title,
#             auto_c_color='cool',
#             out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.pdf')
#         )



# # Heterogeneities
# comparisons = [
#         #transcriptional dynamics
#     (p('td_constant_gamma_beta'),p('td_fvc_gamma_beta'),1,[0,3,6],
#         r'$\frac{1}{2}\left(2,4\right)$', r'c', 'td_gamma_beta_half'),
#     (p('td_constant_gamma_beta'),p('td_fvc_gamma_beta'),1,[1,4,7],
#         r'$\left(2,4\right)$', r'c', 'td_gamma_beta_one'),
#     (p('td_constant_gamma_beta'),p('td_fvc_gamma_beta'),1,[2,5,8],
#         r'2$\left(2,4\right)$', r'c', 'td_gamma_beta_double'),
    
#         # Lotka volterra below
#     (p('lv_constant_ic_range'),p('lv_fvc_ic_range'),1,[0,3,6],
#         r'$x_0 \sim U\left(375,625\right)$', r'c', 'lv_ic_range_250'),
#     (p('lv_constant_ic_range'),p('lv_fvc_ic_range'),1,[1,4,7],
#         r'$x_0 \sim U\left(250,750\right)$', r'c', 'lv_ic_range_500'),
#     (p('lv_constant_ic_range'),p('lv_fvc_ic_range'),1,[2,5,8],
#         r'$x_0 \sim U\left(125,875\right)$', r'c', 'lv_ic_range_750')
# ]

# if section == 6 or section == 'all':
#     for ref, novel, key_idx, subset_idx,plot_title, ltitle, tag in comparisons:
#         E.pareto_frontier(
#             ref, novel,
#             legend_key_index=key_idx,
#             legend_title = '$' + ltitle + '$',
#             subset_idx=subset_idx,
#             subset_idx_ref=subset_idx,
#             plot_title=plot_title,
#             auto_c_color='PuBuGn',
#             out_filename=os.path.join(OUTPUT_DIR, f'{tag}_pareto.pdf')
#         )