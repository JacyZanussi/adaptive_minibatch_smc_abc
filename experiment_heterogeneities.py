'''
Figure 4: sweeps each model's source of between-particle heterogeneity (TD: gamma/beta shape
parameters for site position and nuclear length; LV: initial-condition range) rather than the
physical rate parameters, comparing constant-batch vs. FVC at each perturbed value.
Inline arguments: $SLURM_ARRAY_TASK_ID
Array index -> sweep: 0-1 = TD [constant_gamma_param, fvc_gamma_param]; 2-3 = LV
[constant_ic_range, fvc_ic_range].

Note: baseline constant/fvc runs (unperturbed heterogeneity) are already covered by
experiment_hyperparameters.py and are not repeated here.
'''
#imports
import experiments
import sys

satid = int(sys.argv[1])
print(f"Slurm Array Task ID: {satid}")

num_sims = 10

# We use n0 to sweep across various c values, since they change with n0 approximately like 1/n0
# Baseline constant and fvc can be obtained from the hyperparameter sweep
sweeps_td = {
    # Transcriptional Dynamics
    # baseline: [2,4] (bp,gp)
    'constant_gamma_param' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'heterogeneities' : [[f*2,f*4] for f in [0.5,1.0,2.0]], 'scheme':['constant']
    }),
    'fvc_gamma_param' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16]],
        'heterogeneities' : [[f*2,f*4] for f in [0.5,1.0,2.0]], 'scheme':['fvc']
    })
}

sweeps_lv = { 
    # Lotka Volterra
    # baseline: [999] (ic_range)
    'constant_ic_range' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'heterogeneities' : [[w] for w in [1,251,501,751]], 'scheme':['constant']
    }),
    'fvc_ic_range' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'heterogeneities' : [[w] for w in [1,251,501,751]], 'scheme':['fvc']
    })
}


if satid < 2:
    model = 'td'
    sweeps = sweeps_td
else:
    satid = satid - 2
    model = 'lv'
    sweeps = sweeps_lv

sweep_keys = [k for k in sweeps.keys()]
sweep_name = sweep_keys[satid]
param_list = sweeps[sweep_keys[satid]]

print(f"Performing experiment: {sweep_name} on {model}")
print(f"There will be: {len(param_list)} times {num_sims} estimations performed.")

filename = 'results/' + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)