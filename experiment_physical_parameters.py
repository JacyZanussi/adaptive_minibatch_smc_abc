'''
Experiments for Figure 2: Transcriptional dynamics
Inline arguments: $SLURM_ARRAY_TASK_ID
'''
#imports
import experiments
import sys

satid = int(sys.argv[1])
print(f"Slurm Array Task ID: {satid}")


num_sims = 10


#Baseline constant and fvc can be obtained from the hyperparameter sweep
sweeps_td = { #baseline: [30,10,0.1]
    'constant_kplus' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[k,10,0.1] for k in [15,60]], 'scheme':['constant']
    }),
    'fvc_kplus' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16]],
        'physical_params' : [[k,10,0.1] for k in [15,60]], 'scheme':['fvc']
    }),
    'constant_rburst' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[30,r,0.1] for r in [5,20]], 'scheme':['constant']
    }),
    'fvc_rburst' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16]],
        'physical_params' : [[30,r,0.1] for r in [5,20]], 'scheme':['fvc']
    }),
    'constant_diffusivity' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[30,10,d] for d in [0.05,0.2]], 'scheme':['constant']
    }),
    'fvc_diffusivity' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16]],
        'physical_params' : [[30,10,d] for d in [0.05,0.2]], 'scheme':['fvc']
    })
}

sweeps_lv = { #baseline: [4,0.01,4]
    'constant_alpha' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[a,0.01,4] for a in [2,8]], 'scheme':['constant']
    }),
    'fvc_alpha' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[a,0.01,4] for a in [2,8]], 'scheme':['fvc']
    }),
    'constant_beta' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[4,b,4] for b in [0.05,0.02]], 'scheme':['constant']
    }),
    'fvc_beta' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[4,b,4] for b in [0.05,0.02]], 'scheme':['fvc']
    }),
    'constant_gamma' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[4,0.01,g] for g in [2,8]], 'scheme':['constant']
    }),
    'fvc_gamma' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[4,0.01,g] for g in [2,8]], 'scheme':['fvc']
    })
}

if satid < 6:
    model = 'td'
    sweeps = sweeps_td
else:
    satid = satid - 6
    model = 'lv'
    sweeps = sweeps_lv

sweep_keys = [k for k in sweeps.keys()]
sweep_name = sweep_keys[satid]
param_list = sweeps[sweep_keys[satid]]

print(f"Performing experiment: {sweep_name} on {model}")
print(f"There will be: {len(param_list)} times {num_sims} estimations performed.")

filename = 'results/' + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)



