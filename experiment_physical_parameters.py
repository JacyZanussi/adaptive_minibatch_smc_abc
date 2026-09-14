'''
Experiments for Figure 2: Transcriptional dynamics
Inline arguments: $SLURM_ARRAY_TASK_ID
'''
#imports
import experiments
import sys

satid = int(sys.argv[1])
output_location = str(sys.argv[2])
print(f"Slurm Array Task ID: {satid}")
print(f"Output file location: {output_location}")


num_sims = 10

def stop_func(est):
    return (est.generation >= 20) or (est.acceptance_rate < 0.001) or (est.total_time > 3600)



#Baseline constant and fvc can be obtained from the hyperparameter sweep
sweeps_td = { #baseline: [30,10,0.1]
    'constant_kplus' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[15*k,10,0.1] for k in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_kplus' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'physical_params' : [[15*k,10,0.1] for k in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'constant_rburst' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[15,10*r,0.1] for r in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_rburst' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'physical_params' : [[15,10*r,0.1] for r in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'constant_diffusivity' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[15,10,0.1*d] for d in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_diffusivity' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'physical_params' : [[15,10,0.1*d] for d in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    })
}

sweeps_lv = { #baseline: [4,0.01,4]
    'constant_alpha' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[a*5,0.02,5] for a in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_alpha' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[a*5,0.02,5] for a in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'constant_beta' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[5,b*0.02,5] for b in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_beta' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[5,b*0.02,5] for b in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'constant_gamma' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32,64,128]],
        'physical_params' : [[5,0.02,g*5] for g in [0.2,5]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_gamma' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4,8]],
        'physical_params' : [[5,0.02,g*5] for g in [0.2,5]], 'scheme':['fvc'], 'stop_func':[stop_func]
    })
}


sweeps_pb = { #baseline: [30,10,0.1]
    'constant_kplus_pb' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[k,10,0.1] for k in [15,60]], 'scheme':['constant'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    }),
    'fvc_kplus_pb' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16,32]],
        'physical_params' : [[k,10,0.1] for k in [15,60]], 'scheme':['fvc'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    }),
    'constant_rburst_pb' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[30,r,0.1] for r in [5,20]], 'scheme':['constant'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    }),
    'fvc_rburst_pb' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16,32]],
        'physical_params' : [[30,r,0.1] for r in [5,20]], 'scheme':['fvc'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    }),
    'constant_diffusivity_pb' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'physical_params' : [[30,10,d] for d in [0.05,0.2]], 'scheme':['constant'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    }),
    'fvc_diffusivity_pb' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,16,32]],
        'physical_params' : [[30,10,d] for d in [0.05,0.2]], 'scheme':['fvc'],
        'stop_func' : [stop_func],
        'stats_per_batch' : [True]
    })
}


if satid < 6:
    model = 'td'
    sweeps = sweeps_td
elif satid < 12:
    satid = satid - 6
    model = 'lv'
    sweeps = sweeps_lv
else:
    model = 'td'
    satid = satid - 12
    sweeps = sweeps_pb
    print(f"Performing sweep fror per batch summaries for td")


sweep_keys = [k for k in sweeps.keys()]
sweep_name = sweep_keys[satid]
param_list = sweeps[sweep_keys[satid]]

print(f"Performing experiment: {sweep_name} on {model}")
print(f"There will be: {len(param_list)} times {num_sims} estimations performed.")

filename = output_location + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)
