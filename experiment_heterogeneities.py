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



a = 1/3.0
b = 1.0
c = 3.0



## We use n0 + automatic c selection to sweep across various c values, since they are likely changed by n0
#Baseline constant and fvc can be obtained from the hyperparameter sweep
sweeps_td = { #baseline: [2,4] [bp,gp]
    'constant_scale0' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [a]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'constant_scale1' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [b]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'constant_scale2' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [128,256,512]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [c]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_scale0' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [a]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale2' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,8,32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a,b,c] for g in [c]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
}


sweeps_lv = { #baseline: [4,0.01,4]
    'constant_ic_range0' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [32]],
        'heterogeneities' : [[w] for w in [1,250,500,750]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'constant_ic_range1' : experiments.make_parameter_list({
        'scheme_params':[[x] for x in [64,128]],
        'heterogeneities' : [[w] for w in [1,250,500,750]], 'scheme':['constant'], 'stop_func':[stop_func]
    }),
    'fvc_ic_range0' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [2,4]],
        'heterogeneities' : [[w] for w in [1,250,500,750]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_ic_range1' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [8,16]],
        'heterogeneities' : [[w] for w in [1,250,500,750]], 'scheme':['fvc'], 'stop_func':[stop_func]
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


filename = output_location + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)
