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
    return (est.generation >= 12)


a = 1/3.0
b = 1.0
c = 3.0

# We use n0 + automatic c selection to sweep across various c values, since they are likely changed by n0
#Baseline constant and fvc can be obtained from the hyperparameter sweep
sweeps = {
    'fvc_scale1_8_a' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [8]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1_8_b' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [8]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [b] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1_8_c' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [8]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [c] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1_32_a' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1_32_b' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [b] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale1_32_c' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [c] for g in [b]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale2_32_a' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [a] for g in [c]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale2_32_b' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [b] for g in [c]], 'scheme':['fvc'], 'stop_func':[stop_func]
    }),
    'fvc_scale2_32_c' : experiments.make_parameter_list({
        'scheme_params':[[n0,None,1] for n0 in [32]],
        'heterogeneities' : [[(2,2),(f*10,g/10)] for f in [c] for g in [c]], 'scheme':['fvc'], 'stop_func':[stop_func]
    })
}




model = 'td'
#sweeps = sweeps_td

sweep_keys = [k for k in sweeps.keys()]
sweep_name = sweep_keys[satid]
param_list = sweeps[sweep_keys[satid]]

print(f"Performing experiment: {sweep_name} on {model}")
print(f"There will be: {len(param_list)} times {num_sims} estimations performed.")


filename = output_location + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)
