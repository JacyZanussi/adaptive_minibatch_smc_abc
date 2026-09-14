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
    return (est.generation >= 20) or (est.acceptance_rate < 0.001)

def stop_func_lv(est):
    return (est.generation >= 20) or (est.acceptance_rate < 0.001)


sweeps_td = {
    # Transcriptional Dynamics 
    'constant' : experiments.make_parameter_list({
                    'scheme_params':[[x] for x in [32,64,128,256,512]], 
                    'scheme': ['constant'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func]
                }),
    'fvc_lambda' : experiments.make_parameter_list({
                    'scheme_params':[[2,None,l] for l in [0,0.25,0.5,0.75,1.0]], 
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func]
                }),
    'fvc_n0' : experiments.make_parameter_list({
                    'scheme_params':[[n0,None,1.0] for n0 in [2,4,8,16,32]], 
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func]
                }),
    'fvc_c' : experiments.make_parameter_list({
                    'scheme_params':[[2,c,1.0] for c in [0.1,0.2,0.3,0.4,None]], 
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func]
                })
}
sweeps_lv = {
    # Lotka Volterra 
    'constant' : experiments.make_parameter_list({
                    'scheme_params':[[x] for x in [8,16,32,64,128]],
                    'scheme': ['constant'],
                    'input_filename': ['datasets/lv_stochastic.pkl'],
                    'stop_func':[stop_func_lv]
                }),
    'fvc_lambda' : experiments.make_parameter_list({
                    'scheme_params':[[2,None,l] for l in [0,0.25,0.5,0.75,1.0]], 
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/lv_stochastic.pkl'],
                    'stop_func':[stop_func_lv]
                }),
    'fvc_n0' : experiments.make_parameter_list({
                    'scheme_params':[[n0,None,1.0] for n0 in [2,4,8,16,32]],
                    'scheme': ['fvc'],
                    'input_filename':[ 'datasets/lv_stochastic.pkl'],
                    'stop_func':[stop_func_lv]
                }),
    'fvc_c' : experiments.make_parameter_list({
                    'scheme_params':[[2,c,1.0] for c in [0.1,0.2,0.3,0.4,None]], 
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/lv_stochastic.pkl'],
                    'stop_func':[stop_func_lv]
                })
}

sweeps_pb = {
    # Transcriptional Dynamics
    'constant_pb' : experiments.make_parameter_list({
                    'scheme_params':[[x] for x in [32,64,128,256,512]],
                    'scheme': ['constant'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func],
                    'stats_per_batch':[True]
                }),
    'fvc_lambda_pb' : experiments.make_parameter_list({
                    'scheme_params':[[2,None,l] for l in [0,0.25,0.5,0.75,1.0]],
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func],
                    'stats_per_batch':[True]
                }),
    'fvc_n0_pb' : experiments.make_parameter_list({
                    'scheme_params':[[n0,None,1.0] for n0 in [2,4,8,16,32]],
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func],
                    'stats_per_batch':[True]
                }),
    'fvc_c_pb' : experiments.make_parameter_list({
                    'scheme_params':[[2,c,1.0] for c in [0.1,0.2,0.3,0.4,None]],
                    'scheme': ['fvc'],
                    'input_filename': ['datasets/td.pkl'],
                    'stop_func':[stop_func],
                    'stats_per_batch':[True]
                })
}


if satid < 4:
    model = 'td'
    sweeps = sweeps_td
elif satid < 8:
    satid = satid - 4
    model = 'lv'
    sweeps = sweeps_lv
else:
    sweeps = sweeps_pb
    model = 'td'
    satid = satid - 8
    print(f"Performing per-batch for td")


sweep_keys = [k for k in sweeps.keys()]
sweep_name = sweep_keys[satid]
param_list = sweeps[sweep_keys[satid]]

print(f"Performing experiment: {sweep_name} on {model}")
print(f"There will be: {len(param_list)} times {num_sims} estimations performed.")
print(param_list)

filename = output_location + model + "_" + sweep_name + '.pkl'
experiments.simulate_experiment(filename, param_list, model, num_sims=num_sims)
