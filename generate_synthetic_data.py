'''
Generate synthetic data
 
'''
#### Generate lotka volterra data
import lotka_volterra as lv
import transcriptional_dynamics as td
import numpy as np
import pickle



# Lotka Volterra
seed = 0
np.random.seed(seed)

sample_size = 128
alpha,beta,gamma,delta = 4,0.02,4,1
true_params = (alpha,beta,gamma,delta)
dt = 0.01
t_max = 5
ic_range = 1000
ic = np.random.choice(ic_range,size = (sample_size,2)) + 1

times,ts = lv.tau_leaping(ic,t_max,alpha,beta,gamma,delta,dt = dt)

dataset = {
    'data':ts,
    'ic':ic,
    'true_params':true_params,
    'dt':dt,
    't_max':t_max,
    'sample_size':sample_size,
    'seed':seed
}

filename = 'datasets/lv_stochastic.pkl'
with open(filename,"wb") as f:
    pickle.dump(dataset,f)
print(f"Lotka Volterra data saved to {filename}")


# Transcriptional Dynamics
np.random.seed(0)
sample_size = 512
kplus,kminus,rburst,diffusivity = 30,10,10,0.1
true_params = (kplus,kminus,rburst,diffusivity)
dt = 0.01
t_max = 5
gp,bp = 4,2
lengths = np.random.gamma(shape = gp, scale = 1/gp, size = (sample_size,))
sites = np.random.beta(a = bp, b = bp, size = (sample_size,)) * lengths

data = td.simulate(kplus, kminus, rburst, diffusivity, t_max, dt, sites, lengths, seed=0)

dataset = {
    'data':list(data),
    'sites':sites,
    'lengths':lengths,
    'gp':gp,
    'bp':bp,
    'true_params':true_params,
    'dt':dt,
    't_max':t_max,
    'sample_size':sample_size,
    'seed':seed
}

filename = 'datasets/td.pkl'
with open(filename,"wb") as f:
    pickle.dump(dataset,f)
print(f"Transcriptional_dynamics data saved to {filename}")