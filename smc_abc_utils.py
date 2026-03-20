### Auxiliary functions for smc abc
import numpy as np
import timeit
from matplotlib import pyplot as plt

### Uniform prior triple generator
## Prior
def uniform_prior(prior_domain):
    '''
    Generates a uniform prior triple for use in the smc abc iterator. 

    prior_domain : np array of shape (N,2) for N parameters to be estimated    
    '''
    prior_domain = np.array(prior_domain)
    lower = prior_domain[:,0]
    upper = prior_domain[:,1]
    vol = np.prod(upper - lower)
    def prior_func():
        return np.random.uniform(prior_domain.T[0],prior_domain.T[1])
    def density(x):
        x = np.atleast_2d(x)
        in_bounds = np.all((x>= lower) & (x <= upper), axis = 1)
        densities = np.zeros((x.shape[0],1))
        densities[in_bounds] = 1/vol
        return densities[:,0]
    prior = [prior_func,prior_domain,density]
    return prior



### stop function
comparison_dict = {
    'lte': lambda x,y : x <= y,
    'lt': lambda x,y : x < y,
    'gte': lambda x,y : x >= y,
    'gt': lambda x,y : x > y
}
def continue_func(attribute = 'current_alpha_threshold',comparison = "lte",threshold = np.inf):
    """
    Returns a function that takes in an smc_abc object and returns True until the stopping criteria are satisfied. 
    Functions so that While stopping criteria are not met, generate.

    attribute : string - attribute of the smc_abc iterator object to be used 
    """
    if comparison not in [x for x in comparison_dict]:
        raise ValueError("comparison input must be \"lte\" \"lt\" \"gte\" or \"gt\". \n " \
        "These compare the estimate value x with a threshold y as: x <= y, x < y, x >= y, x > y, respectively. ")
    comparison_func = comparison_dict[comparison]
    def func(est):
        criteria = stop(est,attribute,threshold,comparison_func)
        return criteria
    return func

def stop(est,attribute_name,threshold = 0.025,compare = lambda x,y : x < y):
    if compare(getattr(est,attribute_name),threshold):
        return False
    else:
        return True

## Wrapper for continue_func
def continue_func_multiple(criteria : tuple):
    ## criteria must be a list of tuples like (attribute, comparison, threshold)
    return



### Function for visualizing and estimating time increase per sample
# Benchmark Parameters
def benchmark_time(batch_sizes, simulator):
    #batch_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000]
    results = []

    print("Running benchmarks...")
    for n in batch_sizes:
        # Use lambda to pass your parameters to the simulator
        # repeat=3 runs the test 3 times, number=10 runs the function 10 times per test
        t = timeit.repeat(lambda: simulator(n), 
                        repeat=3, number=10)
        
        # Standard practice: take the best of the repetitions
        avg_time_per_call = min(t) / 10
        results.append(avg_time_per_call)
        print(f"Batch {n}: {avg_time_per_call:.6f} sec")
    print(f"Average increase in seconds per sample: {np.mean(np.diff(results)/np.diff(batch_sizes))}")


    # Simple Plot
    plt.figure(figsize=(6, 4))
    plt.plot(batch_sizes, results, marker='o', linestyle='-', color='b')
    plt.title("GMM Simulator Performance")
    plt.xlabel("Batch Size ($N$)")
    plt.ylabel("Time per call (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()