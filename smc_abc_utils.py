### Auxiliary functions for smc abc
import numpy as np
import timeit
from matplotlib import pyplot as plt
from numba import njit

def uniform_prior(prior_domain, seed = None):
    # Ensure it's a clean, picklable NumPy array
    prior_domain = np.ascontiguousarray(prior_domain, dtype=np.float64)
    lower = prior_domain[:, 0]
    upper = prior_domain[:, 1]
    
    # Calculate volume here (standard numpy math is picklable)
    vol = np.prod(upper - lower)

    # These wrappers only reference the arrays, which pickle perfectly
    base_rng = np.random.default_rng(seed) if seed is not None else None
    def prior_func(rng=None):
        if rng is not None:
            return rng.uniform(lower, upper)
        if base_rng is not None:
            return base_rng.uniform(lower, upper)
        return n_sample(lower, upper)

    def density(x):
        return n_density(x, lower, upper, vol)

    return [prior_func, prior_domain, density]

# Use standard NJIT functions instead of a JitClass instance
@njit
def n_sample(l, u):
    out = np.empty(len(l))
    for i in range(len(l)):
        out[i] = np.random.uniform(l[i], u[i])
    return out

@njit
def n_density(x, l, u, v):
    # Handle batching or single
    x_arr = np.atleast_2d(x)
    res = np.zeros(len(x_arr))
    for i in range(len(x_arr)):
        in_bounds = True
        for j in range(x_arr.shape[1]):
            if x_arr[i, j] < l[j] or x_arr[i, j] > u[j]:
                in_bounds = False
                break
        res[i] = 1.0 / v if in_bounds else 0.0
    return res if x.ndim > 1 else res[0]


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

### Function for visualizing and estimating time increase per sample
# Benchmark Parameters
def benchmark_time(batch_sizes, simulator):
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
    plt.title("Simulator Performance")
    plt.xlabel("Batch Size ($N$)")
    plt.ylabel("Time per call (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()