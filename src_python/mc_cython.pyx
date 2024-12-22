import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt

def mc_cython(double S0, double K, double T, double r, double sigma,
                       int num_paths, int length_simulation, int num_paths_to_plot=0):
    cdef double dt = T / length_simulation
    cdef double payoff_sum = 0.0
    cdef int i, j
    #cdef cnp.ndarray[cnp.float64_t, ndim=2] paths_to_plot = np.zeros((num_paths_to_plot, length_simulation + 1), dtype=np.float64)
    cdef double S_t, payoff
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Z = np.random.normal(0.0, 1.0, length_simulation)

    for i in range(num_paths):
        S_t = S0
        #if i < num_paths_to_plot:
        #    paths_to_plot[i, 0] = S_t

        for j in range(length_simulation):
            S_t = S_t * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z[j])
            #if i < num_paths_to_plot:
            #    paths_to_plot[i, j + 1] = S_t

        payoff = max(S_t - K, 0.0)
        payoff_sum += payoff

    option_price = (exp(-r * T) * payoff_sum) / num_paths
    return option_price #, paths_to_plot
