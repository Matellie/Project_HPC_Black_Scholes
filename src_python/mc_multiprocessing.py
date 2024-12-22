import numpy as np
from multiprocessing import Pool

def simulate_path(S0, K, T, r, sigma, length_simulation):
    """Simulate a single path and compute its payoff."""
    dt = T / length_simulation
    path = [S0]
    for _ in range(length_simulation):
        Z = np.random.normal(0, 1)
        S_t = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(S_t)
    payoff = max(path[-1] - K, 0)  
    return payoff #path

def mc_multiprocessing(S0, K, T, r, sigma, num_paths, length_simulation, num_paths_to_plot=0, num_processes=2):
    """Parallel Monte Carlo simulation for Black-Scholes option pricing."""
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            simulate_path,
            [(S0, K, T, r, sigma, length_simulation) for _ in range(num_paths)]
        )
    
    #paths = np.zeros((num_paths_to_plot, length_simulation + 1))
    payoffs = []
    
    for i, (payoff) in enumerate(results):
        #if i < num_paths_to_plot:
        #    paths[i, :] = path
        payoffs.append(payoff)
    
    payoff_sum = sum(payoffs)
    option_price = (np.exp(-r * T) * payoff_sum) / num_paths
    return option_price#, paths
