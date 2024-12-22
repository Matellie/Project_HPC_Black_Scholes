import numpy as np
from concurrent.futures import ThreadPoolExecutor

def mc_path(S0, K, T, r, sigma, dt, length_simulation):
    path = [S0]
    for _ in range(length_simulation):
        Z = np.random.normal(0, 1)
        S_t = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(S_t)
    return path

def mc_multithreading(S0, K, T, r, sigma, num_paths, length_simulation, num_paths_to_plot=0):
    dt = T / length_simulation 
    payoff_sum = 0.0

    # Using ThreadPoolExecutor to run the simulations in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all the paths to be computed in parallel
        futures = [executor.submit(mc_path, S0, K, T, r, sigma, dt, length_simulation) for _ in range(num_paths)]
        
        for future in futures:
            path = future.result()  # Get the result of the simulation
            payoff_sum += max(path[-1] - K, 0)

    option_price = (np.exp(-r * T) * payoff_sum) / num_paths
    
    return option_price
