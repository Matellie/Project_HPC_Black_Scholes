import numpy as np

def mc_classic(S0, K, T, r, sigma, num_paths, length_simulation, num_paths_to_plot=0):
    dt = T / length_simulation  
    paths = np.zeros((num_paths, length_simulation + 1))  
    paths[:, 0] = S0  
    
    payoff_sum = 0.0

    for i in range(num_paths):
        path = [S0]
        for _ in range(length_simulation):
            Z = np.random.normal(0, 1) 
            S_t = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(S_t)
        #if i < num_paths_to_plot:
        #	paths[i, :] = path
        payoff_sum += max(path[-1] - K, 0)  

    option_price = (np.exp(-r * T) * payoff_sum) / num_paths
    #paths_to_plot = paths[:num_paths_to_plot, :]
    
    return option_price#, paths_to_plot

