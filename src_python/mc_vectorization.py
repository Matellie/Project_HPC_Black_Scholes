import numpy as np

def mc_vectorized(S0, K, T, r, sigma, num_paths, length_simulation):
    dt = T / length_simulation
    Z = np.random.normal(0, 1, (num_paths, length_simulation))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack((np.zeros((num_paths, 1)), log_paths)) 
    paths = S0 * np.exp(log_paths)

    payoff = np.maximum(paths[:, -1] - K, 0) 
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price#, paths
