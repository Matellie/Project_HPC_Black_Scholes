import numpy as np
from numba import njit, prange

@njit
def simulate_path_numba(S0, K, T, r, sigma, length_simulation):
    """Simulates a single stock price path using the Black-Scholes model."""
    dt = T / length_simulation
    path = [S0]
    for _ in range(length_simulation):
        Z = np.random.normal(0, 1)
        S_t = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(S_t)
    payoff = max(path[-1] - K, 0)  # Call option payoff
    return path, payoff

@njit(parallel=True)
def mc_numba(S0, K, T, r, sigma, num_paths, length_simulation, num_paths_to_plot=0):
    """
    Monte Carlo simulation for option pricing using the Black-Scholes model.
    """
    dt = T / length_simulation
    payoffs = np.zeros(num_paths)
    #paths_to_plot = np.zeros((num_paths_to_plot, length_simulation + 1))

    for i in prange(num_paths):
        path = np.zeros(length_simulation + 1)  # Use NumPy array instead of a list
        path[0] = S0
        for t in range(1, length_simulation + 1):
            Z = np.random.normal(0, 1)
            path[t] = path[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        payoffs[i] = max(path[-1] - K, 0)  # Call option payoff

        #if i < num_paths_to_plot:
        #    paths_to_plot[i, :] = path  # Now `path` is a NumPy array

    # Discounted average payoff
    option_price = (np.exp(-r * T) * payoffs.sum()) / num_paths
    return option_price#, paths_to_plot