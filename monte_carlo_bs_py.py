import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

def monte_carlo_black_scholes(S0, K, T, r, sigma, num_paths, length_simulation, num_paths_to_plot):
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
        if i < num_paths_to_plot:
        	paths[i, :] = path
        payoff_sum += max(path[-1] - K, 0)  # Call option payoff

    # Discounted average payoff
    option_price = (np.exp(-r * T) * payoff_sum) / num_paths
    paths_to_plot = paths[:num_paths_to_plot, :]
    return option_price, paths_to_plot

def save_to_csv(paths, filename):
    df = pd.DataFrame(paths.T)
    df.to_csv(filename, index=False)
    print(f"Paths saved to {filename}")

def plot_paths(paths, filename):
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(paths[i, :], alpha=0.7)
    plt.title("Monte Carlo Simulated Stock Price Paths")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.grid()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Black-Scholes Option Pricing")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--num_paths", type=int, default=int(1e4), help="Number of simulation paths")
    parser.add_argument("--length_simulation", type=int, default=100, help="Number of time intervals")
    parser.add_argument("--num_paths_to_plot", type=int, default=100, help="Number of paths to plot")
    args = parser.parse_args()

    start_time = time.time()
    option_price, paths_to_plot = monte_carlo_black_scholes(
        args.S0, args.K, args.T, args.r, args.sigma, args.num_paths, args.length_simulation, args.num_paths_to_plot
    )
    end_time = time.time()

    print(f"Option Price: {option_price}")
    print(f"Elapsed Time: {end_time - start_time} seconds")

    csv_filename = "paths.csv"
    plot_filename = "paths_plot.png"

    save_to_csv(paths_to_plot, csv_filename)
    plot_paths(paths_to_plot, plot_filename)
