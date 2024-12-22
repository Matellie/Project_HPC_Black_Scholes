import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from mc_classic import mc_classic
from mc_vectorization import mc_vectorized
from mc_multiprocessing import mc_multiprocessing
from mc_numba import mc_numba
from mc_cython import mc_cython
from mc_multithreading import mc_multithreading

methods = {
    "Classic": mc_classic,
    "Vectorized": mc_vectorized,
    "Multithreading": mc_multithreading,
    "Multiprocessing": mc_multiprocessing,
    "Numba": mc_numba,
    "Cython": mc_cython,
}

def benchmark_method(method, n_runs, S0, K, T, r, sigma, num_paths, length_simulation):
    exec_times = []
    prices = []
    for _ in range(n_runs):
        start_time = time.time()
        price = method(S0, K, T, r, sigma, num_paths, length_simulation)
        end_time = time.time()
        exec_times.append(end_time - start_time)
        prices.append(price)
    return exec_times, prices

def boxplot_results(results, filename):
    plt.figure(figsize=(12, 6))
    plt.boxplot(results.values(), labels=results.keys())
    plt.title("Execution Time Comparison for Monte Carlo Methods")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename)
    print(f"Box plot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Black-Scholes Option Pricing Benchmark")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--num_paths", type=int, default=int(1e4), help="Number of simulation paths")
    parser.add_argument("--length_simulation", type=int, default=100, help="Number of time intervals")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs for each method")
    parser.add_argument("--output_plot", type=str, default="benchmark_results.png", help="Filename for the box plot")
    args = parser.parse_args()

    results = {}
    for method_name, method_func in methods.items():
        print(f"Benchmarking {method_name} :")
        exec_times, prices = benchmark_method(
            method_func,
            args.n_runs,
            args.S0,
            args.K,
            args.T,
            args.r,
            args.sigma,
            args.num_paths,
            args.length_simulation,
        )
        results[method_name] = exec_times
        print(f"{method_name} - Mean option price: {np.mean(prices):.4f}")
        print(f"{method_name} - Mean execution time: {np.mean(exec_times):.4f} seconds")

    boxplot_results(results, args.output_plot)
