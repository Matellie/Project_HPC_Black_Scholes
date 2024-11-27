import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

def explicit_fd(S_max, K, T, r, sigma, M, N):
    dS = S_max / M
    dt = T / N
    grid = np.zeros((M + 1, N + 1))
    stock_prices = np.linspace(0, S_max, M + 1)

    grid[:, -1] = np.maximum(stock_prices - K, 0)

    for j in range(N + 1):
        grid[0, j] = 0
        grid[-1, j] = S_max - K * np.exp(-r * (T - j * dt))

    alpha = 0.5 * dt * (sigma**2 * np.arange(M + 1)**2 - r * np.arange(M + 1))
    beta = 1 - dt * (sigma**2 * np.arange(M + 1)**2 + r)
    gamma = 0.5 * dt * (sigma**2 * np.arange(M + 1)**2 + r * np.arange(M + 1))

    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            grid[i, j] = alpha[i] * grid[i - 1, j + 1] + beta[i] * grid[i, j + 1] + gamma[i] * grid[i + 1, j + 1]

    return grid, stock_prices

def implicit_fd(S_max, K, T, r, sigma, M, N):
    dS = S_max / M
    dt = T / N
    grid = np.zeros((M + 1, N + 1))
    stock_prices = np.linspace(0, S_max, M + 1)

    grid[:, -1] = np.maximum(stock_prices - K, 0)

    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * (T - np.arange(N + 1) * dt))

    j = np.arange(1, M)
    alpha = -0.5 * dt * (sigma**2 * j**2 - r * j)
    beta = 1 + dt * (sigma**2 * j**2 + r)
    gamma = -0.5 * dt * (sigma**2 * j**2 + r * j)
    A = np.diag(alpha[1:], -1) + np.diag(beta) + np.diag(gamma[:-1], 1)

    for n in range(N - 1, -1, -1):
        grid[1:M, n] = np.linalg.solve(A, grid[1:M, n + 1])

    return grid, stock_prices

def crank_nicolson(S_max, K, T, r, sigma, M, N):
    dS = S_max / M
    dt = T / N
    grid = np.zeros((M + 1, N + 1))
    stock_prices = np.linspace(0, S_max, M + 1)

    grid[:, -1] = np.maximum(stock_prices - K, 0)

    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * (T - np.arange(N + 1) * dt))

    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta = -0.5 * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)
    A = np.diag(-alpha[1:], -1) + np.diag(1 - beta) + np.diag(-gamma[:-1], 1)
    B = np.diag(alpha[1:], -1) + np.diag(1 + beta) + np.diag(gamma[:-1], 1)

    for n in range(N - 1, -1, -1):
        b = B @ grid[1:M, n + 1]
        grid[1:M, n] = np.linalg.solve(A, b)

    return grid, stock_prices

def finite_difference_black_scholes(method, S0, K, T, r, sigma, S_max, M, N):
    if method == 'explicit':
        dS = S_max / M  
        dt = T / N     
        # Verify the stability condition (using CFL)
        max_allowed_dt = (dS ** 2) / (2 * sigma ** 2 * S_max ** 2)
        if dt <= max_allowed_dt:
            print(f"The explicit method is stable. Maximum allowed Δt: {max_allowed_dt:.6f}")
            grid, stock_prices = explicit_fd(S_max, K, T, r, sigma, M, N)
        else:
            print(f"The explicit method is unstable. Δt is {dt:.6f} for maximum allowed of: {max_allowed_dt:.6f}. Reduce Δt.")
            raise ValueError("Stability condition not satisfied for explicit method. Adjust Δt or ΔS.")
    elif method == 'implicit':
        grid, stock_prices = implicit_fd(S_max, K, T, r, sigma, M, N)
    elif method == 'crank_nicolson':
        grid, stock_prices = crank_nicolson(S_max, K, T, r, sigma, M, N)
    else:
        raise ValueError("Invalid method. Choose 'explicit', 'implicit', or 'crank_nicolson'.")

    option_price = np.interp(S0, stock_prices, grid[:, 0])
    return option_price, grid, stock_prices

def save_to_csv(grid, stock_prices, filename):
    df = pd.DataFrame(grid, index=stock_prices)
    df.to_csv(filename, index_label="Stock Price")
    print(f"Grid saved to {filename}")

def plot_grid(grid, stock_prices, filename):
    plt.figure(figsize=(10, 6))
    for t in range(grid.shape[1]):
        if t % (grid.shape[1] // 10) == 0 or t == grid.shape[1] - 1:
            plt.plot(stock_prices, grid[:, t], label=f"t={t}")
    plt.title("Finite Difference Option Prices Across Time Steps")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite Difference Black-Scholes Option Pricing")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--S_max", type=float, default=200.0, help="Maximum stock price for grid")
    parser.add_argument("--M", type=int, default=100, help="Number of stock price steps")
    parser.add_argument("--N", type=int, default=100, help="Number of time steps")
    args = parser.parse_args()

    methods = ['explicit', 'implicit', 'crank_nicolson']

    for method in methods:
        try: 
            start_time = time.time()
            option_price, grid, stock_prices = finite_difference_black_scholes(
                method, args.S0, args.K, args.T, args.r, args.sigma, args.S_max, args.M, args.N
            )
            end_time = time.time()
            print(f"Method: {method}, Option Price: {option_price:.4f}")
            print(f"Elapsed Time: {end_time - start_time} seconds")
            
            csv_filename = f"fdm_grid_{method}.csv"
            plot_filename = f"fdm_plot_{method}.png"

            save_to_csv(grid, stock_prices, csv_filename)
            plt.plot(stock_prices, grid[:, 0], label=method)
        except Exception as e:
            print(f"An error occurred for method {method}: {e}")

    plt.title("Finite Difference Methods for Black-Scholes")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.legend()
    plt.show()

