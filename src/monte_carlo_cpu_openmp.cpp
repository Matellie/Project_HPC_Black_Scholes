#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <random>

struct SimulationParams {
  double S0 = 100.0;          // Initial stock price
  double K = 100.0;           // Strike price
  double T = 1.0;             // Time to maturity (1 year)
  double r = 0.05;            // Risk-free rate (5%)
  double sigma = 0.2;         // Volatility (20%)
  int nSimul = 1'000'000;     // Number of simulation paths
  int nThreads = 1;           // Number of threads
  int lengthSimulation = 252; // Number of time intervals
};

// Monte Carlo simulation using Black-Scholes model
double monteCarloBlackScholes(const SimulationParams &params) {
  // Precalculate all the constants before entering the loops
  const double dt = params.T / params.lengthSimulation; // Time step
  const double drift = params.r - 0.5 * params.sigma * params.sigma;
  const double diffusion = params.sigma * std::sqrt(dt);
  double payoffSum = 0.0; // Sum of payoffs

#pragma omp parallel reduction(+ : payoffSum)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

#pragma omp for
    for (int i = 0; i < params.nSimul; ++i) {
      double ST = params.S0; // Starting stock price
      for (int j = 0; j < params.lengthSimulation; ++j) {
        double Z = dist(gen); // Standard normal random variable
        ST *= std::exp(drift * dt + diffusion * Z);
      }
      // Call option payoff
      double payoff = std::max(ST - params.K, 0.0);
      payoffSum += payoff;
    }
  }

  // Discounted payoff, averaged on all the nSimul simulations
  return std::exp(-params.r * params.T) * payoffSum / params.nSimul;
}

// Command-line argument parser
void parseArguments(int argc, char **argv, SimulationParams &params) {
  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--nSimul") == 0) {
      params.nSimul = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--S0") == 0) {
      params.S0 = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--K") == 0) {
      params.K = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--T") == 0) {
      params.T = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--r") == 0) {
      params.r = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--sigma") == 0) {
      params.sigma = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--nThreads") == 0) {
      params.nThreads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--lengthSimulation") == 0) {
      params.lengthSimulation = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0) {
      std::cout
          << "Options:\n"
          << "  --S0 <double>: Initial stock price (default 100.0)\n"
          << "  --K <double>: Strike price (default 100.0)\n"
          << "  --T <double>: Time to maturity (default 1.0)\n"
          << "  --r <double>: Risk-free rate (default 0.05)\n"
          << "  --sigma <double>: Volatility (default 0.2)\n"
          << "  --num_threads <int>: Number of threads (default 1)\n"
          << "  --numSimul <int>: Number of simulation paths (default 1e6)\n"
          << "  --length_simulation <int>: Number of time intervals (default "
             "10)\n"
          << "  --num_paths_to_plot <int>: Number of paths to plot (default "
             "10)\n"
          << "  --help (-h): Print this message\n";
      exit(0);
    }
  }
}

int main(int argc, char **argv) {
  SimulationParams params;
  parseArguments(argc, argv, params);

  omp_set_num_threads(params.nThreads);

  auto start = std::chrono::high_resolution_clock::now();
  double price = monteCarloBlackScholes(params);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Option Price: " << price << std::endl;
  std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

  return 0;
}
