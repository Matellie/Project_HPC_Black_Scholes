#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

// Parameters
static double S0 = 100.0;  // Initial stock price
static double K = 100.0;   // Strike price
static double T = 1.0;     // Time to maturity (1 year)
static double r = 0.05;    // Risk-free rate (5%)
static double sigma = 0.2; // Volatility (20%)
static int nSimul = 1e6;   // Number of simulation paths
static int nThreads = 1;
static int lengthSimulation = 100; // Number of time intervals
static int nSimulToPlot = 100;     // Number of paths to plot

// This function will fill the array simulationsToPlot with the intermediate
// values from the first nSimulToPlot simulations to allow for output to a csv
// file. For the rest, it will only compute the final value of the simulation
// (at time T).
double
monteCarloBlackScholes(double S0, double K, double T, double r, double sigma,
                       int nSimul, int lengthSimulation,
                       std::vector<std::vector<double>> &simulationsToPlot) {
  // Random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, 1.0);

  double dt = T / lengthSimulation; // Time step
  double payoffSum = 0.0;           // sum all the payoffs to compute average

  // Initialize paths to plot
  simulationsToPlot.resize(nSimulToPlot,
                           std::vector<double>(lengthSimulation + 1, S0));

#pragma omp parallel for reduction(+ : payoffSum)
  for (int i = 0; i < nSimul; ++i) {
    double ST = S0; // Starting stock price
    std::vector<double> path(lengthSimulation + 1, S0);
    for (int j = 0; j < lengthSimulation; ++j) {
      double Z = dist(gen); // Standard normal random variable
      ST *=
          std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
      if (i < nSimulToPlot)
        path[j + 1] = ST;
    }
    if (i < nSimulToPlot) {
#pragma omp critical
      simulationsToPlot[i] = path;
    }
    double payoff = std::max(ST - K, 0.0); // Call option payoff
    payoffSum += payoff;
  }

  // Discounted average payoff
  return (std::exp(-r * T) * payoffSum) / nSimul;
}

int main(int argc, char **argv) {

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--nSimul") == 0) {
      nSimul = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--S0") == 0) {
      S0 = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--K") == 0) {
      K = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--T") == 0) {
      T = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--r") == 0) {
      r = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--sigma") == 0) {
      sigma = strtod(argv[++i], nullptr);
    } else if (strcmp(argv[i], "--nThreads") == 0) {
      nThreads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--lengthSimulation") == 0) {
      lengthSimulation = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--nSimulToPlot") == 0) {
      nSimulToPlot = atoi(argv[++i]);
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
      return 0;
    }
  }

  omp_set_num_threads(nThreads);

  std::vector<std::vector<double>> simulationsToPlot;
  auto start = std::chrono::high_resolution_clock::now();
  double price = monteCarloBlackScholes(S0, K, T, r, sigma, nSimul,
                                        lengthSimulation, simulationsToPlot);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Option Price: " << price << std::endl;
  std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

  // Extract the program name
  std::string commandLine;
  for (int i = 0; i < argc; ++i) {
    if (!commandLine.empty())
      commandLine += "_";
    commandLine += argv[i];
  }
  std::replace(commandLine.begin(), commandLine.end(), ' ', '_');
  std::string outputFilename = commandLine + "_paths.csv";

  // Write paths to a CSV file for plotting
  std::ofstream outFile(outputFilename);
  for (size_t i = 0; i < simulationsToPlot[0].size(); ++i) {
    for (size_t j = 0; j < simulationsToPlot.size(); ++j) {
      outFile << simulationsToPlot[j][i];
      if (j < simulationsToPlot.size() - 1)
        outFile << ",";
    }
    outFile << "\n";
  }
  outFile.close();
  std::cout << "Paths saved to " << outputFilename
            << ". Use Python or another tool to visualize.\n";

  return 0;
}
