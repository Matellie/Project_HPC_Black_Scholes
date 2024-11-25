#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <string>

// Parameters
static double S0 = 100.0;    // Initial stock price
static double K = 100.0;     // Strike price
static double T = 1.0;       // Time to maturity (1 year)
static double r = 0.05;      // Risk-free rate (5%)
static double sigma = 0.2;   // Volatility (20%)
static int numPaths = 1e6;   // Number of simulation paths
static int num_threads = 1;
static int length_simulation = 100; // Number of time intervals
static int num_paths_to_plot = 100; // Number of paths to plot

double monteCarloBlackScholes(
    double S0, double K, double T, double r, double sigma, int numPaths, int length_simulation, 
    std::vector<std::vector<double>>& pathsToPlot)
{
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    double dt = T / length_simulation; // Time step
    double payoffSum = 0.0;

    // Initialize paths to plot
    pathsToPlot.resize(num_paths_to_plot, std::vector<double>(length_simulation + 1, S0));

    #pragma omp parallel for reduction(+:payoffSum)
    for (int i = 0; i < numPaths; ++i) {
        double ST = S0; // Starting stock price
        std::vector<double> path(length_simulation + 1, S0);
        for (int j = 0; j < length_simulation; ++j) {
            double Z = dist(gen); // Standard normal random variable
            ST *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
            if (i < num_paths_to_plot) path[j + 1] = ST;
        }
        if (i < num_paths_to_plot) {
            #pragma omp critical
            pathsToPlot[i] = path;
        }
        double payoff = std::max(ST - K, 0.0); // Call option payoff
        payoffSum += payoff;
    }

    // Discounted average payoff
    return (std::exp(-r * T) * payoffSum) / numPaths;
}

int main(int argc, char **argv) {

    // Read command line arguments.
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--numPaths") == 0) {
            numPaths = atoi(argv[++i]);
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
        } else if (strcmp(argv[i], "--num_threads") == 0) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--length_simulation") == 0) {
            length_simulation = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num_paths_to_plot") == 0) {
            num_paths_to_plot = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Options:\n"
                      << "  --S0 <double>: Initial stock price (default 100.0)\n"
                      << "  --K <double>: Strike price (default 100.0)\n"
                      << "  --T <double>: Time to maturity (default 1.0)\n"
                      << "  --r <double>: Risk-free rate (default 0.05)\n"
                      << "  --sigma <double>: Volatility (default 0.2)\n"
                      << "  --num_threads <int>: Number of threads (default 1)\n"
                      << "  --numPaths <int>: Number of simulation paths (default 1e6)\n"
                      << "  --length_simulation <int>: Number of time intervals (default 10)\n"
                      << "  --num_paths_to_plot <int>: Number of paths to plot (default 10)\n"
                      << "  --help (-h): Print this message\n";
            return 0;
        }
    }

    omp_set_num_threads(num_threads);

    std::vector<std::vector<double>> pathsToPlot;
    auto start = std::chrono::high_resolution_clock::now();
    double price = monteCarloBlackScholes(S0, K, T, r, sigma, numPaths, length_simulation, pathsToPlot);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Option Price: " << price << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

    // Extract the program name
    std::string commandLine;
    for (int i = 0; i < argc; ++i) {
        if (!commandLine.empty()) commandLine += "_";
        commandLine += argv[i];
    }
    std::replace(commandLine.begin(), commandLine.end(), ' ', '_');
    std::string outputFilename = commandLine + "_paths.csv";

    // Write paths to a CSV file for plotting
    std::ofstream outFile(outputFilename);
    for (size_t i = 0; i < pathsToPlot[0].size(); ++i) {
        for (size_t j = 0; j < pathsToPlot.size(); ++j) {
            outFile << pathsToPlot[j][i];
            if (j < pathsToPlot.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
    std::cout << "Paths saved to " << outputFilename << ". Use Python or another tool to visualize.\n";

    return 0;
}

