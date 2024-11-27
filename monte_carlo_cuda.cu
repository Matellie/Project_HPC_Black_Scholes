#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// Parameters
static double S0 = 100.0;   // Initial stock price
static double K = 150.0;    // Strike price
static double T = 1.0;      // Time to maturity (1 year)
static double r = 0.05;     // Risk-free rate (5%)
static double sigma = 0.2;  // Volatility (20%)
static int nbSim = 1e6;     // Number of simulation paths
static int lengthSim = 100; // Number of time intervals


__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, long int random_thing) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init((1234 + random_thing*threadIdx.x*blockIdx.x*blockDim.x)%14569, id, 0, &state[id]);
}


__global__ void generate_monte_carlo_bs(
    curandStatePhilox4_32_10_t *state,
    long int nbSim,
    int lengthSim,
    double* result,
    double K,
    double S0 = 100.0,  // Initial stock price
    double T = 1.0,     // Time to maturity (1 year)
    double r = 0.05,    // Risk-free rate (5%)
    double sigma = 0.2  // Volatility (20%)
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t localState = state[id];

    double dt = T / lengthSim;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * sqrt(dt);

    for (int i=id; i<nbSim; i+=blockDim.x*gridDim.x) {
        double ST = S0;
        for (int j=0; j<lengthSim; ++j) {
            double r = curand_normal_double(&localState);
            ST *= exp(drift + diffusion * r);
            state[id] = localState;
        }
        if (ST > K) {
            result[i] = ST;
        } else {
            result[i] = 0;
        }
    }

}


// This function will fill the array simulationsToPlot with the intermediate
// values from the first nSimulToPlot simulations to allow for output to a csv
// file. For the rest, it will only compute the final value of the simulation
// (at time T).
double monteCarloBlackScholes(
    double S0,
    double K,
    double T,
    double r,
    double sigma,
    int nbSim,
    int lengthSim
) {
    const unsigned int threadsPerBlock = 256;
    const unsigned int blockCount = 256;
    curandStatePhilox4_32_10_t *devPHILOXStates;
    cudaMalloc((void **)&devPHILOXStates, threadsPerBlock*blockCount * sizeof(curandStatePhilox4_32_10_t));

    long int nb_sim = 1000000000;
    int length_sim = 100;
    double *result_gpu;
    double* result_cpu = new double[nb_sim];

    cudaMalloc(&result_gpu, nb_sim * sizeof(double));

    auto now = std::chrono::high_resolution_clock::now();
    long int nanos = (long int)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    auto start = std::chrono::high_resolution_clock::now();
    setup_kernel<<<blockCount, threadsPerBlock>>>(devPHILOXStates, nanos);
    printf("Kernel set up\n");

    generate_monte_carlo_bs<<<blockCount, threadsPerBlock>>>(
        devPHILOXStates, nb_sim, length_sim, result_gpu, K
    );
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("Monte carlo generated in: %lf seconds\n", elapsed.count());

    cudaMemcpy(result_cpu, result_gpu, nb_sim* sizeof(double), cudaMemcpyDeviceToHost);
    printf("Cuda memcopy done\n");

    double sum = std::accumulate(result_cpu, result_cpu + nb_sim, 0.0);
    double mean = sum / nb_sim;
    printf("Mean computed\n");

    return mean;
}


int main(int argc, char **argv) {
    // Read command line arguments.
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--nbSim") == 0) {
            nbSim = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--lengthSim") == 0) {
            lengthSim = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--S0") == 0) {
            S0 = strtod(argv[++i], nullptr);
        }
        else if (strcmp(argv[i], "--K") == 0) {
            K = strtod(argv[++i], nullptr);
        }
        else if (strcmp(argv[i], "--T") == 0) {
            T = strtod(argv[++i], nullptr);
        }
        else if (strcmp(argv[i], "--r") == 0) {
            r = strtod(argv[++i], nullptr);
        }
        else if (strcmp(argv[i], "--sigma") == 0) {
            sigma = strtod(argv[++i], nullptr);
        }
        else if (strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Options:\n"
                << "  --S0 <double>: Initial stock price (default 100.0)\n"
                << "  --K <double>: Strike price (default 100.0)\n"
                << "  --T <double>: Time to maturity (default 1.0)\n"
                << "  --r <double>: Risk-free rate (default 0.05)\n"
                << "  --sigma <double>: Volatility (default 0.2)\n"
                << "  --num_threads <int>: Number of threads (default 1)\n"
                << "  --numSimul <int>: Number of simulation paths (default 1e6)\n"
                << "  --length_simulation <int>: Number of time intervals (default 10)\n"
                << "  --num_paths_to_plot <int>: Number of paths to plot (default 10)\n"
                << "  --help (-h): Print this message\n";
            return 0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    double price = monteCarloBlackScholes(
        S0, K, T, r, sigma, nbSim, lengthSim
    );
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Option Price: " << price << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
