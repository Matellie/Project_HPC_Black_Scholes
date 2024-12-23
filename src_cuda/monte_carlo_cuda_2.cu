#include <chrono>
#include <iostream>
#include <numeric>

#include <curand_kernel.h>


__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, long int random_thing) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    //curand_init(1234, id, 0, &state[id]);
    curand_init((1234 + random_thing * threadIdx.x * blockIdx.x * blockDim.x) % 14569, id, 0, &state[id]);
}


__global__ void generate_monte_carlo_bs(
    curandStatePhilox4_32_10_t* state,
    long int nbSim,
    int lengthSim,
    double* result,
    double K,
    double S0,
    double T,
    double r,
    double sigma
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
            result[i] = ST - K;
        } else {
            result[i] = 0;
        }
    }
}


__global__ void reduce_monte_carlo_bs(long int nbSim, double* gen_result, double* red_result) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int offset = nbSim / gridDim.x;

    if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
        printf("Grid dim: %d\n", gridDim.x);
    }

    for (long int stride=1; stride<offset; stride *= 2) {
        for (int i=(offset*bid+tid); i<(offset*(bid+1)); i += blockDim.x) {
            if (i % (2*stride) == 0) {
                //printf("bid:%d i:%d s:%ld result:%lf\n", bid, i, stride, gen_result[i + stride]);
                gen_result[i] += gen_result[i + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        //printf("%lf \n", gen_result[bid*offset]);
        red_result[bid] = gen_result[bid*offset];
    }

}


double monteCarloBlackScholes(
    double S0,
    double K,
    double T,
    double r,
    double sigma,
    long int nbSim,
    int lengthSim
) {
    const unsigned int threadsPerBlock = 256;
    const unsigned int blockCount = 256;
    curandStatePhilox4_32_10_t* devPHILOXStates;
    cudaMalloc((void**)&devPHILOXStates, threadsPerBlock*blockCount * sizeof(curandStatePhilox4_32_10_t));

    double* result_gen_gpu;
    double* result_red_gpu;
    double* result_cpu = new double[blockCount];
    cudaMalloc(&result_gen_gpu, nbSim * sizeof(double));
    cudaMalloc(&result_red_gpu, blockCount * sizeof(double));
    // Get cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR cuda malloc: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    auto now = std::chrono::high_resolution_clock::now();
    long int nanos = (long int)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    setup_kernel<<<blockCount, threadsPerBlock>>>(devPHILOXStates, nanos);
    // Get cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR setup kernel: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    generate_monte_carlo_bs<<<blockCount, threadsPerBlock>>>(
        devPHILOXStates, nbSim, lengthSim, result_gen_gpu, K, S0, T, r, sigma
    );
    // Get cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR monte carlo: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    reduce_monte_carlo_bs<<<256, 256>>>(nbSim, result_gen_gpu, result_red_gpu);
    // Get cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR reduction: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(result_cpu, result_red_gpu, blockCount * sizeof(double), cudaMemcpyDeviceToHost);
    // Get cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR memcopy: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    double sum = std::accumulate(result_cpu, result_cpu + blockCount, 0.0);
    double mean = sum / nbSim;

    return mean;
}


int main(int argc, char** argv) {
    // Parameters
    double S0 = 100.0;      // Initial stock price
    double K = 150.0;       // Strike price
    double T = 1.0;         // Time to maturity (1 year)
    double r = 0.05;        // Risk-free rate (5%)
    double sigma = 0.2;     // Volatility (20%)
    long int nbSim = 1e6;   // Number of simulation paths
    int lengthSim = 100;    // Number of time intervals

    // Read command line arguments.
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--nbSim") == 0) {
            nbSim = atol(argv[++i]);
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
                << "  --nbSim <int>: Number of simulation paths (default 1e6)\n"
                << "  --lengthSim <int>: Number of time intervals (default 10)\n"
                << "  --help (-h): Print this message\n";
            return 0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    double price = monteCarloBlackScholes(S0, K, T, r, sigma, nbSim, lengthSim);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Option Price: " << price << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
