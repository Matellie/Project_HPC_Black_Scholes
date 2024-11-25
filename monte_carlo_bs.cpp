#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstring>

// Parameters
static double S0 = 100.0;    // Initial stock price
static double K = 100.0;     // Strike price
static double T = 1.0;       // Time to maturity (1 year)
static double r = 0.05;      // Risk-free rate (5%)
static double sigma = 0.2;   // Volatility (20%)
static int numPaths = 1e6;   // Number of simulation paths
static int num_threads = 1;

double monteCarloBlackScholes(
    double S0, double K, double T, double r, double sigma, int numPaths)
{
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    double payoffSum = 0.0;
    
    #pragma omp parallel for
    for (int i = 0; i < numPaths; ++i) {
        // Simulate one path
        double Z = dist(gen); // Standard normal random variable
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(ST - K, 0.0); // Call option payoff
        payoffSum += payoff;
    }

    // Discounted average payoff
    return (std::exp(-r * T) * payoffSum) / numPaths;
}

int main(int argc, char **argv) {

    // Read command line arguments.
    for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "--numPaths" ) == 0 )) {
            numPaths = atoi( argv[ ++i ] );
            printf( "  User numPaths is %d\n", numPaths );
        } else if ( ( strcmp( argv[ i ], "--S0" ) == 0 )) {
            S0 = strtol( argv[ ++i ] , NULL, 10 );
            printf( "  User S0 is %lf\n", S0 );
        } else if ( ( strcmp( argv[ i ], "--K" ) == 0 )) {
            K = strtol( argv[ ++i ] , NULL, 10 );
            printf( "  User K is %lf\n", K );
        } else if ( ( strcmp( argv[ i ], "--T" ) == 0 )) {
            T = strtol( argv[ ++i ] , NULL, 10 );
            printf( "  User T is %lf\n", T );
        } else if ( ( strcmp( argv[ i ], "--r" ) == 0 )) {
            r = strtol( argv[ ++i ] , NULL, 10 );
            printf( "  User r is %lf\n", r );
        } else if ( ( strcmp( argv[ i ], "--sigma" ) == 0 )) {
            sigma = strtol( argv[ ++i ] , NULL, 10 );
            printf( "  User sigma is %lf\n", sigma );
        } else if ( ( strcmp( argv[ i ], "--num_threads" ) == 0 )) {
            num_threads = atoi( argv[ ++i ] );
            printf( "  User num_threads is %d\n", num_threads );
        } else if ( ( strcmp( argv[ i ], "--h" ) == 0 ) || ( strcmp( argv[ i ], "--help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  --S0 <int>:              Initial stock price (by default 100.0)\n" );
            printf( "  --K <int>:              Strike price (by default 100.0)\n" );
            printf( "  --T <int>:              Time to maturity (by default 1 year)\n" );
            printf( "  --r <int>:              Risk-free rate (by default 0.05)\n" );
            printf( "  --sigma <int>:              Volatility (by default 0.2)\n" );
            printf( "  --num_threads <int>:              Number of Threads (by default 1)\n" );           
            printf( "  --numPaths <int>:              Number of simulation paths (by default 1e6)\n" );            
            printf( "  --help (-h):            print this message\n\n" );
            exit( 1 );
        }
    }

    omp_set_num_threads(num_threads);

    auto start = std::chrono::high_resolution_clock::now();
    double price = monteCarloBlackScholes(S0, K, T, r, sigma, numPaths);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Option Price: " << price << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}

