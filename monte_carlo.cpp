#include <cstdio>
#include <random>
#include <sys/time.h>
#include <omp.h>
#include "include/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {
    printf("Program begins here!\n");

    // Initialize variables
    size_t length = 101;
    size_t length_sim = 1000;
    int nb_iter = 2000000;
    double mean = 0.0;
    double stddev = 1.0;

    // Initialize random normal distribution
    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<double> d(mean, stddev);

    // Initatialize time arrays
    std::vector<double> time(length);
    std::vector<double> time_sim(length_sim);
    for(int i=0; i<length; ++i){time.at(i) = i;}
    for(int i=0; i<length_sim; ++i){time_sim.at(i) = length - 1 + i;}

    // Generate random walk before prediction
    std::vector<double> market_price(length);
    market_price.at(0) = 50;
    for(int i=1; i<length; ++i){market_price.at(i) = market_price.at(i-1) + d(gen);}
    plt::plot(time, market_price, "Black");


    // Timer products.
    struct timeval begin, end;
    gettimeofday( &begin, NULL );

    // Compute predictions
    #pragma parallel for
    for(int i=0; i<nb_iter; ++i){
        std::vector<double> cum_arr(length_sim);

        cum_arr.at(0) = market_price.at(length-1);
        for(int j=1; j<length_sim; ++j) {
            cum_arr.at(j) = cum_arr.at(j-1) + d(gen);
        }

        //plt::plot(time_sim, cum_arr);
    }

    gettimeofday( &end, NULL );
    // Calculate time.
    double exec_time = 1.0 * ( end.tv_sec - begin.tv_sec ) + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    printf("Time: %lf s\n", exec_time);


    plt::grid(true);
    plt::title("Price as a function of time");
    plt::xlabel("Time");
    plt::ylabel("Price");
    plt::show();

    printf("Program ends here!\n");
    return 0;
}
