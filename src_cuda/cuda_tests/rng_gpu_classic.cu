/*
 * This program uses the device CURAND API to calculate what
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_normal_kernel(curandStatePhilox4_32_10_t *state, int n, float *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random normals */
    x = curand_normal(&localState);
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] = x;
}


int main(int argc, char *argv[])
{
    const unsigned int threadsPerBlock = 2;
    const unsigned int blockCount = 2;
    const unsigned int totalThreads = threadsPerBlock * blockCount;

    unsigned int i;
    curandStatePhilox4_32_10_t *devPHILOXStates;
    float *devResults, *hostResults;
    int sampleCount = 10000;

    /* Allocate space for results on host */
    hostResults = (float *)calloc(totalThreads, sizeof(float));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, totalThreads *sizeof(float)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads *sizeof(float)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devPHILOXStates, totalThreads *sizeof(curandStatePhilox4_32_10_t)));

    /* Setup prng states */
    setup_kernel<<<64, 64>>>(devPHILOXStates);



    /* Generate and use normal pseudo-random  */
    generate_normal_kernel<<<blockCount, threadsPerBlock>>>(devPHILOXStates, sampleCount, devResults);

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads *sizeof(float), cudaMemcpyDeviceToHost));

    /* Show result */
    for(i = 0; i < totalThreads; i++) {
        printf("%f ", hostResults[i]);
    }
    printf("\n");

    /* Cleanup */
    CUDA_CALL(cudaFree(devPHILOXStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}