#include <iostream>
#include <math.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#define RAND_MAX=1024

#ifndef Ni
  #define Nn=1024
  #define Ni=25088
#endif

// function to fill arrays with random values
void fill_arrays(float (&weights)[Nn][Ni], float (&data_in)[Ni], float (&data_out))
{
  curandState_t state;
  /* we have to initialize the state */
  curand_init(0, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      weights[n][i] = static_cast <float> (rcurand(&state)) % static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    data_in[i] = static_cast <float> (rcurand(&state)) % static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    data_out[n] = 0.0f; //i;
  }
}

// function to add the elements of two arrays
__global__ void classifier()
{
  // compute classifer output layer
}

int main( int argc, char *argv[] )
{
  if (argc == 0) {
    printf("Too few arguments, requires one argument passed.");
  } else if (argc > 1) {
    printf("Too many arguments, requires one argument passed.");
  } else if (argv[0] == 1) {
    #define Nn=4096
    #define Ni=25088
    #define
  } else if (argv[1] == 2) {
    #define Nn=1024
    #define Ni=4096
  } else {
    printf("Unrecognized input argument");
  }

  // define general array sizes
  int inputSize = sizeof(float)*Ni;
  int weightsSize = sizeof(float)*Ni*Nn
  int outputSize = sizeof(float)*Nn;

  // instantiate array pointers
  float *data_in, *weights, *data_out
  float *d_data_in, *d_weights, *d_data_out

  // allocate arrays in host memory
  data_in = (float *)malloc(inputSize);
  weights = (float *)malloc(weightsSize);
  data_out = (float *)malloc(outputSize)

  // allocate arrays in device memory
  cudaMalloc(&d_data_in, inputSize)
  cudaMalloc(&d_weights, weightsSize)
  cudaMalloc(&d_data_out, outputSize)

  // fill arrays in host memory
  fill_arrays(weights,data_in,data_out)

  // transfer data to device
  cudaMemcpy(d_data_in, data_in, cudaMemcpyHostToDevice)
  cudaMemcpy(d_weights, weights, cudaMemcpyHostToDevice)
  cudaMemcpy(d_data_out, data_out, cudaMemcpyHostToDevice)

  // perform computation for single classifer layer
  int blockSize = 1
  int numBlocks = 1
  classifier<<<numBlocks,blockSize>>>

  // synchronize threads
  cudaDeviceSynchronize();

  // transfer data back to host
  cudaMemcpy(data_in, d_data_in, cudaMemcpyDeviceToHost)
  cudaMemcpy(weights, d_weights, cudaMemcpyDeviceToHost)
  cudaMemcpy(data_out, d_data_out, cudaMemcpyDeviceToHost)

  // Free memory on device
  cudaFree(d_data_in);
  cudaFree(d_weights);
  cudaFree(d_data_out);

  // Free memory on host
  free(data_in);
  free(weights);
  free(data_out);

  return 0;
}
