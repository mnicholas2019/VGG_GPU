#include <iostream>
#include <math.h>

// function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for (int i = index; i < n; i+= stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements
  int size = sizeof(float)*N;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(size);
  y = (float *)malloc(size);
  

  // Alloc device space
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy vectors from host to device
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the CPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks,256>>>(N, d_x, d_y);

  // synchronize threads before preceeding
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);

  free(x);
  free(y);

  return 0;
}