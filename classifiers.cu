#include <iostream>
#include "dnn.hpp"

#include <stdio.h>
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

using namespace std;


// this will be called in main() once we have tiling values figured out
#ifndef Tii
 // Tiling Sizes
 #define Tnn 32
 #define Tii 32
 //#define Tn 5
 //#define Ti 25
 #define Tn 16
 #define Ti 16
#endif

void fill_classifier(VTYPE (&weights)[Nn][Ni], VTYPE (&data_in)[Ni],
   VTYPE (&data_out)[Nn], VTYPE (&data_out_block)[Nn]) {

  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      //weights[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
      weights[n][i] = static_cast <float> (i*n);
    }
  }
  for(int i = 0; i < Ni; ++i) {
    //data_in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    data_in[i] = 1.0f;
  }
  for(int n = 0; n < Nn; ++n) {
    data_out[n] = 0;
    data_out_block[n] = 0;
  }
}

__global__ void classifier_layer_gpu(VTYPE *d_weights, VTYPE *d_data_in, VTYPE *d_data_out) {
  // blockDim = threads in block
  // 1 thread per output data
  // printf("Kernel called from block %d, thread %d\n", blockIdx.x, threadIdx.x);
  int ix = blockIdx.x * blockDim.x + threadIdx.x;

  VTYPE tmp = 0;
  if(ix < Nn){
    for (int n = 0; n < Ni; n++) {
      int startidx = ix * Ni;
      tmp += d_weights[startidx + n] * d_data_in[n];
    }
    d_data_out[ix] = tmp;
  }
}

// original host version of classifier layers
void classifier_layer_host(VTYPE (&weights)[Nn][Ni], VTYPE (&data_in)[Ni], VTYPE (&data_out)[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE tmp=0;
    for (int i = 0; i < Ni; i++) {
      tmp += weights[n][i] * data_in[i];
    }
    data_out[n] = transfer(tmp);
  }
}

// not yet converted to CUDA operation
void classifier_layer_blocked_host(VTYPE (&weights)[Nn][Ni], VTYPE (&data_in)[Ni],
                             VTYPE (&data_out_block)[Nn]) {
 VTYPE sum[Nn]={0};
 for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
   for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
     for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
       for (int ii = iii; ii < iii + Tii; ii += Ti) {
         // — Original code —
         for (int n = nn; n < nn + Tn; n++) {
           VTYPE sum_sc=0;
           for (int i = ii; i < ii + Ti; i++) {
             sum_sc += (weights[n][i] * data_in[i]);
           }
           sum[n]+=sum_sc;
         }
       }
     }
   }
   for (int nn = nnn; nn < nnn + Tnn; nn++) {
     data_out_block[nn] = transfer(sum[nn]);
   }
 }
}

//Arrays:
VTYPE weights[Nn][Ni] __attribute__((aligned(64)));
VTYPE data_in[Ni] __attribute__((aligned(64)));
VTYPE data_out[Nn] __attribute__((aligned(64)));
VTYPE data_out_block[Nn] __attribute__((aligned(64)));
VTYPE data_out_gpu[Nn] __attribute__((aligned(64)));

int main(int argc, char** argv) {


  cout << "initializing arrays\n";
  //fill_classifier(weights,data_in,data_out,data_out_block);
  fill_classifier(weights,data_in,data_out,data_out_block);

  cout << "Host classifier computation begin\n";
  begin_roi();
  classifier_layer_host(weights,data_in,data_out);
  end_roi();
  cout << "Host classifier computation end\n";

  cout << "blocked computation begin!\n";
  //begin_roi();
  //classifier_layer_blocked_host(weights,data_in,data_out_block);
  //end_roi();
  cout << "blocked computation complete!\n";

  // allocate arrays in device memory
  int inputSize = sizeof(VTYPE)*Ni;
  int weightsSize = sizeof(VTYPE)*Ni*Nn;
  int outputSize = sizeof(VTYPE)*Nn;
  VTYPE *d_data_in, *d_weights, *d_data_out;
  cudaMalloc(&d_data_in, inputSize);
  cudaMalloc(&d_weights, weightsSize);
  cudaMalloc(&d_data_out, outputSize);

  // transfer data to device
  cudaMemcpy(d_data_in, &data_in, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, &weights, weightsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data_out, &data_out, outputSize, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256; // threads per block
  int numBlocks = (Nn + (threadsPerBlock - 1)) / threadsPerBlock; // number of blocks

  cout << "Cuda classifier computation begin\n";
  begin_roi();
  classifier_layer_gpu<<<numBlocks,threadsPerBlock>>>(d_weights,d_data_in,d_data_out);
  cudaDeviceSynchronize();
  cudaMemcpy(&data_out_gpu, d_data_out, outputSize, cudaMemcpyDeviceToHost);
  end_roi();
  cout << "Cuda classifier computation done\n";

  compare(data_out, data_out_gpu, Nn);

  cudaFree(d_data_in);
  cudaFree(d_data_out);
  cudaFree(d_weights);

  cout << "done\n";
}
