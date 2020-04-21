#include <iostream>
#include "dnn.hpp"

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
   VTYPE (&data_out)[Nn], VTYPE (&data_out_tiled)[Nn]) {
 for(int n = 0; n < Nn; ++n) {
   for(int i = 0; i < Ni; ++i) {
     weights[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
   }
 }
 for(int i = 0; i < Ni; ++i) {
   data_in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
 }
 for(int n = 0; n < Nn; ++n) {
   data_out[n] = 0; //i;
   data_out_tiled[n] = 0;
 }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
 int total_calc=0;
 for (int n = 0; n < Nn; n++) {
   VTYPE temp=0;
   for (int i = 0; i < Ni; i++) {
     temp += synapse[n][i] * neuron_i[i];
   }
   neuron_n[n] = transfer(temp);
 }
}

void classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni],
                             VTYPE (&neuron_n)[Nn]) {
 int total_calc=0;
 VTYPE sum[Nn]={0};
 for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
   for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
     for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
       for (int ii = iii; ii < iii + Tii; ii += Ti) {
         // — Original code —
         for (int n = nn; n < nn + Tn; n++) {
           VTYPE sum_sc=0;
           for (int i = ii; i < ii + Ti; i++) {
             sum_sc += (synapse[n][i] * neuron_i[i]);
           }
           sum[n]+=sum_sc;
         }
       }
     }
   }
   for (int nn = nnn; nn < nnn + Tnn; nn++) {
     neuron_n[nn] = transfer(sum[nn]);
   }
 }
}

int main(int argc, char** argv) {

  if (argc == 0) {
    printf("Too few arguments, requires one argument passed.");
  } else if (argc > 1) {
    printf("Too many arguments, requires one argument passed.");
  } else if (argv[0] == 1) {
    #define Nn=4096
    #define Ni=25088
    // add tiling values here per case
  } else if (argv[1] == 2) {
    #define Nn=1024
    #define Ni=4096
    // add tiling values here per case
  } else {
    printf("Unrecognized input argument");
  }

  //Arrays:
  VTYPE weights[Nn][Ni] __attribute__((aligned(64)));
  VTYPE data_in[Ni] __attribute__((aligned(64)));
  VTYPE data_out[Nn] __attribute__((aligned(64)));
  VTYPE data_out_tiled[Nn] __attribute__((aligned(64)));

  // allocate arrays in device memory
  float *d_data_in, *d_weights, *d_data_out, *d_data_out_tiled
  cudaMalloc(&d_data_in, inputSize)
  cudaMalloc(&d_weights, weightsSize)
  cudaMalloc(&d_data_out, outputSize)
  cudaMalloc(&d_data_out_tiled, outputSize)

  cout << "initializing arrays\n";

  // fill arrays in host memory
  //fill_classifier(synapse,neuron_i,neuron_n,neuron_n2);
  fill_classifier(weights,data_in,data_out,d_data_out_tiled)

  // transfer data to device
  cudaMemcpy(d_data_in, data_in, cudaMemcpyHostToDevice)
  cudaMemcpy(d_weights, weights, cudaMemcpyHostToDevice)
  cudaMemcpy(d_data_out, data_out, cudaMemcpyHostToDevice)
  cudaMemcpy(d_data_out_tiled, data_out_tiled, cudaMemcpyHostToDevice)

  // need to define number of threads per block and number of blocks
  int blockSize = 1
  int numBlocks = 1

  cout << "starting computation\n";

  begin_roi();
  classifier_layer<<<numBlocks,blockSize>>>(d_weights,d_data_in,d_data_out);
  end_roi();

  // synchronize thread execuation and copy results back to host
  cudaDeviceSynchronize();
  cudaMemcpy(data_out, d_data_out, cudaMemcpyDeviceToHost)

  cout << "simple version complete!\n";

  begin_roi();
  classifier_layer_blocked<<<numBlocks,blockSize>>>(d_weights,d_data_in,d_data_out_tiled);
  end_roi();

  // synchronize thread execuation and copy results back to host
  cudaDeviceSynchronize();
  cudaMemcpy(data_out_tiled, d_data_out_tiled, cudaMemcpyDeviceToHost)

  cout << "blocked computation complete!\n";

  compare(data_out,data_out_tiled,Nn);

  cout << "done\n";
}