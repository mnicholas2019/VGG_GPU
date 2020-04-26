#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n_gpu)[NYSCL][NXSCL][Nn];
VTYPE (*test)[3211264];
// VTYPE synapse[Ky][Kx][Nn][Ni] __attribute__((aligned(64)));
// VTYPE  neuron_i[NYPAD][NXPAD][Ni] __attribute__((aligned(64)));
// VTYPE  neuron_n[NYSCL][NXSCL][Nn] __attribute__((aligned(64)));
// VTYPE neuron_n2[NYSCL][NXSCL][Nn] __attribute__((aligned(64)));
// VTYPE neuron_n_gpu[NYSCL][NXSCL][Nn] __attribute__((aligned(64)));
// VTYPE test[3211264] __attribute__((aligned(64)));

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

std::pair<int,int> convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int c1=0,c2=0;
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

__global__ void convolution_layer_gpu(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                      VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                                      VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = sum[n];
        }
      }
      xout++; 
    }
    yout++;
  }
}
// __global__ void convolution_layer_test_gpu(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
//                                       VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
//                                       VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
//   int yout = 0;
//   for (int y = 0; y < Ny; y += Sy) { // tiling for y;
//     int xout = 0;
//     for (int x = 0; x < Ny; x += Sx) { // tiling for x;
//       for (int nn = 0; nn < Nn; nn += Tn) {
//         for (int n = nn; n < nn + Tn; n++) {
//           neuron_n[yout][xout][n] = 69;
//           neuron_n[0][0][0] = 100;
//         }
//       }
//       xout++;
//     }
//     yout++;
//   }
// }

// __global__ void convolution_layer_opt_gpu(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
//                                       VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
//                                       VTYPE (&neuron_n)[NYSCL][NXSCL][Nn],
//                                       VTYPE (&test)[3211264]) {
//   __shared__ int sum[64];
//   int tid = threadIdx.x;
//   int blockId = blockIdx.x;
//   ones[tid] = 1;
//   __syncthreads();
//   int sum = 0;
//   for (int i = 0; i < blockDim.x; i++ ){
//     sum+=ones[i];
//   }
//   if (tid == 1){
//     test[blockId] = sum;
//     neuron_n[0][0][0] = sum;
//     neuron_n[0][0][1] = sum;
//     if (blockId == 0){
//       printf("Sum: %d, thread: %d, block: %d, test: %f\n", sum, tid, blockId, test[blockId]);
//     }
//     // printf("Block: %d\n", blockId);
//     // printf("Sum: %d, thread: %d, block: %d, neuron_n: %f\n", sum, tid, blockId, neuron_n[0][0][0]);
//   }
//   neuron_n[0][0][0] = 64;
//   neuron_n[0][0][1] = 64;
// }

__global__ void convolution_layer_opt_gpu(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                      VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                                      VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int blockSize = blockDim.x;
  int tid = threadIdx.x;
  int blockId = blockIdx.x;
  __shared__ float mult_adds[Ni];

  int nn = blockId/(NYSCL*NXSCL);
  int nxscl = (blockId%(NYSCL*NXSCL))/NYSCL;
  int nyscl = blockId%NYSCL;
  
  float result = 0;
  int ky = 0;
  int kx = 0;
  for (int x = nxscl; x < nxscl + Kx; x++){
    ky = 0;
    for (int y = nyscl; y < nyscl + Ky; y++){
      float value = neuron_i[y][x][tid];
      float weight = synapse[ky][kx][nn][tid];
      result += value*weight;
      // if (tid == 1 && blockId == 0){
      //   printf("Value %f, weight %f, result %f\n", value, weight, result);
      // }
      ky++;
    }
    kx++;
  }
  mult_adds[tid] = result;
  __syncthreads();

  float conv = 0;
  if (tid == 0) {
    for (int i = 0; i < blockSize; i++){
      //printf("y: %d, x: %d, nn: %d, Result: %f\n", nyscl, nxscl, nn, mult_adds[i]);
      //printf("thread: %d, block: %d\n", threadIdx.x, blockIdx.x);
      conv += mult_adds[i];
    }
    neuron_n[nyscl][nxscl][nn] = conv;
    // if (conv == 0){
    //   printf("0 at : %d,%d,%d\n", nyscl, nxscl, nn);
    // }
  }


  // int sum = 0;
  // for (int i = 0; i < blockDim.x; i++ ){
  //   sum+=ones[i];
  // }
  // if (tid == 1){
  //   test[blockId] = sum;
  //   neuron_n[0][0][0] = sum;
  //   neuron_n[0][0][1] = sum;
  //   if (blockId == 0){
  //     printf("Sum: %d, thread: %d, block: %d, test: %f\n", sum, tid, blockId, test[blockId]);
  //   }
  //   // printf("Block: %d\n", blockId);
  //   // printf("Sum: %d, thread: %d, block: %d, neuron_n: %f\n", sum, tid, blockId, neuron_n[0][0][0]);
  // }
  // neuron_n[0][0][0] = 64;
  // neuron_n[0][0][1] = 64;
}

int main(const int argc, const char** argv) {
  int cuda1 = 0;
  int cuda2 = 1;

  cout << "allocating memory\n";

  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n_gpu = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  test = (VTYPE (*)[3211264])malloc(3211264*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_shared_simple(*synapse,*neuron_i);
  // printf("Synapse: %f\n", *synapse[0][0][0][0]);
  // printf("neuron_i: %f\n", *neuron_i[0][0][0]);

  cout << "starting computation\n";

  //Simple Version
  begin_roi();
  convolution_layer(*synapse,*neuron_i,*neuron_n);
  end_roi();

  cout << "simple version complete!\n";  


  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
  end_roi();


  cout << "blocked computation complete!\n";  
  cout << "Compare two host computations: ";
  compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  // printf("neuron n: %f, neuron n2: %f\n", *neuron_n[0][0][0], *neuron_n2[0][0][0]);

   // allocate arrays in device memory for cuda kernels
  int inputSize = sizeof(VTYPE)*NYPAD*NXPAD*Ni;
  int weightsSize = sizeof(VTYPE)*Ky*Kx*Ni*Nn;
  int outputSize = sizeof(VTYPE)*NYSCL*NXSCL*Nn;
  int testSize = sizeof(VTYPE)*3211264;
  VTYPE (*d_synapse)[Ky][Kx][Nn][Ni];
  VTYPE  (*d_neuron_i)[NYPAD][NXPAD][Ni];
  VTYPE  (*d_neuron_n)[NYSCL][NXSCL][Nn];
  VTYPE  (*d_test)[3211264];

  cudaMalloc((void **)&d_neuron_i, inputSize);
  cudaMalloc((void **)&d_synapse, weightsSize);
  cudaMalloc((void **)&d_neuron_n, outputSize);
  cudaMalloc((void **)&d_test, testSize);
  
  // transfer data to device
  cudaMemcpy(d_neuron_i, neuron_i, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_synapse, synapse, weightsSize, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_neuron_n, neuron_n, outputSize, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_test, test, testSize, cudaMemcpyHostToDevice);
  
  // first cuda kernel
  if (cuda1){
  	begin_roi();
	  convolution_layer_gpu<<<1,1>>>(*d_synapse,*d_neuron_i,*d_neuron_n);
	  cudaDeviceSynchronize();
	  end_roi();

	  cout << "Cuda computation 1 complete!\n";

	  // transfre back to host
	  cudaMemcpy(neuron_n_gpu, d_neuron_n, outputSize, cudaMemcpyDeviceToHost);
    cout << "Compare cuda 1 to host: ";
    compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n_gpu,NYSCL*NXSCL*Nn);
    // for (int i = 0; i < 10; i++){
    //   printf("Neuron_n: %f\n", *neuron_n_gpu[0][0][i]);
    // }
  }
  
  // second cuda kernel
  if (cuda2){
    int blocksize = Ni;
    printf("blocksize = %d\n", Ni);
    int numblocks = NXSCL*NYSCL*Nn;
    printf("number of blocks = %d\n", numblocks);

    begin_roi();
    //convolution_layer_opt_gpu<<<numblocks,blocksize>>>(*d_synapse,*d_neuron_i,*d_neuron_n, *d_test);
    convolution_layer_opt_gpu<<<numblocks,blocksize>>>(*d_synapse,*d_neuron_i,*d_neuron_n);
    //convolution_layer_opt_gpu<<<1,1>>>(*d_synapse,*d_neuron_i,*d_neuron_n, *d_test);
    //convolution_layer_test_gpu<<<1,1>>>(*d_synapse,*d_neuron_i,*d_neuron_n);
    cudaDeviceSynchronize();
    end_roi();

    cout << "Cuda computation 2 complete!\n";

    // transfre back to host
    //cudaMemcpy(test, d_test, testSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(neuron_n_gpu, d_neuron_n, outputSize, cudaMemcpyDeviceToHost);
    cout << "Compare cuda 2 to host: ";
    compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n_gpu,NYSCL*NXSCL*Nn);

    // printf("Neuron N memcopy: %f\n", *neuron_n_gpu[0][0][0]);
    // for (int i = 0; i < blocksize; i++){
    //   printf("test %d: %f\n", i, *test[i]);
    // }
    // for (int i = 0; i < 2; i++)
    //   printf("value at %d: %f\n", i, *neuron_n_gpu[0][0][i]);
  }

  



  cout << "done\n";
}


