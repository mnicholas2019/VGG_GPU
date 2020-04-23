#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifdef Tnn
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

VTYPE (*filters)[Ky][Kx][Nn][Ni];

VTYPE  (*data_in)[NYPAD][NXPAD][Ni];
VTYPE  (*data_out)[NYSCL][NXSCL][Nn];
VTYPE (*data_out_block)[NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&filters)[Ky][Kx][Nn][Ni],
                                    VTYPE (&data_in)[NYPAD][NXPAD][Ni],
                                    VTYPE (&data_out)[NYSCL][NXSCL][Nn],
                                    VTYPE (&data_out_block)[NYSCL][NXSCL][Nn]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          filters[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {
      for(int ni = 0; ni < Ni; ++ni) {
        data_in[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
  for(int yy = 0; yy < NYSCL; ++yy) {
    for(int xx = 0; xx < NXSCL; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        data_out[yy][xx][nn] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        data_out_block[yy][xx][nn] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

std::pair<int,int> convolution_layer_blocked(
                              VTYPE (&filters)[Ky][Kx][Nn][Ni],
                              VTYPE (&data_in)[NYPAD][NXPAD][Ni],
                              VTYPE (&data_out)[NYSCL][NXSCL][Nn]) {
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
                        VTYPE sv = filters[ky][kx][n][i];
                        VTYPE nv = data_in[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                data_out[yout][xout][n] = transfer(sum[n]);
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

void  convolution_layer(VTYPE (&filters)[Ky][Kx][Nn][Ni],
                               VTYPE (&data_in)[NYPAD][NXPAD][Ni],
                               VTYPE (&data_out)[NYSCL][NXSCL][Nn]) {
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
                VTYPE sv = filters[ky][kx][n][i];
                VTYPE nv = data_in[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          data_out[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++;
    }
    yout++;
  }
}


int main(const int argc, const char** argv) {
  cout << "allocating memory\n";
  // allocate memory on host device
  filters   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  data_in  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  data_out  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  data_out_block = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));

  // fill
  cout << "initializing arrays\n";
  fill_convolution_shared_simple(*filters,*data_in,*data_out,*data_out_block);

  cout << "starting computation\n";

  //Simple Version
  begin_roi();
  convolution_layer(*filters,*data_in,*data_out);
  end_roi();

  cout << "simple version complete!\n";


  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*filters,*data_in,*data_out_block);
  end_roi();


  cout << "blocked computation complete!\n";

  compare((VTYPE*)*data_out,(VTYPE*)*data_out_block,NYSCL*NXSCL*Nn);

  cout << "done\n";
}
