#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

struct mriq_kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

__constant__ __device__ mriq_kValues ck[KERNEL_Q_K_ELEMS_PER_GRID];

#include "pets_common.h"
#define MRIQ_GRID_DIM (SM_NUM * 1)