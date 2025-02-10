#pragma once
#include <endian.h>
#include <malloc.h>
#include <inttypes.h>

#define PI   3.1415926535897932384626433832795029
#define PIx2 6.2831853071795864769252867665590058

/* Adjustable parameters */
#define KERNEL_RHO_PHI_THREADS_PER_BLOCK 256
#define KERNEL_FH_THREADS_PER_BLOCK 256
#define KERNEL_FH_K_ELEMS_PER_GRID 256

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#define CUDA_ERRCK							\
  {cudaError_t err;							\
    if ((err = cudaGetLastError()) != cudaSuccess) {			\
      fprintf(stderr, "CUDA error on line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
      exit(-1);								\
    }									\
  }

struct mrif_kValues {
  float Kx;
  float Ky;
  float Kz;
  float RhoPhiR;
  float RhoPhiI;
};

struct mrif_kValues_int {
  int Kx;
  int Ky;
  int Kz;
  int RhoPhiR;
  int RhoPhiI;
};

#include "pets_common.h"
#define MRIF_GRID_DIM (SM_NUM * 1)
