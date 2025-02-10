#define BLOCKSIZEX 16
#define BLOCKSIZEY 4
#define UNROLLX 8

#define MAXATOMS 64
// #define MAXATOMS 4000
#define VOLSIZEX 4096
#define VOLSIZEY 4096
// #define ATOMCOUNT 40000
#define ATOMCOUNT 4000

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
__constant__ float4 atominfo[MAXATOMS];

#include "pets_common.h"
#define CP_GRID_DIM (SM_NUM * 6)