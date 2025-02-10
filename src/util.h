// util.h
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <mma.h>
#include <malloc.h>
using namespace nvcuda; 

#include "header/pets_common.h"

#define DIAGONAL do { \
    static int id = 0; \
    std::cout << "File: " << __FILE__ << ", Line: " << __LINE__ \
              << ", Column: " << __COUNTER__ << ", ID: " << id++ << std::endl; \
} while(0);

#define CU_SAFE_CALL(err) __checkCudaErrors(err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) \
{                                                                             \
  if (CUDA_SUCCESS != err)                                           \
  {                                                                           \
    const char *errorStr = NULL;                                              \
    cuGetErrorString(err, &errorStr);                                         \
    fprintf(stderr, "CU_SAFE_CALL() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, errorStr, file, line); \
    exit(EXIT_FAILURE);                                                       \
  }                                                                           \
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                                   \
  do                                                                                        \
  {                                                                                         \
    cudaError_t result = (x);                                                               \
    if (result != cudaSuccess)                                                              \
    {                                                                                       \
      const char *msg = cudaGetErrorString(result);                                         \
      std::stringstream safe_call_ss;                                                       \
      safe_call_ss << "\nerror: " #x " failed with error"                                   \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
      throw std::runtime_error(safe_call_ss.str());                                         \
    }                                                                                       \
  } while (0)


#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}
