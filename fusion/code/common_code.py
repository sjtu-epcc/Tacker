'''
Author: diagonal
Date: 2023-11-15 22:33:21
LastEditors: diagonal
LastEditTime: 2023-11-18 21:29:46
FilePath: /tacker/mix_kernels/code/common_code.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''

common_header = """
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
using namespace nvcuda; 

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\\n", stat, file, line);
   }
}

#define checkKernelErrors(expr)                             \\
  do {                                                      \\
    expr;                                                   \\
                                                            \\
    cudaError_t __err = cudaGetLastError();                 \\
    if (__err != cudaSuccess) {                             \\
      printf("Line %d: '%s' failed: %s\\n", __LINE__, #expr, \\
             cudaGetErrorString(__err));                    \\
      abort();                                              \\
    }                                                       \\
  } while (0)
"""

time_event_create_code = """
	  // variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------
"""

main_func_begin_code = """
int main(int argc, char* argv[]) {
    int errors = 0;
"""

main_func_end_code = """
}
"""