
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
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
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

#include <mma.h>
using namespace nvcuda; 
#include "header/stencil_header.h"
#include "kernel/stencil_kernel.cu"

int main(int argc, char* argv[]) {
    int errors = 0;

	  // variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    int stencil_blks = 3;
    int stencil_iter = 1;
    // stencil variables
    // ---------------------------------------------------------------------------------------
        float *host_stencil_ori_a0;
        float *stencil_ori_a0;
        float *stencil_ori_anext;

        float *host_stencil_ptb_a0;
        float *stencil_ptb_a0;
        float *stencil_ptb_anext;

        float *host_stencil_gptb_a0;
        float *stencil_gptb_a0;
        float *stencil_gptb_anext;

        float c0=1.0f/6.0f;
        float c1=1.0f/6.0f/6.0f;

        // nx = 128 ny = 128 nz = 32 iter = 100
        // nx = 512 ny = 512 nz = 64 iter = 100
        int nx = 128 * 4;
        int ny = 128 * 4;
        int nz = 32 * 2;
        // int nz = 16 * 1;
        
        // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
        host_stencil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_anext, nx * ny * nz * sizeof(float)));

        host_stencil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_anext, nx * ny * nz * sizeof(float)));

        host_stencil_gptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_gptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_gptb_anext, nx * ny * nz * sizeof(float)));

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, stencil_ori_a0, nx * ny * nz));
        cudaErrCheck(cudaMemcpy(stencil_ori_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_gptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_gptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 stencil_grid, ori_stencil_grid;
        dim3 stencil_block, ori_stencil_block;
        stencil_block.x = tile_x;
        stencil_block.y = tile_y;
        stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
        stencil_grid.y = (ny + tile_y - 1) / tile_y;
        ori_stencil_block = stencil_block;
        ori_stencil_grid = stencil_grid;

        printf("[ORI] Running with stencil...\n");
        printf("[ORI] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ori_a0, stencil_ori_anext, nx, ny, nz, stencil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] stencil took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    std::vector<float> time_vec;
    // GPTB
    // ---------------------------------------------------------------------------------------
        dim3 gptb_kernel_grid = dim3(204, 1, 1);
        dim3 gptb_kernel_block = dim3(128, 1, 1);
        for(int i = 0; i < 30; ++i) {
            cudaErrCheck(cudaEventRecord(startKERNEL));
            checkKernelErrors((g_general_ptb_stencil <<<gptb_kernel_grid, gptb_kernel_block>>>(c0, c1, stencil_gptb_a0, stencil_gptb_anext, nx, ny, nz,
    ori_stencil_grid.x, ori_stencil_grid.y, ori_stencil_grid.z, ori_stencil_block.x, ori_stencil_block.y, ori_stencil_block.z,
    0, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, 1020, 0)));
            cudaErrCheck(cudaEventRecord(stopKERNEL));
            cudaErrCheck(cudaEventSynchronize(stopKERNEL));
            cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
        }

        // sort & get average
        std::sort(time_vec.begin(), time_vec.end());
        float gptb_stencil_time = 0.0f;
        for(int i = 10; i < 20; ++i) {
            gptb_stencil_time += time_vec[i];
        }
        gptb_stencil_time /= 10.0f;
        time_vec.clear();
        printf("[GPTB] stencil took %f ms\n", gptb_stencil_time);
        printf("[GPTB] stencil blks: %d\n\n", 1020 - 0);

        printf("---------------------------\n");

}
