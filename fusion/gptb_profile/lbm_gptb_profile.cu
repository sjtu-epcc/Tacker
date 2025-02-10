
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

#include "header/lbm_header.h"
#include "kernel/lbm_kernel.cu"

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

    // lbm variables
    // ---------------------------------------------------------------------------------------
        int lbm_blks = 1;
        int lbm_iter = 1;
        float *lbm_ori_src;
        float *lbm_ori_dst;
        float *lbm_ptb_src;
        float *lbm_ptb_dst;
        float *lbm_gptb_src;
        float *lbm_gptb_dst;
        float *host_lbm_ori_dst;
        float *host_lbm_ptb_dst;
        float *host_lbm_gptb_dst;

        const size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

        host_lbm_ori_dst = (float *)malloc(size);
        host_lbm_ptb_dst = (float *)malloc(size);
        host_lbm_gptb_dst = (float *)malloc(size);
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_dst, size));

        curandGenerator_t lbm_gen;
        curandErrCheck(curandCreateGenerator(&lbm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(lbm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        cudaErrCheck(cudaMemcpy(lbm_ptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        lbm_ori_src += REAL_MARGIN;
        lbm_ori_dst += REAL_MARGIN;
        lbm_ptb_src += REAL_MARGIN;
        lbm_ptb_dst += REAL_MARGIN;
        lbm_gptb_src += REAL_MARGIN;
        lbm_gptb_dst += REAL_MARGIN;

    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 lbm_block, lbm_grid, ori_lbm_block, ori_lbm_grid;
        lbm_block.x = SIZE_X;
        lbm_grid.x = SIZE_Y;
        lbm_grid.y = SIZE_Z;
        lbm_block.y = lbm_block.z = lbm_grid.z = 1;
        ori_lbm_block = lbm_block;
        ori_lbm_grid = lbm_grid;
        printf("[ORI] Running with lbm...\n");
        printf("[ORI] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block>>>(lbm_ori_src, lbm_ori_dst, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB running
    // ---------------------------------------------------------------------------------------
        int lbm_block_dim_x = lbm_block.x;
        int lbm_block_dim_y = lbm_block.y;
        int lbm_block_dim_z = lbm_block.z;
        int lbm_grid_dim_x = lbm_grid.x;
        int lbm_grid_dim_y = lbm_grid.y;
        int lbm_grid_dim_z = lbm_grid.z;

        lbm_grid.x = lbm_blks == 0 ? lbm_grid_dim_x * lbm_grid_dim_y : SM_NUM * lbm_blks;
        lbm_grid.y = lbm_grid.z = 1;
        lbm_block.x = lbm_block_dim_x * lbm_block_dim_y * lbm_block_dim_z;
        lbm_block.y = lbm_block.z = 1;
        printf("[PTB] Running with lbm...\n");
        printf("[PTB] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_lbm<<<lbm_grid, lbm_block>>>(lbm_ptb_src, lbm_ptb_dst,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    std::vector<float> time_vec;
    // GPTB
    // ---------------------------------------------------------------------------------------
        dim3 gptb_kernel_grid = dim3(68, 1, 1);
        dim3 gptb_kernel_block = dim3(128, 1, 1);
        for(int i = 0; i < 30; ++i) {
            cudaErrCheck(cudaEventRecord(startKERNEL));
            checkKernelErrors((g_general_ptb_lbm <<<gptb_kernel_grid, gptb_kernel_block>>>(lbm_gptb_src, lbm_gptb_dst,
    ori_lbm_grid.x, ori_lbm_grid.y, ori_lbm_grid.z, ori_lbm_block.x, ori_lbm_block.y, ori_lbm_block.z,
    0, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, 16320, 0)));
            cudaErrCheck(cudaEventRecord(stopKERNEL));
            cudaErrCheck(cudaEventSynchronize(stopKERNEL));
            cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
        }

        // sort & get average
        std::sort(time_vec.begin(), time_vec.end());
        float gptb_lbm_time = 0.0f;
        for(int i = 10; i < 20; ++i) {
            gptb_lbm_time += time_vec[i];
        }
        gptb_lbm_time /= 10.0f;
        time_vec.clear();
        printf("[GPTB] lbm took %f ms\n", gptb_lbm_time);
        printf("[GPTB] lbm blks: %d\n\n", 16320 - 0);

        printf("---------------------------\n");

}
