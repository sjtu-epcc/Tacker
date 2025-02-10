/*** 
 * @Author: diagonal
 * @Date: 2023-12-07 15:42:50
 * @LastEditors: diagonal
 * @LastEditTime: 2023-12-07 20:26:54
 * @FilePath: /tacker/mix_kernels/tune/fft_mrif_3_2.cu
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
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

#include "header/fft_header.h"
#include "kernel/fft_kernel.cu"

#include "header/mrif_header.h"
#include "kernel/mrif_kernel.cu"

#include "mix_kernel/fft_mrif_3_2.cu" 

int main(int argc, char* argv[]) {
    int errors = 0;

    float ori_sum_time = 0.0f;
	  // variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // fft variables
    // ---------------------------------------------------------------------------------------
		//8*1024*1024;
        int fft_blks = 3;
	    int fft_iter = 1;
		int n_bytes = FFT_N * FFT_B * sizeof(float2);
		int nthreads = FFT_T;
		srand(54321);

		float *host_shared_source =(float *)malloc(n_bytes);  
		float2 *source    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ori_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ptb_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_gptb_result	= (float2 *)malloc( n_bytes );

		for(int b=0; b<FFT_B;b++) {	
			for( int i = 0; i < FFT_N; i++ ) {
				source[b*FFT_N+i].x = (rand()/(float)RAND_MAX)*2-1;
				source[b*FFT_N+i].y = (rand()/(float)RAND_MAX)*2-1;
			}
		}

		// allocate device memory
		float2 *fft_ori_source;
		float *fft_ori_shared_source;
		cudaMalloc((void**) &fft_ori_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ori_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

		float2 *fft_ptb_source;
		float *fft_ptb_shared_source;
		cudaMalloc((void**) &fft_ptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_source, source, n_bytes, cudaMemcpyHostToDevice);

		// gptb
		float2 *fft_gptb_source;
		float *fft_gptb_shared_source;
		cudaMalloc((void**) &fft_gptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_gptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_source, source, n_bytes, cudaMemcpyHostToDevice);
    // ---------------------------------------------------------------------------------------

	// SOLO running
    // ---------------------------------------------------------------------------------------
		dim3 fft_grid, ori_fft_grid;
		dim3 fft_block, ori_fft_block;
		fft_grid.x = FFT_B;
		fft_block.x = nthreads;
		ori_fft_grid = fft_grid;
		ori_fft_block = fft_block;

		printf("[ORI] Running with fft...\n");
		printf("[ORI] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_fft<<<fft_grid, fft_block>>>(fft_ori_source, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] fft took %f ms\n\n", kernel_time);
        ori_sum_time += kernel_time;
    // ---------------------------------------------------------------------------------------
 
	// PTB running
    // ---------------------------------------------------------------------------------------
		int fft_grid_dim_x = fft_grid.x;
		int fft_block_dim_x = fft_block.x;
		fft_grid.x = fft_blks == 0 ? fft_grid_dim_x : SM_NUM * fft_blks;
		fft_block.x = fft_block_dim_x;
		printf("[PTB] Running with fft...\n");
		printf("[PTB] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_fft<<<fft_grid, fft_block>>>(fft_ptb_source, fft_grid_dim_x, fft_block_dim_x, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] fft took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // mrif variables
    // ---------------------------------------------------------------------------------------
        int mrif_blks = 3;
        int mrif_iter = 1;
        int mrif_numX, mrif_numK;		                /* Number of X and K values */
        int original_numK;		            /* Number of K values in input file */
        float *mrif_base_kx, *mrif_base_ky, *mrif_base_kz;		        /* K trajectory (3D vectors) */
        float *mrif_base_x, *mrif_base_y, *mrif_base_z;		            /* X coordinates (3D vectors) */
        float *mrif_base_phiR, *mrif_base_phiI;		            /* Phi values (complex) */
        float *mrif_base_dR, *mrif_base_dI;		                /* D values (complex) */
        float *mrif_base_realRhoPhi, *mrif_base_imagRhoPhi;     /* RhoPhi values (complex) */
        mrif_kValues* mrif_kVals;		                /* Copy of X and RhoPhi.  Its
                                            * data layout has better cache
                                            * performance. */
        inputData(
            &original_numK, &mrif_numX,
            &mrif_base_kx, &mrif_base_ky, &mrif_base_kz,
            &mrif_base_x, &mrif_base_y, &mrif_base_z,
            &mrif_base_phiR, &mrif_base_phiI,
            &mrif_base_dR, &mrif_base_dI);
        mrif_numK = original_numK;

        // createDataStructs(mrif_numK, mrif_numX, mrif_base_realRhoPhi, mrif_base_imagRhoPhi, base_outR, base_outI);
        mrif_kVals = (mrif_kValues *)calloc(mrif_numK, sizeof (mrif_kValues));
        mrif_base_realRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));
        mrif_base_imagRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));

        // kernel 1
        float *ori_phiR, *ori_phiI;
        float *ori_dR, *ori_dI;
        float *ori_realRhoPhi, *ori_imagRhoPhi;
        // kernel 2
        float *ori_x, *ori_y, *ori_z;
        float *ori_outI, *ori_outR;
        float *host_ori_outI;		            /* Output signal (complex) */

        // kernel 2
        float *ptb_x, *ptb_y, *ptb_z;
        float *ptb_outI, *ptb_outR;
        float *host_ptb_outI;		            /* Output signal (complex) */

        // gptb kernel
        float *gptb_x, *gptb_y, *gptb_z;
        float *gptb_outI, *gptb_outR;
        float *host_gptb_outI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&ori_phiR, mrif_numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, mrif_numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, mrif_numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, mrif_base_phiR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, mrif_base_phiI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, mrif_base_dR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, mrif_base_dI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&ori_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ori_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&ptb_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&gptb_x, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_y, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_z, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(gptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&gptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outI, 0, mrif_numX * sizeof(float)));

        host_ori_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_ptb_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_gptb_outI = (float*) calloc (mrif_numX, sizeof (float));
    // ---------------------------------------------------------------------------------------

    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
        // computeRhoPhi_GPU(mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
        dim3 mrif_grid1;
        dim3 mrif_block1;
        mrif_grid1.x = mrif_numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_grid1.y = 1;
        mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_block1.y = 1;
        printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \n", 
                    mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrif_grid1, mrif_block1 >>> (mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
        cudaErrCheck(cudaMemcpy(mrif_base_realRhoPhi, ori_realRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(mrif_base_imagRhoPhi, ori_imagRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < mrif_numK; k++) {
            mrif_kVals[k].Kx = mrif_base_kx[k];
            mrif_kVals[k].Ky = mrif_base_ky[k];
            mrif_kVals[k].Kz = mrif_base_kz[k];
            mrif_kVals[k].RhoPhiR = mrif_base_realRhoPhi[k];
            mrif_kVals[k].RhoPhiI = mrif_base_imagRhoPhi[k];
        }
    // ---------------------------------------------------------------------------------------
    
    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 mrif_grid2, ori_mrif_grid2;
        dim3 mrif_block2, ori_mrif_block2;
        mrif_grid2.x = mrif_numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrif_grid2.y = 1;
        mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrif_block2.y = 1;
        ori_mrif_grid2 = mrif_grid2;
        ori_mrif_block2 = mrif_block2;
        printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n", 
                    mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        mrif_kValues* mrif_kValsTile = mrif_kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, mrif_numK - FHGridBase);
        cudaMemcpyToSymbol(c, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mrif <<< mrif_grid2, mrif_block2 >>> (
                mrif_numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] mrif took %f ms\n\n", kernel_time);
        ori_sum_time += kernel_time;
    // ---------------------------------------------------------------------------------------

    // PTB running
    // ---------------------------------------------------------------------------------------
        int mrif_grid2_dim_x = mrif_grid2.x;
        int mrif_block2_dim_x = mrif_block2.x;
        mrif_grid2.x = mrif_blks == 0 ? mrif_grid2_dim_x : SM_NUM * mrif_blks;
        printf("[PTB] Running with mrif...\n");
        printf("[PTB] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n", 
            mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int task_blk_num = mrif_grid2_dim_x / 2;
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_mrif <<< mrif_grid2, mrif_block2 >>> (mrif_numK, FHGridBase, ptb_x, ptb_y, ptb_z, ptb_outR, ptb_outI, 
                                    task_blk_num, mrif_block2_dim_x,
                                    mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] mrif took %f ms\n\n", kernel_time);
        // printf("[PTB] mrif task block %d/%d\n\n", task_blk_num, mrif_grid2_dim_x);
    // ---------------------------------------------------------------------------------------



    int mix_mrif_task_blk_num = 512;
    int solo_mrif_task_blk_num = mrif_grid2_dim_x - mix_mrif_task_blk_num;
    printf("[MIX] mrif task block %d/%d\n\n", mix_mrif_task_blk_num, mrif_grid2_dim_x);


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(896, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_fft_mrif_kernel_3_2 <<<mix_kernel_grid, mix_kernel_block>>>(fft_gptb_source, 
        ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
        0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_fft_grid.x * ori_fft_grid.y * ori_fft_grid.z, mrif_numK, FHGridBase, gptb_x, gptb_y, gptb_z, gptb_outR, gptb_outI, 
            ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, mix_mrif_task_blk_num)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));

        printf("[MIX] fft_mrif 3_2 took %f ms\n", kernel_time);
  // ---------------------------------------------------------------------------------------

      float sum_kernel_time = 0.0f;
      sum_kernel_time += kernel_time;

  // 补充solo
    dim3 solo_kernel_grid = dim3(SM_NUM * 4, 1, 1);
    dim3 solo_kernel_block = mrif_block2;
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((g_general_ptb_mrif <<<solo_kernel_grid, solo_kernel_block>>>(
            mrif_numK, FHGridBase, gptb_x, gptb_y, gptb_z, gptb_outR, gptb_outI, 
            ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
            mix_mrif_task_blk_num, solo_kernel_grid.x * solo_kernel_grid.y * solo_kernel_grid.z, ori_mrif_grid2.x * ori_mrif_grid2.y * ori_mrif_grid2.z, 0)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[SOLO] mrif took %f ms\n\n", kernel_time);
    // printf("[SOLO] mrif task block %d/%d\n\n", ori_mrif_grid2.x * ori_mrif_grid2.y * ori_mrif_grid2.z - mix_mrif_task_blk_num, mrif_grid2_dim_x);

    sum_kernel_time += kernel_time;
    printf("[EVAL] ori sum time: %f ms\n", ori_sum_time);
    printf("[EVAL] fused + solo time: %f ms\n\n", sum_kernel_time);

    printf("[EVAL] improvement: %f%\n\n", ((ori_sum_time - sum_kernel_time) * 100 / ori_sum_time));



    // ---------------------------------------------------------------------------------------
	cudaMemcpy(host_fft_ori_result, fft_ori_source, n_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_fft_gptb_result, fft_gptb_source, n_bytes, cudaMemcpyDeviceToHost);
    errors = 0;
	for (int i = 0; i < FFT_N * FFT_B; i++) {
		float v1 = host_fft_ori_result[i].x;
		float v2 = host_fft_gptb_result[i].x;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\n", v1, v2);
		}
		if (i < 3) printf("%d %f %f\n", i, v1, v2);

		v1 = host_fft_ori_result[i].y;
		v2 = host_fft_gptb_result[i].y;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\n", v1, v2);
		}
	}
	if (errors > 0) {
		printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
	}
	else {
		printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
	}
    // ---------------------------------------------------------------------------------------

    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_ori_outI, ori_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_gptb_outI, gptb_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        errors = 0;
        for (int i = 0; i < mrif_numX; i++) {
            float v1 = host_ori_outI[i];
            float v2 = host_gptb_outI[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 5) printf("%f %f\n", v1, v2);
            }
            if (i < 3) printf("%d %f %f\n", i, v1, v2);
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------

}
