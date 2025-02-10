
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

#include "header/cp_header.h"
#include "kernel/cp_kernel.cu"

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/cp_sgemm_1_1.cu" 
#include <vector>

#define VOLSIZEX 40960
#define VOLSIZEY 4096
#define ATOMCOUNT 4000

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

    // cp variables
    // ---------------------------------------------------------------------------------------
		int cp_blks = 8;
	    int cp_iter = 1;
        float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

        float *ori_output;	
		float *ptb_output;
		float *gptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
		cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
		float *host_gptb_energy = (float *) malloc(volmemsz);

        dim3 cp_grid, cp_block, ori_cp_grid, ori_cp_block;
        int atomstart = 1;
		int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		ori_cp_grid = cp_grid;
		ori_cp_block = cp_block;
		printf("[ORI] Running with cp...\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

        //printf("runatoms: %d\n", runatoms);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\n\n", kernel_time);
        ori_sum_time += kernel_time;

        cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

	// PTB running
    // ---------------------------------------------------------------------------------------
        int solo_ptb_cp_blks = 6;
	    cp_iter = 1;
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		cp_grid.x = solo_ptb_cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * solo_ptb_cp_blks;
		cp_grid.y = 1;
		//printf("[PTB] Running with cp...\n");
		//printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					// cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

        //printf("runatoms: %d\n", runatoms);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb2_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		//printf("[PTB] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

    // sgemm variables
    // ---------------------------------------------------------------------------------------
        int sgemm_blks = 4;
        int sgemm_iter = 1;
        float *sgemm_ori_a;
        float *sgemm_ori_b;
        float *sgemm_ori_c;
        float *sgemm_ptb_a;
        float *sgemm_ptb_b;
        float *sgemm_ptb_c;
        float *sgemm_gptb_a;
        float *sgemm_gptb_b;
        float *sgemm_gptb_c;
        float *host_sgemm_ori_c;
        float *host_sgemm_ptb_c;
        float *host_sgemm_gptb_c;
        

        // parallel experiment
        int NORMAL_M = 4096;
        int NORMAL_N = 4128;
        int NORMAL_K = 4064;

        NORMAL_M = (NORMAL_M / 10) * sgemm_iter;

        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

        host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_gptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

        curandGenerator_t sgemm_gen;
        curandErrCheck(curandCreateGenerator(&sgemm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(sgemm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        curandErrCheck(curandDestroyGenerator(sgemm_gen));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 sgemm_grid, ori_sgemm_grid;
        dim3 sgemm_block, ori_sgemm_block;
        sgemm_block.x = TILE_N;
        sgemm_block.y = TILE_TB_HEIGHT;
        sgemm_grid.x = NORMAL_M/TILE_M;
        sgemm_grid.y = NORMAL_N/TILE_N;
        ori_sgemm_grid = sgemm_grid;
        ori_sgemm_block = sgemm_block;
        printf("[ORI] Running with sgemm...\n");
        printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] sgemm took %f ms\n\n", kernel_time);
        ori_sum_time += kernel_time;
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int sgemm_grid_dim_x = sgemm_grid.x;
        int sgemm_grid_dim_y = sgemm_grid.y;
        // int sgemm_block_dim_x = sgemm_block.x;
        // int sgemm_block_dim_y = sgemm_block.y;
        sgemm_grid.x = sgemm_grid_dim_x * sgemm_grid_dim_y;
        sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : 68 * sgemm_blks;
        sgemm_grid.y = 1;
        // sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
        // sgemm_block.y = 1;
        //printf("[PTB] Running with sgemm...\n");
        //printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            // sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb2_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K,
                    sgemm_grid_dim_x, sgemm_grid_dim_y, 
                    1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        //printf("[PTB] sgemm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    int mix_cp_task_blk_num = 11264;
    int solo_cp_task_blk_num = cp_grid_dim_x * cp_grid_dim_y - mix_cp_task_blk_num;
    //printf("mix_cp_task_blk_num: %d\n", mix_cp_task_blk_num);
    //printf("solo_cp_task_blk_num: %d\n", solo_cp_task_blk_num);

    std::vector<float> time_vec;
    
    // gptb cp
    dim3 gptb_cp_grid = dim3(SM_NUM * cp_blks, 1, 1);
    dim3 gptb_cp_block = dim3(128, 1, 1);
    // warmup
    for(int i = 0; i < 20; ++i) {
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((g_general_ptb_cp <<<gptb_cp_grid, gptb_cp_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z,
        0, gptb_cp_grid.x * gptb_cp_grid.y * gptb_cp_grid.z, mix_cp_task_blk_num, 0)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float gptb_cp_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_cp_time += time_vec[i];
    }
    gptb_cp_time /= 10.0f;
    

    // float gptb_cp_time = kernel_time;
    // gptb sgemm

        time_vec.clear();

        dim3 gptb_sgemm_grid = dim3(SM_NUM * 4, 1, 1);
        dim3 gptb_sgemm_block = dim3(128, 1, 1);
        // warmup
        for(int i = 0; i < 20; ++i) {
            cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((general_ptb_sgemm <<< gptb_sgemm_grid, gptb_sgemm_block >>> (
                    sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K,
                    ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, gptb_sgemm_grid.x * gptb_sgemm_grid.y * gptb_sgemm_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z, 0)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
        }

        // 排序后取中间10个数据，计算平均值
        std::sort(time_vec.begin(), time_vec.end());
        float gptb_sgemm_time = 0.0f;
        for(int i = 5; i < 15; ++i) {
            gptb_sgemm_time += time_vec[i];
        }
        gptb_sgemm_time /= 10.0f;


        // float gptb_sgemm_time = kernel_time;


  // MIX
  // ---------------------------------------------------------------------------------------

        atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);
        //printf("runatoms: %d\n", runatoms);

        dim3 mix_kernel_grid = dim3(272, 1, 1);
        dim3 mix_kernel_block = dim3(256, 1, 1);

        time_vec.clear();

        // warmup
        for(int i = 0; i < 50; ++i) {
            cudaErrCheck(cudaEventRecord(startKERNEL));
            checkKernelErrors((mixed_cp_sgemm_kernel_1_1 <<<mix_kernel_grid, mix_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
        0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, mix_cp_task_blk_num, 
        sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
                ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
                0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
            cudaErrCheck(cudaEventRecord(stopKERNEL));
            cudaErrCheck(cudaEventSynchronize(stopKERNEL));
            cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
        }

        // 排序后取中间30个数据，计算平均值
        std::sort(time_vec.begin(), time_vec.end());
        float mix_time = 0.0f;
        for(int i = 10; i < 40; ++i) {
            mix_time += time_vec[i];
        }
        mix_time /= 30.0f;


        // float mix_time = kernel_time;
  // ---------------------------------------------------------------------------------------

    float sum_kernel_time = 0.0f;
    sum_kernel_time += kernel_time;


	// copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

    // 补充cp solo
    //printf("[SOLO] Running with cp...\n");
    dim3 solo_kernel_grid = dim3(SM_NUM * cp_blks, 1, 1);
    dim3 solo_kernel_block = dim3(128, 1, 1);
    //printf("[SOLO] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
               // solo_kernel_grid.x, solo_kernel_grid.y, solo_kernel_grid.z, solo_kernel_block.x, solo_kernel_block.y, solo_kernel_block.z);
    time_vec.clear();
    for(int i = 0; i < 20; ++i) {
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((g_general_ptb_cp <<<solo_kernel_grid, solo_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z,
        mix_cp_task_blk_num, solo_kernel_grid.x * solo_kernel_grid.y * solo_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z, 0)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float cp_solo_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        cp_solo_time += time_vec[i];
    }
    cp_solo_time /= 10.0f;

    //printf("[SOLO] solo_cp took %f ms\n\n", kernel_time);
    // cp gptb time / sgemm gptb time
    float load_ratio = gptb_cp_time / gptb_sgemm_time;
    printf("load_ratio: %f\n", load_ratio);
    printf("mix_duration: %f\n", sum_kernel_time);
    printf("cp gptb time: %f , sgemm gptb time: %f, cp_blk_num: %d, sgemm_blk_num: %d\n", gptb_cp_time, gptb_sgemm_time, mix_cp_task_blk_num, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z);

    sum_kernel_time += cp_solo_time;

    printf("ori sum time: %f, fuse_solo time: %f, improvement: %f%\n", ori_sum_time, sum_kernel_time, ((ori_sum_time - sum_kernel_time) * 100 / ori_sum_time));

	// Checking results
    // ---------------------------------------------------------------------------------------
    	// cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	    // cudaMemcpy(host_gptb_energy, gptb_output, volmemsz,  cudaMemcpyDeviceToHost);
            
        // errors = 0;
        // for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
        //     float v1 = host_ori_energy[i];
        //     float v2 = host_gptb_energy[i];
        //     if (fabs(v1 - v2) > 0.001f) {
        //         errors++;
        //         if (errors < 10) //printf("%f %f\n", v1, v2);
        //     }
        // }
        // if (errors > 0) {
        //     //printf("ORI VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        // }
        // else {
        //     //printf("Results verified: ORIG VERSION and GPTB VERSION agree.\n");
        // }
	// ---------------------------------------------------------------------------------------

    // Checking results
    // ---------------------------------------------------------------------------------------
        // cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
        // cudaErrCheck(cudaMemcpy(host_sgemm_gptb_c, sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

        // errors = 0;
        // for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
        //     float v1 = host_sgemm_ori_c[i];
        //     float v2 = host_sgemm_gptb_c[i];
        //     if (fabs(v1 - v2) > 0.001f) {
        //     errors++;
        //     if (errors < 10) //printf("%f %f\n", v1, v2);
        //     }
        // }
        // if (errors > 0) {
        //     //printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        // }
        // else {
        //     //printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        // }
    // ---------------------------------------------------------------------------------------


}
