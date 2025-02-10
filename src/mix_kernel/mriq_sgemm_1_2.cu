#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
using namespace nvcuda; 
#include "header/sgemm_header.h"
#include "header/mriq_header.h"

// mriq
__device__ void mriq_sgemm_mriq0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }
        
        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2_MRIQ * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2_MRIQ * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

// sgemm
__device__ void mriq_sgemm_sgemm0(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
	        int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
    
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    // // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = (threadIdx.x - thread_base); // TODO: check
        int m = block_id_x * TILE_M + mid;
        int n = block_id_y * TILE_N + thread_id_x;
        

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

__device__ void mriq_sgemm_sgemm1(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
	        int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
    
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    // // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = (threadIdx.x - thread_base); // TODO: check
        int m = block_id_x * TILE_M + mid;
        int n = block_id_y * TILE_N + thread_id_x;
        

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}
// mriq-sgemm-1-2
__global__ void mixed_mriq_sgemm_kernel_1_2(int mriq0_numK, int mriq0_kGlobalIndex, float* mriq0_x, float* mriq0_y, float* mriq0_z, float* mriq0_Qr, float* mriq0_Qi, int mriq0_grid_dimension_x, int mriq0_grid_dimension_y, int mriq0_grid_dimension_z, int mriq0_block_dimension_x, int mriq0_block_dimension_y, int mriq0_block_dimension_z, int mriq0_ptb_start_block_pos, int mriq0_ptb_iter_block_step, int mriq0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        mriq_sgemm_mriq0(
            mriq0_numK, mriq0_kGlobalIndex, mriq0_x, mriq0_y, mriq0_z, mriq0_Qr, mriq0_Qi, mriq0_grid_dimension_x, mriq0_grid_dimension_y, mriq0_grid_dimension_z, mriq0_block_dimension_x, mriq0_block_dimension_y, mriq0_block_dimension_z, mriq0_ptb_start_block_pos + 0 * mriq0_ptb_iter_block_step, mriq0_ptb_iter_block_step * 1, mriq0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        mriq_sgemm_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        mriq_sgemm_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 384
        );
    }

}
