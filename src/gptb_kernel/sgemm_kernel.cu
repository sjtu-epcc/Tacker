#include "header/sgemm_header.h"

extern "C" __global__ void general_ptb_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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