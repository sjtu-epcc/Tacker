#include "header/sgemm_header.h"
#include "header/fft_header.h"
// fft
__device__ void G_GPU_exchange_fft_sgemm_fft0( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_sgemm_fft0(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_sgemm_fft0( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_sgemm_fft0(float2* data, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	
	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

	// // ori
	// int thread_id_x = threadIdx.x - thread_step;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

		// // ori
		// int block_id_x = block_pos;
        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

		float2 *ori_data = data + block_id_x * FFT_N;
		float2 v[FFT_R];
		// data = ori_data;

		int idxG = thread_id_x; 
		for (int r=0; r<FFT_R; r++) {  
			v[r] = ori_data[idxG + r*FFT_T];
		} 
		G_GPU_DoFft_fft_sgemm_fft0( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

// sgemm
__device__ void fft_sgemm_sgemm0(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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

__device__ void fft_sgemm_sgemm1(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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

__device__ void fft_sgemm_sgemm2(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

__device__ void fft_sgemm_sgemm3(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

// fft-sgemm-1-4
__global__ void mixed_fft_sgemm_kernel_1_4(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        fft_sgemm_fft0(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 0 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 1, fft0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        fft_sgemm_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        fft_sgemm_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        fft_sgemm_sgemm2(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 2 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        fft_sgemm_sgemm3(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 3 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 512
        );
    }

}
