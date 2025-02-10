

__global__ void ori_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, int iteration) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;
    
    float alpha = 2.0f;
    float beta = 2.0f;

    for (int loop = 0; loop < iteration; loop++) {
        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
        int m = blockIdx.x * TILE_M + mid;
        int n = blockIdx.y * TILE_N + threadIdx.x;
        __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[threadIdx.y][threadIdx.x] = B[n + (i + threadIdx.y) * ldb];
            __syncthreads();
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            __syncthreads();
        }
        int t = ldc * blockIdx.y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}


__global__ void ptb_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K,
                        int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = threadIdx.x;
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + thread_id_x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
                __syncthreads();
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                __syncthreads();
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__global__ void ptb2_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K,
                        int grid_dimension_x, int grid_dimension_y, int iteration) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x;
    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + threadIdx.x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[threadIdx.y][threadIdx.x] = B[n + (i + threadIdx.y) * ldb];
                __syncthreads();
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                __syncthreads();
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__device__ void mix_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
                    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
                    int iteration, int thread_step) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += SGEMM_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = (threadIdx.x - thread_step);
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
}

__device__ void general_ptb_sgemm0(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

__device__ void general_ptb_sgemm1(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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

__device__ void general_ptb_sgemm2(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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

__device__ void general_ptb_sgemm3(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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

__device__ void general_ptb_sgemm4(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}


__device__ void general_ptb_sgemm5(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

__device__ void general_ptb_sgemm6(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

extern "C" __global__ void g_general_ptb_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
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