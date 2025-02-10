#include "sgemm_kernel.h"
#include "Logger.h"
#include "header/sgemm_header.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;
    
    float alpha = 2.0f;
    float beta = 2.0f;

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


OriSGEMMKernel::OriSGEMMKernel(int id){
    Id = id;
    this->kernelName = "sgemm";
    initParams();
}

OriSGEMMKernel::~OriSGEMMKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->SGEMMKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriSGEMMKernel::initParams(){
    // sgemm variables
    // ---------------------------------------------------------------------------------------
        int sgemm_blks = 4;
        int sgemm_iter = 1;
        float *sgemm_ori_a;
        float *sgemm_ori_b;
        float *sgemm_ori_c;
        // float *sgemm_ptb_a;
        // float *sgemm_ptb_b;
        // float *sgemm_ptb_c;
        // float *sgemm_gptb_a;
        // float *sgemm_gptb_b;
        // float *sgemm_gptb_c;
        float *host_sgemm_ori_c;
        // float *host_sgemm_ptb_c;
        // float *host_sgemm_gptb_c;
        

        // parallel experiment
        int NORMAL_M = 4096;
        int NORMAL_N = 4128;
        int NORMAL_K = 4064;

        NORMAL_M = (NORMAL_M) * sgemm_iter;

        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

        host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        // host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        // host_sgemm_gptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

        curandGenerator_t sgemm_gen;
        curandErrCheck(curandCreateGenerator(&sgemm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(sgemm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
        // cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(sgemm_gptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(sgemm_gptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        curandErrCheck(curandDestroyGenerator(sgemm_gen));
    // ---------------------------------------------------------------------------------------
        dim3 sgemm_grid;
        dim3 sgemm_block;
        sgemm_block.x = TILE_N;
        sgemm_block.y = TILE_TB_HEIGHT;
        sgemm_grid.x = NORMAL_M/TILE_M;
        sgemm_grid.y = NORMAL_N/TILE_N;

        this->launchGridDim = sgemm_grid;
        this->launchBlockDim = sgemm_block;

        this->SGEMMKernelParams = new OriSGEMMParamsStruct();
        this->SGEMMKernelParams->A = sgemm_ori_a;
        this->SGEMMKernelParams->B = sgemm_ori_b;
        this->SGEMMKernelParams->C = sgemm_ori_c;
        this->SGEMMKernelParams->NORMAL_M = NORMAL_M;
        this->SGEMMKernelParams->NORMAL_N = NORMAL_N;
        this->SGEMMKernelParams->NORMAL_K = NORMAL_K;

        this->kernelParams.push_back(&(this->SGEMMKernelParams->A));
        this->kernelParams.push_back(&(this->SGEMMKernelParams->B));
        this->kernelParams.push_back(&(this->SGEMMKernelParams->C));
        this->kernelParams.push_back(&(this->SGEMMKernelParams->NORMAL_M));
        this->kernelParams.push_back(&(this->SGEMMKernelParams->NORMAL_N));
        this->kernelParams.push_back(&(this->SGEMMKernelParams->NORMAL_K));

        this->smem = 512;
        this->kernelFunc = (void*) ori_sgemm;
}

void OriSGEMMKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, 0));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
