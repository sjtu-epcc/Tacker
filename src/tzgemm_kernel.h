//tzgemm_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"
#include "Logger.h"
#include "header/tzgemm_header.h"
#include "header/pets_common.h"

extern long long MAX_ORI_WMMA_A;
extern long long MAX_ORI_WMMA_B;
extern long long MAX_ORI_WMMA_C;
extern Logger logger;
extern bool gemm_malloced;
#ifdef AKER_INT8
extern int8_t *ori_wmma_A;
extern int8_t *ori_wmma_B;
extern int16_t *ori_wmma_C;
#else
extern half *ori_wmma_A;
extern half *ori_wmma_B;
extern float *ori_wmma_C;
#endif
extern float *ori_host_A;
extern float *ori_host_B;
extern int MAX_M_GLOBAL;
extern int MAX_N_GLOBAL;
extern int MAX_K_GLOBAL;

// return GLOBAL of M,N,K and gridDim.x
inline std::vector<int> getTZGEMMGridDim(int m, int n, int k) {
    int M_GLOBAL = (m < 128) ? 128 : (m / 128) * 128;
    int N_GLOBAL = (n < 128) ? 128 : (n / 128) * 128;
    int K_GLOBAL = (k < 128) ? 128 : (k / 128) * 128;

    int M_TILES = M_GLOBAL / WMMA_M;
    int N_TILES = N_GLOBAL / WMMA_N;
    int K_TILES = K_GLOBAL / WMMA_K;

    int gridDimX = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);

    std::vector<int> ret;
    ret.push_back(M_GLOBAL);
    ret.push_back(N_GLOBAL);
    ret.push_back(K_GLOBAL);
    ret.push_back(gridDimX);

    return ret;
}


extern __global__ void ptb_tzgemm(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x);

class OriTZGEMMKernel : public Kernel {
public:
    // 构造函数
    OriTZGEMMKernel(int id, int m, int n, int k) {
        this->Id = id;
        this->kernelName = "tzgemm";
        this->m = m;
        this->n = n;
        this->k = k;
        this->initParams();
    }

    // 析构函数
    ~OriTZGEMMKernel(){};

    // impl virtual func
    void executeImpl(cudaStream_t stream) {
        printf("should not execute ori tzgemm!\n");
        exit(1);
        // printf("ptb_tzgemm blks num: %d\n", launchGridDim.x);
        // checkKernelErrors((ptb_tzgemm<<<launchGridDim.x, launchBlockDim.x, 0, stream>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, 
		// 					M_GLOBAL, N_GLOBAL, K_GLOBAL,
		// 					ori_blks, launchBlockDim.x)));
    }
    void initParams() {

        int M_GLOBAL = (m < 128) ? 128 : (m / 128) * 128;
        int N_GLOBAL = (n < 128) ? 128 : (n / 128) * 128;
        int K_GLOBAL = (k < 128) ? 128 : (k / 128) * 128;

        this->M_GLOBAL = M_GLOBAL;
        this->N_GLOBAL = N_GLOBAL;
        this->K_GLOBAL = K_GLOBAL;

        long long cur_ori_wmma_A = (long long)sizeof(half) * M_GLOBAL * K_GLOBAL;
        long long cur_ori_wmma_B = (long long)sizeof(half) * N_GLOBAL * K_GLOBAL;
        long long cur_ori_wmma_C = (long long)sizeof(float) * M_GLOBAL * N_GLOBAL;
        MAX_ORI_WMMA_A = max(MAX_ORI_WMMA_A, cur_ori_wmma_A);
        MAX_ORI_WMMA_B = max(MAX_ORI_WMMA_B, cur_ori_wmma_B);
        MAX_ORI_WMMA_C = max(MAX_ORI_WMMA_C, cur_ori_wmma_C);

        if (!gemm_malloced) {
            // printf("should not init gemm in ori tzgemm!\n");
            // exit(1);
            printf("[ori_tzgemm][initParams] try to malloc ori_wmma_A->%f MB, ori_wmma_B->%f MB, ori_wmma_C->%f MB\n", sizeof(half) * MAX_M_GLOBAL * MAX_K_GLOBAL / 1024.0 / 1024.0, sizeof(half) * MAX_N_GLOBAL * MAX_K_GLOBAL / 1024.0 / 1024.0, sizeof(float) * MAX_M_GLOBAL * MAX_N_GLOBAL / 1024.0 / 1024.0); 
            cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), MAX_ORI_WMMA_A));
            cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), MAX_ORI_WMMA_B));
            cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), MAX_ORI_WMMA_C));
            gemm_malloced = true;
            cudaErrCheck(cudaMemset((void*)ori_wmma_C, 0.0f, MAX_ORI_WMMA_C));
            cudaErrCheck(cudaMemset((void*)ori_wmma_A, 0.0f, MAX_ORI_WMMA_A));
            cudaErrCheck(cudaMemset((void*)ori_wmma_B, 0.0f, MAX_ORI_WMMA_B));
        }

        // printf("M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
        // cudaErrCheck(cudaMemset(ori_wmma_C, 1.0f, sizeof(float) * M_GLOBAL * N_GLOBAL));
        // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
        // cudaErrCheck(cudaMemset(ori_wmma_B, 0.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));


        int M_TILES = M_GLOBAL / WMMA_M;
        int N_TILES = N_GLOBAL / WMMA_N;
        int K_TILES = K_GLOBAL / WMMA_K;

        this->launchGridDim.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
        ori_blks = this->launchGridDim.x;
        this->launchGridDim.y = 1;
        this->launchGridDim.z = 1;
        this->launchBlockDim.x = THREADS_PER_BLOCK;
        this->launchBlockDim.y = 1;
        this->launchBlockDim.z = 1;

        this->kernelParams.push_back(&ori_wmma_A);
        this->kernelParams.push_back(&ori_wmma_B);
        this->kernelParams.push_back(&ori_wmma_C);
        this->kernelParams.push_back(&(this->M_GLOBAL));
        this->kernelParams.push_back(&(this->N_GLOBAL));
        this->kernelParams.push_back(&(this->K_GLOBAL));

        this->smem = 18432;
    }

    int m, n, k;
    int M_GLOBAL, N_GLOBAL, K_GLOBAL;
    int ori_blks;

    std::vector<int> getArgs() override {
    return std::vector<int>();
}
private:
    void loadKernel(){}; 
};


// extern boost::property_tree::ptree ptr;

