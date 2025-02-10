// TaskManager.cpp
#include "TaskManager.h"
#include "Logger.h"
#include "Recorder.h"
#include "tzgemm_kernel.h"
#include "GPTBKernel.h"
#include "MixKernel.h"
#include "Creator.h"
#include "json.h"
#include <time.h>
#include <cuda_profiler_api.h>
#include "dnn/vgg16/vgg16.h"
#include "include/dnn.h"


extern Logger logger;
extern Recorder recorder;

int batch_size = -1;

long long MAX_ORI_WMMA_A = 0;
long long MAX_ORI_WMMA_B = 0;
long long MAX_ORI_WMMA_C = 0;
int MAX_M_GLOBAL = 802816;
int MAX_N_GLOBAL = 128;
int MAX_K_GLOBAL = 128;
int MAX_COL_BUFFER = 0;
int MAX_BOTTOM = 0;
bool im2col_malloced = false;
bool gemm_malloced = false;
float *bottom;
float *col_buffer;
float *ori_host_A;
float *ori_host_B;
#ifdef AKER_INT8
int8_t *ori_wmma_A;
int8_t *ori_wmma_B;
int16_t *ori_wmma_C;
#else
half *ori_wmma_A;
half *ori_wmma_B;
float *ori_wmma_C;
#endif
float *cublas_wmma_C;
float *ori_wmma_results1;
float *ori_wmma_results2;

__global__ void im2col_gpu_kernel_(int n, float* data_im,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    float* data_col, int data_im_size, int data_col_size) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            int h_index = i / width_col;
            int h_col = h_index % height_col;
            int w_col = i % width_col;
            int c_im = h_index / height_col;
            int c_col = c_im * kernel_h * kernel_w;
            int h_offset = h_col * stride_h - pad_h;
            int w_offset = w_col * stride_w - pad_w;
            float* data_col_ptr = data_col;
            data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
            float* data_im_ptr = data_im;
            data_im_ptr += (c_im * height + h_offset) * width + w_offset;
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int h_im = h_offset + i * dilation_h;
                    int w_im = w_offset + j * dilation_w;
                    if (h_col >= height_col || w_col >= width_col || h_col < 0 || w_col < 0 || (data_col_ptr - data_col) >= data_col_size) {
                        // printf("h_col: %d, w_col: %d, height_col: %d, width_col: %d, data_col_ptr - data_col: %d\n", h_col, w_col, height_col, width_col, data_col_ptr - data_col);
                        continue;
                    }
                    *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width && (i * dilation_h * width + j * dilation_w < data_im_size)) ?
                        data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                    data_col_ptr += height_col * width_col;
                }
            }
        }
}


__global__ void convertFp32ToFp16_ (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

std::vector<int> get_mnk_from_cudnn_args(std::vector<int>& cudnn_args) {
    int input_n = cudnn_args[0];
    int input_c = cudnn_args[1];
    int input_h = cudnn_args[2];
    int input_w = cudnn_args[3];
    int kernel_k = cudnn_args[4];
    int kernel_c = cudnn_args[5];
    int kernel_h = cudnn_args[6];
    int kernel_w = cudnn_args[7];

    int pad_h = cudnn_args[8];
	int pad_w = cudnn_args[9];
	int stride_h = cudnn_args[10];
	int stride_w = cudnn_args[11];
    int dilation_h = cudnn_args[12];
	int dilation_w = cudnn_args[13];
    
    int height_col = (input_h + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (input_w + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int M_INPUT = input_n * height_col * width_col; // 1 * 112 * 112
    int N_INPUT = kernel_k; // 64
    int K_INPUT = kernel_h * kernel_w * input_c;  // 7 * 7 * 3

    // if (input_n != batch_size) {
    //     M_INPUT = batch_size * height_col * width_col;
    // }

    vector<int> mnk = {M_INPUT, N_INPUT, K_INPUT};
    return mnk;
}

void mixcudnnConvolutionForward(std::vector<int>& cudnn_args, GPTBKernel* gptb_cd_kernel, int cd_start_blk, int cd_end_blk, cudaStream_t stream) {
    // printf("--------------------------------mixcudnnConvolutionForward-%d\n", batch_size);
    // cudaEvent_t startKERNEL, stopKERNEL;
	// cudaErrCheck(cudaEventCreate(&startKERNEL));
	// cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // float milliseconds = 0;

    // cudaErrCheck(cudaEventRecord(startKERNEL));
    

    // img2col参数
    int input_n = cudnn_args[0];
	int input_c = cudnn_args[1];
	int input_h = cudnn_args[2];
	int input_w = cudnn_args[3];
	// int output_n;
	// int output_c;
	// int output_h;
	// int output_w;

	int col_n;
	int col_c;
	int col_h;
	int col_w;

    int kernel_k = cudnn_args[4];
    int kernel_c = cudnn_args[5];
    int kernel_h = cudnn_args[6];
	int kernel_w = cudnn_args[7];
	int pad_h = cudnn_args[8];
	int pad_w = cudnn_args[9];
	int stride_h = cudnn_args[10];
	int stride_w = cudnn_args[11];
    int dilation_h = cudnn_args[12];
	int dilation_w = cudnn_args[13];

    
    col_n = input_n;
    col_c = input_c;

    int height_col = (input_h + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (input_w + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = input_n * input_c * height_col * width_col;

    // assert(input_n == output_n);
    // assert(output_h == height_col);
    // assert(output_w == width_col);

    col_h = height_col;
    col_w = width_col;

    // printf("col_n:%d, col_c:%d, col_h:%d, col_w:%d\n", col_n, col_c, col_h, col_w);

    // std::cin >> foo;

    MAX_COL_BUFFER = max(MAX_COL_BUFFER, col_n * col_c * col_h * col_w);
    MAX_BOTTOM = max(MAX_BOTTOM, input_n * input_c * input_h * input_w);

    // printf("MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);
    
    if (!im2col_malloced) {
        cudaErrCheck(cudaMalloc((void**)&bottom, MAX_BOTTOM * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&col_buffer, MAX_COL_BUFFER * sizeof(float)));
        im2col_malloced = true;
    }

    // curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    // curandErrCheck(curandGenerateUniform(gen, bottom, input_n * input_c * input_h * input_w));
    // curandErrCheck(curandGenerateUniform(gen, col_buffer, col_n * col_c * col_h * col_w));
    // cudaErrCheck(cudaMemset(bottom, 1.0f, input_n * input_c * input_h * input_w * sizeof(float)));
    // cudaErrCheck(cudaMemset(col_buffer, 1.0f, col_n * col_c * col_h * col_w * sizeof(float)));

    // 调用 img2col
    dim3 im_grid;
	dim3 im_block;
    im_block.x = 256;
	im_grid.x = int(num_kernels / 256);
	im_grid.x = SM_NUM * 2;

    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("pre im2col took %f ms\n", milliseconds);
    // milliseconds = 0;

    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // launch im2col
    checkKernelErrors((im2col_gpu_kernel_<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, input_n * input_c * input_h * input_w, col_n * col_c * col_h * col_w)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("im2col took %f ms\n", milliseconds);
    // milliseconds = 0;

    // cudaErrCheck(cudaEventRecord(startKERNEL));

    int M_INPUT = input_n * height_col * width_col; // 32 * 112 * 112
    int N_INPUT = kernel_k; // 64
    int K_INPUT = kernel_h * kernel_w * input_c;  // 7 * 7 * 3

    int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

    // cudaErrCheck(cudaEventRecord(startKERNEL));

    auto ori_tzgemm = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    auto gptb_tzgemm_kernel = new GPTBKernel(
        1, 
        "tzgemm",
        "gptb_tzgemm", 
        ori_tzgemm,
        dim3(SM_NUM * 2, 1, 1), 
        dim3(128, 1, 1), 
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );

    // std::cout << "gptb_cd_kernel->kernelName " << gptb_cd_kernel->kernelName << std::endl;
    std::string mix_kernel_name = "tzgemm_" + gptb_cd_kernel->kernelName;

    auto mix_kernel = new MixKernel(
        1, 
        mix_kernel_name, 
        gptb_cd_kernel,
        gptb_tzgemm_kernel, 
        dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
        dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
        cd_start_blk,
        cd_end_blk,
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );

    // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));
	// convertFp32ToFp16_ <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    // convertFp32ToFp16_ <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
    // cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // printf("M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("cd_start_blk: %d, cd_end_blk: %d\n", mix_kernel->kernel1_start_block_pos, mix_kernel->kernel1_end_block_pos);
    // printf("tzgemm start block: %d, end block: %d\n", mix_kernel->kernel2_start_block_pos, mix_kernel->kernel2_end_block_pos);
    // printf("mix grid: %d, block: %d\n", mix_kernel->launchGridDim.x, mix_kernel->launchBlockDim.x);

    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("pre conv took %f ms\n", milliseconds);
    // milliseconds = 0;

    // cudaErrCheck(cudaEventRecord(startKERNEL));

    mix_kernel->execute(stream);
    
    // gptb_cd_kernel->gptbParams.ptb_end_block_pos -= cd_end_blk * fget_kernel_info(mix_kernel_name, "block_ratio");
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("cudnnConv kernel took %f ms\n", milliseconds);
    // milliseconds = 0;

    // printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	// printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	// printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

    // printf("MAX_M_GLOBAL: %d, MAX_N_GLOBAL: %d, MAX_K_GLOBAL: %d\n", MAX_M_GLOBAL, MAX_N_GLOBAL, MAX_K_GLOBAL);

    // printf("--------------------------------\n");
    // free(ori_tzgemm);
    // free(gptb_tzgemm_kernel);
    // free(mix_kernel);

    return ;
}


void mixcublasSgemm(std::vector<int>& gemm_args, GPTBKernel* gptb_cd_kernel, int cd_start_blk, int cd_end_blk, cudaStream_t stream)
{
    // printf("--------------------------------mixcublasSgemm-%d\n", batch_size);
    int M_INPUT = gemm_args[0];
    int N_INPUT = gemm_args[1];
    int K_INPUT = gemm_args[2];

    int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = SM_NUM * 2;
	wmma_block.x = THREADS_PER_BLOCK;


    // if (!gemm_malloced) {
    //     cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MAX_M_GLOBAL * MAX_K_GLOBAL));
    //     cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MAX_N_GLOBAL * MAX_K_GLOBAL));
    //     cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * MAX_M_GLOBAL * MAX_K_GLOBAL));
    //     cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * MAX_N_GLOBAL * MAX_K_GLOBAL));
    //     cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * MAX_M_GLOBAL * MAX_N_GLOBAL));
    //     gemm_malloced = true;
    // }
    // assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	// assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	// assert(((unsigned long long)ori_wmma_C) % 128 == 0);
    
    // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));
	// convertFp32ToFp16_ <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    // convertFp32ToFp16_ <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
    // cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // printf("Running with gemm...\n");
    // printf("gemm M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("gemm block dim: %d, grid dim: %d\n", wmma_block_dim_x, wmma_grid_dim_x);
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    auto ori_tzgemm = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    auto gptb_tzgemm_kernel = new GPTBKernel(
        1, 
        "tzgemm",
        "gptb_tzgemm", 
        ori_tzgemm,
        dim3(SM_NUM * 2, 1, 1), 
        dim3(128, 1, 1), 
        0,
        wmma_grid_dim_x
    );


    
    std::string mix_kernel_name = "tzgemm_" + gptb_cd_kernel->kernelName;


    auto mix_kernel = new MixKernel(
        1, 
        mix_kernel_name, 
        gptb_cd_kernel,
        (gptb_tzgemm_kernel), 
        dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
        dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
        cd_start_blk,
        cd_end_blk,
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );
    // printf("M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("cd_start_blk: %d, cd_end_blk: %d\n", mix_kernel->kernel1_start_block_pos, mix_kernel->kernel1_end_block_pos);
    // printf("tzgemm start block: %d, end block: %d\n", mix_kernel->kernel2_start_block_pos, mix_kernel->kernel2_end_block_pos);
    // printf("mix grid: %d, block: %d\n", mix_kernel->launchGridDim.x, mix_kernel->launchBlockDim.x);

    // printf("cd_start_blk: %d, cd_end_blk: %d\n", mix_kernel->kernel1_start_block_pos, mix_kernel->kernel1_end_block_pos);
    // printf("tzgemm start block: %d, end block: %d\n", mix_kernel->kernel2_start_block_pos, mix_kernel->kernel2_end_block_pos);

    mix_kernel->execute(stream);
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // free(ori_tzgemm);
    // free(gptb_tzgemm_kernel);
    // free(mix_kernel);
    return ;
}

TaskManager::TaskManager(Task* lc_task, const std::string& be_task1, const std::string& be_task2) {
    // logger.INFO("TaskManager is created!");
    this->be_task1_name = be_task1;
    this->be_task2_name = be_task2;
    this->lc_task = lc_task;

}

#include <chrono>
using namespace chrono;

void TaskManager::executeAllTasks(ExecutionMode mode, cudaStream_t stream) {
    float kernel_time = 0.0f, iter_time = 0.0f;
    cudaEvent_t startKERNEL, stopKERNEL, start_i, stop_i;
    cudaEventCreate(&startKERNEL);
    cudaEventCreate(&stopKERNEL);
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    float qos_headroom = 100.0f;
    int be1_kernel_idx = 0;
    int be2_kernel_idx = 0;

    

    vector<float> lc_kernel_time_vec(lc_task->kernels.size(), 0.0f);
    float lc_kernel_time = 0.0f;
    float tmp_time = 0.0f;
    for (int i = 0; i < 5; ++i ){
        lc_task->initExecution();
        for (auto& lc_kernel : lc_task->kernels) {
            lc_kernel->execute(stream);
        }
    }
    char foo;

    

    GPTBKernel * be_kernel1 = createKernel(be_task1_name);
    be_kernel1->kernel_->initParams();
    int mixable_times = 0;

    
    // test ori
    for (int i = 0; i < 10; ++i) {
        lc_task->initExecution();
        int lc_kernel_idx = 0;
        for (auto& lc_kernel : lc_task->kernels) {
            // printf("exec %s...\n", lc_kernel->kernelName.c_str());
            auto start = clock();
            if ((!i) && lc_kernel->mixable != 0) mixable_times++;
            if (lc_kernel->mixable == 1) { // cublassgemm
                std::vector<int> mnk = lc_kernel->getArgs();
                if (mnk.size() != 3) {
                    logger.INFO("lc kernel: " + lc_kernel->kernelName + " mnk size is not 3, but " + std::to_string(mnk.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
                    logger.ERROR("exit");
                }
                int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
                // lc_kernel->execute(stream);
            } else if (lc_kernel->mixable == 2) { // cudnnConv
                std::vector<int> cudnnArgs = lc_kernel->getArgs();
                if (cudnnArgs.size() != 14) {
                    logger.INFO("lc kernel: " + lc_kernel->kernelName + " cudnn args size is not 14, but " + std::to_string(cudnnArgs.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
                    logger.ERROR("exit");
                }
                std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
                int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
                // lc_kernel->execute(stream);
            } else { // no mix
                // lc_kernel->execute(stream);
            }
            lc_kernel->execute(stream);
            cudaDeviceSynchronize();
            cudaStreamSynchronize(stream);
            cudaStreamSynchronize(0);
            auto end = clock();
            auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
            lc_kernel_time_vec[lc_kernel_idx++] += duration;
        }
    }
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(0);
    // cal every kernel time in vec
    for (int i = 0; i < lc_kernel_time_vec.size(); ++i) {
        lc_kernel_time_vec[i] /= 10;
        // printf("[kernel]: %s, [ori_time]: %f\n", lc_task->kernels[i]->kernelName.c_str(), lc_kernel_time_vec[i]);
        lc_kernel_time += lc_kernel_time_vec[i];
    }
    printf("\n[PRE]lc kernel took %f ms to execute.\n", lc_kernel_time);

    // char foo;
    // cin >> foo;

    vector<float> lc_headroom_vec(lc_kernel_time_vec);
    // cal headroom vec
    for (int i = lc_headroom_vec.size() - 2; i >= 0; --i) {
        lc_headroom_vec[i] += lc_headroom_vec[i + 1];
    }

    // // warmup mix
    // for (int i = 0; i < 10; ++i) {
    //     lc_task->initExecution();
    //     be_kernel1->kernel_->initParams();
    //     int lc_kernel_idx = 0;
    //     for (auto& lc_kernel : lc_task->kernels) {
    //         if (lc_kernel->mixable == 1) { // cublassgemm
    //             std::vector<int> mnk = lc_kernel->getArgs();
    //             if (mnk.size() != 3) {
    //                 logger.INFO("lc kernel: " + lc_kernel->kernelName + " mnk size is not 3, but " + std::to_string(mnk.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
    //                 logger.ERROR("exit");
    //             }
    //             int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
    //             mixcublasSgemm(mnk, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);
    //         } else if (lc_kernel->mixable == 2) { // cudnnConv
    //             std::vector<int> cudnnArgs = lc_kernel->getArgs();
    //             if (cudnnArgs.size() != 14) {
    //                 logger.INFO("lc kernel: " + lc_kernel->kernelName + " cudnn args size is not 14, but " + std::to_string(cudnnArgs.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
    //                 logger.ERROR("exit");
    //             }
    //             std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
    //             int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
    //             mixcudnnConvolutionForward(cudnnArgs, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);
    //         } else { // no mix
    //             lc_kernel->execute(stream);
    //         }
    //     }
    //     cudaDeviceSynchronize();
    //     cudaStreamSynchronize(stream);
    //     cudaStreamSynchronize(0);
    // }

    // get cd
    auto be_task1 = createKernel(be_task1_name);
    auto be_task2 = createKernel(be_task2_name);

    float be_task1_ori_time = 0.0f, be_task2_ori_time = 0.0f;
    // init cd
    // be_task1->kernel_->initParams();
    // be_task2->kernel_->initParams();

    // ori sum time
    // warmup
    for (int i = 0; i < 10; ++i) {
        be_task1->execute(stream);
        be_task2->execute(stream);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    be_task1->execute(stream);
    CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task1_ori_time, startKERNEL, stopKERNEL));

    CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    be_task2->execute(stream);
    CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task2_ori_time, startKERNEL, stopKERNEL));

    be_kernel1->kernel_->initParams();

    int no_split_times = 0;
    long long total_be_num = 0;
    int mix_times = 0;
    float time_earned = 0.0f;
    for (int i = 0; i < 5; ++i) {
        lc_task->initExecution();
        qos_headroom = 50.0f;
        long long cd_block_num_executed = 0;
        int lc_kernel_idx = 0;
        mix_times = 0;
        time_earned = 0.0f;

        CUDA_SAFE_CALL(cudaProfilerStart());
        // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
        for (auto& lc_kernel : lc_task->kernels) {
            int cd_block_num = 0;
            // printf("exec %s...\n", lc_kernel->kernelName.c_str());
            if ("Dot_float_float_float_cuda_lib_Dot_133" == lc_kernel->kernelName) {
                lc_kernel->mixable = 0;
            }
            auto start = clock();
            if (qos_headroom < lc_headroom_vec[lc_kernel_idx] + 4.0f) {
                lc_kernel->execute(stream);
            } else if (lc_kernel->mixable == 1) { // cublassgemm
                std::vector<int> mnk = lc_kernel->getArgs();
                std::vector<int> MNKD = getTZGEMMGridDim(mnk[0], mnk[1], mnk[2]);
                if (mode == ExecutionMode::Tacker) cd_block_num = be_kernel1->gptbParams.ptb_end_block_pos;
                else if (mode == ExecutionMode::Aker) {
                    float block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(MNKD[2]));
                    if (block_ratio == JSON_NOT_FOUND) {
                        float base_block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(4096));
                        block_ratio = base_block_ratio * MNKD[2] / 4096;
                    }
                    cd_block_num = min(int(MNKD[3] * block_ratio), be_kernel1->gptbParams.ptb_end_block_pos);
                }
                mixcublasSgemm(mnk, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);

            } else if (lc_kernel->mixable == 2) { // cudnnConv
                std::vector<int> cudnnArgs = lc_kernel->getArgs();
                std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
                std::vector<int> MNKD = getTZGEMMGridDim(mnk[0], mnk[1], mnk[2]);
                if (mode == ExecutionMode::Tacker) cd_block_num = be_kernel1->gptbParams.ptb_end_block_pos;
                else if (mode == ExecutionMode::Aker) {
                    float block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(MNKD[2]));
                    if (block_ratio == JSON_NOT_FOUND) {
                        float base_block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(4096));
                        block_ratio = base_block_ratio * MNKD[2] / 4096;
                    }
                    cd_block_num = min(int(MNKD[3] * block_ratio), be_kernel1->gptbParams.ptb_end_block_pos);
                }
                mixcudnnConvolutionForward(cudnnArgs, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);

            } else { // no mix
                lc_kernel->execute(stream);
            }
            cudaDeviceSynchronize();
            cudaStreamSynchronize(stream);
            cudaStreamSynchronize(0);
            auto end = clock();
            auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
            // if no throughput improve, use ori time
            bool do_mix = false;
            float improve = 0.0f;
            if (lc_kernel->mixable == 0) {
                qos_headroom -= lc_kernel_time_vec[lc_kernel_idx];
            }
            else if (duration > lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos) {
                qos_headroom -= lc_kernel_time_vec[lc_kernel_idx];
            } else {
                qos_headroom -= duration;
                do_mix = true;
                cd_block_num_executed += cd_block_num;
                mix_times += 1;
                time_earned += lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos - duration;
                improve = (lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos - duration) / (lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos);
            }
            // if (i == 4 && lc_kernel_idx < lc_kernel_time_vec.size() && lc_kernel->mixable != 0)  {
            //     printf("[%s]--[kernel]: %s, [ori_time]: %f, [mix_time]: %f, [qos_headroom]: %f, [improve]: %f%\n", do_mix ? "Yes" : "No",  lc_kernel->kernelName.c_str(), lc_kernel_time_vec[lc_kernel_idx], duration, qos_headroom, improve * 100);
            // }
            lc_kernel_idx ++;
        }
        cudaProfilerStop();
        // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
        // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        // CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));

        // printf("\n[Solo]total lc kernel took %f ms to execute.\n", 50.0f - qos_headroom);
        total_be_num = cd_block_num_executed;

        // cd_block_num_executed = 0;
        // ori_exec_num = 0;
    }

    // total_be_num /= 5;
    printf("execed be blks: %d, mix times: %d/%d, time earned: %f\n", total_be_num, mix_times, mixable_times, time_earned);
    // printf("avg_be_blks_num: %d, avg_be_task_num: %d\n", total_be_num, total_be_num / get_kernel_info(be_task1_name, "ori_blks"));
    // printf("no split time: %d/%d\n", no_split_times, mixable_times);
    // printf("total mix chance: %d\n", mixable_times);
    // qos_headroom -= kernel_time;

    // mix cd pair
    // ori sum time
    printf("[Ori] be_task1: %s, be_task2: %s\n", be_task1_name.c_str(), be_task2_name.c_str());
    // printf("[Ori] task1 blks range: %d - %d, task2 blks range: %d - %d\n", be_task1->gptbParams.ptb_start_block_pos, be_task1->gptbParams.ptb_end_block_pos, be_task2->gptbParams.ptb_start_block_pos, be_task2->gptbParams.ptb_end_block_pos);
    // printf("[Ori] BE task1 cost %f ms, BE task2 cost %f ms\n", be_task1_ori_time, be_task2_ori_time);
    printf("[Ori] BE task1 + task2 took %f + %f = %f ms to execute.\n", be_task1_ori_time, be_task2_ori_time, be_task1_ori_time + be_task2_ori_time);

    float stage1_be_time = 0.0f, stage2_be_time = 0.0f;

    stage1_be_time = total_be_num * be_task1_ori_time / be_task1->gptbParams.ptb_end_block_pos;

    auto mix_be_name = be_task1->kernelName[0] < be_task2->kernelName[0] ? be_task1->kernelName + "_" + be_task2->kernelName : be_task2->kernelName + "_" + be_task1->kernelName;
    auto be_mix = createMixKernel(mix_be_name);

    for (int i = 0; i < 5; ++i) be_mix->execute(stream);

    cudaDeviceSynchronize();

    float mix_be_time = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float tmp_time = 0.0f;
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
        be_mix->execute(stream);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&tmp_time, startKERNEL, stopKERNEL));
        mix_be_time += tmp_time;
    }
    mix_be_time /= 10;
    printf("[Mix] BE task1 + task2 took %f ms to execute.\n", mix_be_time);

    if (mix_be_time < be_task1_ori_time + be_task2_ori_time) {
        stage2_be_time = qos_headroom * (be_task1_ori_time + be_task2_ori_time) / (mix_be_time);
    } else {
        stage2_be_time = qos_headroom;
    }

    stage2_be_time = max(0.0f, stage2_be_time);
    if (ExecutionMode::Aker == mode) {
        printf("[Result] Aker BE task took %f + %f(%f) = %f ms to execute.\n", stage1_be_time, stage2_be_time, qos_headroom, stage1_be_time + stage2_be_time);
        // printf("[Result] Tacker BE task took %f + %f = %f ms to execute.\n", stage1_be_time, qos_headroom, stage1_be_time + qos_headroom);
    }
    else if (ExecutionMode::Tacker == mode) {
        printf("[Result] Tacker BE task took %f + %f(%f) = %f ms to execute.\n", stage1_be_time, qos_headroom, qos_headroom, stage1_be_time + qos_headroom);
    }

    // free event
    CUDA_SAFE_CALL(cudaEventDestroy(startKERNEL));
    CUDA_SAFE_CALL(cudaEventDestroy(stopKERNEL));
}


void TaskManager::execute_with_one_cd_kernel(ExecutionMode mode, cudaStream_t stream) {
    float kernel_time = 0.0f, iter_time = 0.0f;
    cudaEvent_t startKERNEL, stopKERNEL, start_i, stop_i;
    cudaEventCreate(&startKERNEL);
    cudaEventCreate(&stopKERNEL);
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    float qos_headroom = 100.0f;
    int be1_kernel_idx = 0;

    vector<float> lc_kernel_time_vec(lc_task->kernels.size(), 0.0f);
    float lc_kernel_time = 0.0f;
    float tmp_time = 0.0f;
    for (int i = 0; i < 5; ++i ){
        lc_task->initExecution();
        for (auto& lc_kernel : lc_task->kernels) {
            lc_kernel->execute(stream);
        }
    }
    cudaDeviceSynchronize();

    GPTBKernel * be_kernel1 = createKernel(be_task1_name);
    // if (be_kernel1->kernel_->Id < 0) {
    //     be_kernel1->kernel_->initParams_int();
    // } else if (be_kernel1->kernel_->Id == 1000) {
    //     printf("no impl yet!\n");
    //     exit(1);
    // } else {
    //     be_kernel1->kernel_->initParams();
    // }
    int mixable_times = 0;
    // test ori
    for (int i = 0; i < 10; ++i) {
        lc_task->initExecution();
        int lc_kernel_idx = 0;
        for (auto& lc_kernel : lc_task->kernels) {
            auto start = clock();
            if ((!i) && lc_kernel->mixable != 0) mixable_times++;
            if (lc_kernel->mixable == 1) { // cublassgemm
                std::vector<int> mnk = lc_kernel->getArgs();
                if (mnk.size() != 3) {
                    logger.INFO("lc kernel: " + lc_kernel->kernelName + " mnk size is not 3, but " + std::to_string(mnk.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
                    logger.ERROR("exit");
                }
                int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
                // lc_kernel->execute(stream);
            } else if (lc_kernel->mixable == 2) { // cudnnConv
                std::vector<int> cudnnArgs = lc_kernel->getArgs();
                if (cudnnArgs.size() != 14) {
                    logger.INFO("lc kernel: " + lc_kernel->kernelName + " cudnn args size is not 14, but " + std::to_string(cudnnArgs.size()) + ", mixable: " + std::to_string(lc_kernel->mixable));
                    logger.ERROR("exit");
                }
                std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
                int cd_block_num = min(int(getTZGEMMGridDim(mnk[0], mnk[1], mnk[2])[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);
                // lc_kernel->execute(stream);
            } else { // no mix
                // lc_kernel->execute(stream);
            }
            lc_kernel->execute(stream);
            cudaDeviceSynchronize();
            cudaStreamSynchronize(stream);
            cudaStreamSynchronize(0);
            auto end = clock();
            auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
            lc_kernel_time_vec[lc_kernel_idx++] += duration;
        }
    }
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(0);
    // cal every kernel time in vec
    for (int i = 0; i < lc_kernel_time_vec.size(); ++i) {
        lc_kernel_time_vec[i] /= 10;
        // printf("[kernel]: %s, [ori_time]: %f\n", lc_task->kernels[i]->kernelName.c_str(), lc_kernel_time_vec[i]);
        lc_kernel_time += lc_kernel_time_vec[i];
    }
    printf("\n[Result]lc kernel took %f ms to execute.\n", lc_kernel_time);

    // char foo;
    // cin >> foo;

    vector<float> lc_headroom_vec(lc_kernel_time_vec);
    // cal headroom vec
    for (int i = lc_headroom_vec.size() - 2; i >= 0; --i) {
        lc_headroom_vec[i] += lc_headroom_vec[i + 1];
    }

    // get cd
    auto be_task1 = createKernel(be_task1_name);

    float be_task1_ori_time = 0.0f, be_task2_ori_time = 0.0f;
    // // init cd
    // be_task1->kernel_->initParams();
    // be_task2->kernel_->initParams();
    // // ori sum time

    int cd_ptb_launch_x = be_task1->launchGridDim.x;
    be_task1->launchGridDim.x = be_task1->gptbParams.ptb_end_block_pos;
    CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    be_task1->execute(stream);
    CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task1_ori_time, startKERNEL, stopKERNEL));
    be_task1->launchGridDim.x = cd_ptb_launch_x;

    printf("[Ori] be_task: %s\n", be_task1_name.c_str());
    printf("[Result] task blks range: %d - %d\n", be_task1->gptbParams.ptb_start_block_pos, be_task1->gptbParams.ptb_end_block_pos);
    printf("[Result] BE task cost %f ms\n", be_task1_ori_time);

    // be_kernel1->kernel_->initParams();

    int no_split_times = 0;
    long long total_be_num = 0;
    int mix_times = 0;
    float time_earned = 0.0f;
    for (int i = 0; i < 5; ++i) {
        lc_task->initExecution();
        qos_headroom = 50.0f;
        long long cd_block_num_executed = 0;
        int lc_kernel_idx = 0;
        mix_times = 0;
        time_earned = 0.0f;

        CUDA_SAFE_CALL(cudaProfilerStart());
        // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
        for (auto& lc_kernel : lc_task->kernels) {
            int cd_block_num = 0;
            // printf("exec %s...\n", lc_kernel->kernelName.c_str());
            if ("Dot_float_float_float_cuda_lib_Dot_133" == lc_kernel->kernelName) {
                lc_kernel->mixable = 0;
            }
            auto start = clock();
            if (qos_headroom < lc_headroom_vec[lc_kernel_idx] + 4.0f) {
                lc_kernel->execute(stream);
            } else if (lc_kernel->mixable == 1) { // cublassgemm
                std::vector<int> mnk = lc_kernel->getArgs();
                std::vector<int> MNKD = getTZGEMMGridDim(mnk[0], mnk[1], mnk[2]);
                if (mode == ExecutionMode::Tacker) cd_block_num = be_kernel1->gptbParams.ptb_end_block_pos;
                else if (mode == ExecutionMode::Aker) {
                    float block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(MNKD[2]));
                    if (block_ratio == JSON_NOT_FOUND) {
                        float base_block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(4096));
                        block_ratio = base_block_ratio * MNKD[2] / 4096;
                    }
                    cd_block_num = int(MNKD[3] * block_ratio);
                    cd_block_num = min((cd_block_num < SM_NUM ? SM_NUM : (cd_block_num / SM_NUM) * SM_NUM), be_kernel1->gptbParams.ptb_end_block_pos);
                }
                mixcublasSgemm(mnk, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);

            } else if (lc_kernel->mixable == 2) { // cudnnConv
                std::vector<int> cudnnArgs = lc_kernel->getArgs();
                std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
                std::vector<int> MNKD = getTZGEMMGridDim(mnk[0], mnk[1], mnk[2]);
                if (mode == ExecutionMode::Tacker) cd_block_num = be_kernel1->gptbParams.ptb_end_block_pos;
                else if (mode == ExecutionMode::Aker) {
                    float block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(MNKD[2]));
                    if (block_ratio == JSON_NOT_FOUND) {
                        float base_block_ratio = fget_kernel_info("tzgemm_" + be_task1_name, std::to_string(4096));
                        block_ratio = base_block_ratio * MNKD[2] / 4096;
                    }
                    cd_block_num = int(MNKD[3] * block_ratio);
                    cd_block_num = min((cd_block_num < SM_NUM ? SM_NUM : (cd_block_num / SM_NUM) * SM_NUM), be_kernel1->gptbParams.ptb_end_block_pos);
                }
                mixcudnnConvolutionForward(cudnnArgs, be_kernel1, be_kernel1->gptbParams.ptb_start_block_pos, cd_block_num, stream);

            } else { // no mix
                lc_kernel->execute(stream);
            }
            cudaDeviceSynchronize();
            cudaStreamSynchronize(stream);
            cudaStreamSynchronize(0);
            auto end = clock();
            auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
            // if no throughput improve, use ori time
            bool do_mix = false;
            float improve = 0.0f;
            if (lc_kernel->mixable == 0) {
                qos_headroom -= lc_kernel_time_vec[lc_kernel_idx];
            }
            else if (duration > lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos) {
                qos_headroom -= lc_kernel_time_vec[lc_kernel_idx];
            } else {
                qos_headroom -= duration;
                do_mix = true;
                cd_block_num_executed += cd_block_num;
                mix_times += 1;
                time_earned += lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos - duration;
                improve = (lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos - duration) / (lc_kernel_time_vec[lc_kernel_idx] + be_task1_ori_time * cd_block_num / be_kernel1->gptbParams.ptb_end_block_pos);
            }

            // if (i == 4 && lc_kernel_idx < lc_kernel_time_vec.size() && lc_kernel->mixable != 0)  {
            //     printf("[%s]--[kernel]: %s, [ori_time]: %f, [mix_time]: %f, [qos_headroom]: %f, [improve]: %f%\n", do_mix ? "Yes" : "No",  lc_kernel->kernelName.c_str(), lc_kernel_time_vec[lc_kernel_idx], duration, qos_headroom, improve * 100);
            // }
            // if (i == 4 && lc_kernel_idx < lc_kernel_time_vec.size() && lc_kernel->mixable != 0)  {
            //     printf("[%s]--[kernel]: %s, [ori_time]: %f, [mix_time]: %f, [qos_headroom]: %f\n", do_mix ? "Yes" : "No",  lc_kernel->kernelName.c_str(), lc_kernel_time_vec[lc_kernel_idx], duration, qos_headroom);
            //     cin >> foo;
            // }
            lc_kernel_idx ++;
        }
        cudaProfilerStop();
        total_be_num = cd_block_num_executed;
    }

    printf("execed be blks: %d, mix times: %d/%d, time earned: %f\n", total_be_num, mix_times, mixable_times, time_earned);

    float stage1_be_time = 0.0f;

    stage1_be_time = total_be_num * be_task1_ori_time / be_task1->gptbParams.ptb_end_block_pos;

    if (ExecutionMode::Aker == mode) {
        printf("[Result] Aker throughput %f + %f = %f ms.\n", stage1_be_time, qos_headroom, stage1_be_time + qos_headroom);
    }
    else if (ExecutionMode::Tacker == mode) {
        printf("[Result] Tacker throughput %f + %f = %f ms.\n", stage1_be_time, qos_headroom, stage1_be_time + qos_headroom);
    }

    // my test
    // int dnn_idx = 6;
    // auto dnn_kernel = lc_task->kernels[dnn_idx];
    // assert(dnn_kernel->mixable == 2);
    // std::vector<int> cudnnArgs = dnn_kernel->getArgs();
    // std::vector<int> mnk = get_mnk_from_cudnn_args(cudnnArgs);
    // int M_INPUT = mnk[0];
    // int N_INPUT = mnk[1];
    // int K_INPUT = mnk[2];
    // int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	// int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	// int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;
    // auto ori_tzgemm = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // auto gptb_tzgemm_kernel = new GPTBKernel(
    //     1, 
    //     "tzgemm",
    //     "gptb_tzgemm", 
    //     ori_tzgemm,
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0,
    //     getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    // );
    // std::string mix_kernel_name = "tzgemm_" + be_task1_name;
    // int cd_blks = min(int(getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3] * fget_kernel_info("tzgemm_" + be_task1_name, "block_ratio")), be_kernel1->gptbParams.ptb_end_block_pos);

    // be_task1->gptbParams.ptb_end_block_pos = cd_blks;
    // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    // be_task1->execute(stream);
    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task1_ori_time, startKERNEL, stopKERNEL));
    // auto mix_kernel = new MixKernel(
    //     1, 
    //     mix_kernel_name, 
    //     be_kernel1,
    //     (gptb_tzgemm_kernel), 
    //     dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
    //     dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
    //     0,
    //     cd_blks,
    //     0,
    //     getTZGEMMGridDim(ori_tzgemm->m, ori_tzgemm->n, ori_tzgemm->k)[3]
    // );

    // cudaEventRecord(startKERNEL, 0);
    // ori_tzgemm->execute(nullptr);
    // cudaEventRecord(stopKERNEL, 0);
    // cudaEventSynchronize(stopKERNEL);
    // float ori_tz_time = 0.0f;
    // cudaEventElapsedTime(&ori_tz_time, startKERNEL, stopKERNEL);

    // float mix_time = 0.0f;
    // for (int i = 0; i < 10; ++i) {
    //     float tmp_time = 0.0f;
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    //     mix_kernel->execute(stream);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&tmp_time, startKERNEL, stopKERNEL));
    //     mix_time += tmp_time;
    //     printf("mix time: %f\n", tmp_time);
    // }
    // mix_time /= 10;
    // printf("\nori tz kernel: %f ms, ori_cd_kernel: %f ms, mix_kernel: %f ms\n", ori_tz_time, be_task1_ori_time, mix_time);
    // printf("M-%d, N-%d, K-%d, %s\n", ori_tzgemm->m, ori_tzgemm->n, ori_tzgemm->k, dnn_kernel->kernelName.c_str());
    // printf("cd blks range: %d - %d, tzgemm blks range: %d - %d\n", be_task1->gptbParams.ptb_start_block_pos, be_task1->gptbParams.ptb_end_block_pos, gptb_tzgemm_kernel->gptbParams.ptb_start_block_pos, gptb_tzgemm_kernel->gptbParams.ptb_end_block_pos);

    // free event
    CUDA_SAFE_CALL(cudaEventDestroy(startKERNEL));
    CUDA_SAFE_CALL(cudaEventDestroy(stopKERNEL));
}

TaskManager::~TaskManager() {
    // logger.INFO("TaskManager is destroyed!");
}

