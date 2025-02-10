// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nnfusion_rt.h"
#include <sstream>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>
#include <fstream>
#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
   #define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define MIN(a,b) ((a)>(b)?(b):(a))
char* group_0_CUDA_GPU0_allocator_memory_pool;
int32_t* Reshape_114_0;
float* OneHot_118_0;
float* Dot_122_0;
int32_t* Reshape_113_0;
int32_t* Reshape_117_0;
float* GatherV2_121_0;
float* Add_142_0;
float* Sum_149_0;
float* Divide_151_0;
float* Reshape_152_0;
float* Reshape_153_0;
float* Multiply_156_0;
float* Sum_157_0;
float* Rsqrt_164_0;
float* Reshape_165_0;
float* Add_175_0;
float* Reshape_176_0;
float* Dot_179_0;
float* Add_185_0;
float* Reshape_188_0;
float* Reshape_191_0;
float* Dot_178_0;
float* Add_183_0;
float* Reshape_187_0;
float* Reshape_190_0;
float* Dot_177_0;
float* Add_181_0;
float* Reshape_186_0;
float* Reshape_189_0;
float* BatchMatMul_192_0;
float* Broadcast_116_0;
float* Reshape_124_0;
int32_t* Reshape_115_0;
float* Convert_120_0;
float* Reshape_126_0;
float* Broadcast_127_0;
float* Multiply_145_0;
float* Multiply_148_0;
float* Reshape_195_0;
float* Broadcast_196_0;
float* Add_197_0;
float* Softmax_198_0;
float* BatchMatMul_199_0;
float* Reshape_200_0;
float* Reshape_201_0;
float* Dot_202_0;
float* Add_205_0;
float* Sum_206_0;
float* Divide_208_0;
float* Reshape_209_0;
float* Reshape_210_0;
float* Multiply_213_0;
float* Sum_214_0;
float* Rsqrt_221_0;
float* Reshape_222_0;
float* Add_232_0;
float* Dot_233_0;
float* Multiply_248_0;
float* Dot_249_0;
float* Add_252_0;
float* Sum_253_0;
float* Divide_255_0;
float* Reshape_256_0;
float* Reshape_257_0;
float* Multiply_260_0;
float* Sum_261_0;
float* Rsqrt_268_0;
float* Reshape_269_0;
float* Add_279_0;
float* Dot_282_0;
float* Add_288_0;
float* Reshape_291_0;
float* Reshape_294_0;
float* Dot_281_0;
float* Add_286_0;
float* Reshape_290_0;
float* Reshape_293_0;
float* Dot_280_0;
float* Add_284_0;
float* Reshape_289_0;
float* Reshape_292_0;
float* BatchMatMul_295_0;
float* Reshape_298_0;
float* Broadcast_299_0;
float* Add_300_0;
float* Softmax_301_0;
float* BatchMatMul_302_0;
float* Reshape_303_0;
float* Reshape_304_0;
float* Dot_305_0;
float* Add_308_0;
float* Sum_309_0;
float* Divide_311_0;
float* Reshape_312_0;
float* Reshape_313_0;
float* Multiply_316_0;
float* Sum_317_0;
float* Rsqrt_324_0;
float* Reshape_325_0;
float* Add_335_0;
float* Dot_336_0;
float* Multiply_351_0;
float* Dot_352_0;
float* Add_355_0;
float* Sum_356_0;
float* Divide_358_0;
float* Reshape_359_0;
float* Reshape_360_0;
float* Multiply_363_0;
float* Sum_364_0;
float* Rsqrt_371_0;
float* Reshape_372_0;
float* Add_382_0;
float* Reshape_383_0;
float* Slice_384_0;
float* Reshape_385_0;
float* Dot_386_0;
float* Tanh_389_0;
float* Dot_390_0;
float* Broadcast_391_0;
float* Add_392_0;
float* Softmax_393_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
__device__ __forceinline__ float mul(float x0, float x1)
{
    return x0 * x1;
}

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ static float reduceMax(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ static float reduceSum(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;

  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}
__device__ __forceinline__ float subtractf(float x0, float x1)
{
    return x0-x1;
}

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
__device__ __forceinline__ float convert(int32_t x0)
{
    return x0;
}
__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
cudnnHandle_t cudnn_handle_0;
cublasHandle_t cublas_handle_0;
__device__ __forceinline__ char  load(const char*  __restrict__ in, int i=0, bool b=true)
{
    char v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
} 
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
int num_SMs;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_110_0;
float* Constant_109_0;
float* Constant_46_0;
float* Constant_32_0;
float* Constant_31_0;
float* Constant_8_0;
float* Constant_4_0;
float* Constant_14_0;
float* Slice_119_0;
float* Reshape_123_0;
float* Reshape_140_0;
float* Constant_19_0;
float* Constant_18_0;
float* Constant_150_0;
float* Constant_22_0;
float* Reshape_161_0;
float* Constant_158_0;
float* Constant_37_0;
float* Constant_30_0;
float* Constant_29_0;
float* Constant_28_0;
float* Constant_27_0;
float* Constant_79_0;
float* Reshape_146_0;
float* Constant_40_0;
float* Reshape_143_0;
float* Constant_78_0;
float* Reshape_137_0;
float* Constant_39_0;
float* Reshape_134_0;
float* Constant_25_0;
float* Constant_45_0;
float* Constant_48_0;
float* Constant_47_0;
float* Constant_207_0;
float* Constant_51_0;
float* Reshape_218_0;
float* Constant_215_0;
float* Constant_60_0;
float* Constant_58_0;
float* Constant_56_0;
float* Constant_55_0;
float* Constant_53_0;
float* Constant_57_0;
float* Constant_54_0;
float* Constant_52_0;
float* Constant_59_0;
float* Constant_62_0;
float* Constant_61_0;
float* Constant_254_0;
float* Constant_65_0;
float* Reshape_265_0;
float* Constant_262_0;
float* Constant_85_0;
float* Constant_71_0;
float* Constant_70_0;
float* Constant_76_0;
float* Constant_69_0;
float* Constant_68_0;
float* Constant_67_0;
float* Constant_66_0;
float* Constant_84_0;
float* Constant_87_0;
float* Constant_86_0;
float* Constant_310_0;
float* Constant_90_0;
float* Reshape_321_0;
float* Constant_318_0;
float* Constant_99_0;
float* Constant_97_0;
float* Constant_95_0;
float* Constant_94_0;
float* Constant_92_0;
float* Constant_96_0;
float* Constant_93_0;
float* Constant_91_0;
float* Constant_98_0;
float* Constant_101_0;
float* Constant_100_0;
float* Constant_357_0;
float* Constant_104_0;
float* Reshape_368_0;
float* Constant_365_0;
float* Constant_111_0;
float* Constant_112_0;

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 3
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {64, 128}
#define NNFUSION_GRAPH_INPUT_DTYPE_1 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_1 {64, 128}
#define NNFUSION_GRAPH_INPUT_DTYPE_2 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_2 {64, 128}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {64, 1001}
#endif

// Node name:	Dot_390
// Description:	Dot
// Input:
//	- name: Tanh_389_0	type: float	shape: Shape{64, 1024}
//	- name: Constant_111_0	type: float	shape: Shape{1024, 1001}
// Output:
//	- name: Dot_390_0	type: float	shape: Shape{64, 1001}
void Dot_float_float_float_cuda_lib_Dot_390(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 64, 1024, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_32_0	type: float	shape: Shape{1024}
//	- name: Dot_179_0	type: float	shape: Shape{8192, 1024}
// Output:
//	- name: Add_185_0	type: float	shape: Shape{8192, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_184<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_32_0, Broadcast_184_0);
// Add_float_float_float_cuda_Add_185<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Dot_179_0, Broadcast_184_0, Add_185_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Add_7(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Add_7<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Sum_149
// Description:	Sum
// Input:
//	- name: Add_142_0	type: float	shape: Shape{64, 128, 1024}
// Output:
//	- name: Sum_149_0	type: float	shape: Shape{64, 128}
extern "C" __launch_bounds__(512) __global__ void Sum_float_float_cuda_Sum_149(float* input0, float* output0)
{

    int width = 1024;
    int block_size = 512;
    const int warp_size = 32;
    __shared__ float shm[warp_size];

    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int data_idx_offset = block_idx * width;

    float val = 0.0;
    for (int tidx = thread_idx; tidx < width; tidx += block_size) {
        int data_idx = tidx + data_idx_offset;
        val += input0[data_idx];
    }
    val = reduceSum(val, thread_idx, block_size, shm);
    if (thread_idx == 0) output0[block_idx] = val;


}
extern void Sum_float_float_cuda_Sum_149_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Sum_float_float_cuda_Sum_149<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_127
// Description:	Broadcast
// Input:
//	- name: Reshape_126_0	type: float	shape: Shape{64, 128}
// Output:
//	- name: Broadcast_127_0	type: float	shape: Shape{64, 128, 128}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_127(float* input0, float* output0)
{
    size_t nthreads = 1048576;uint32_t strides0 = 16384;
    uint32_t strides1 = 128;
    uint32_t strides2 = 1;
    int stride_magic0 = 1;
    int stride_magic1 = 1;
    int stride_magic2 = 1;
    int stride_shift0 = 14;
    int stride_shift1 = 7;
    int stride_shift2 = 0;
    uint32_t reduced_strides0 = 128;
    uint32_t reduced_strides1 = 0;
    uint32_t reduced_strides2 = 1;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        int coordinate2 = division_by_invariant_multiplication(coordinate_product, stride_magic2, stride_shift2);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void Broadcast_float_float_cuda_Broadcast_127_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_127<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_196
// Description:	Broadcast
// Input:
//	- name: Reshape_195_0	type: float	shape: Shape{64, 128, 128}
// Output:
//	- name: Broadcast_196_0	type: float	shape: Shape{64, 16, 128, 128}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_196(float* input0, float* output0)
{
    size_t nthreads = 16777216;uint32_t strides0 = 262144;
    uint32_t strides1 = 16384;
    uint32_t strides2 = 128;
    uint32_t strides3 = 1;
    int stride_magic0 = 1;
    int stride_magic1 = 1;
    int stride_magic2 = 1;
    int stride_magic3 = 1;
    int stride_shift0 = 18;
    int stride_shift1 = 14;
    int stride_shift2 = 7;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 16384;
    uint32_t reduced_strides1 = 0;
    uint32_t reduced_strides2 = 128;
    uint32_t reduced_strides3 = 1;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        int coordinate2 = division_by_invariant_multiplication(coordinate_product, stride_magic2, stride_shift2);
        coordinate_product -= (coordinate2 * strides2);
        int coordinate3 = division_by_invariant_multiplication(coordinate_product, stride_magic3, stride_shift3);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        reduced_idx += coordinate3 * reduced_strides3;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void Broadcast_float_float_cuda_Broadcast_196_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_196<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_116
// Description:	Broadcast
// Input:
//	- name: Constant_25_0	type: float	shape: Shape{}
// Output:
//	- name: Broadcast_116_0	type: float	shape: Shape{64, 128, 1}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_116(float* input0, float* output0)
{
    size_t nthreads = 8192;uint32_t strides0 = 128;
    uint32_t strides1 = 1;
    uint32_t strides2 = 1;
    int stride_magic0 = 1;
    int stride_magic1 = 1;
    int stride_magic2 = 1;
    int stride_shift0 = 7;
    int stride_shift1 = 0;
    int stride_shift2 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 0;
    uint32_t reduced_strides2 = 0;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        int coordinate2 = division_by_invariant_multiplication(coordinate_product, stride_magic2, stride_shift2);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void Broadcast_float_float_cuda_Broadcast_116_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_116<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: Constant_59_0	type: float	shape: Shape{4096, 1024}
void Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_59_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16777216];
    bin_file.read(tmp_mem, 16777216);
    cudaMemcpyAsync(output0, tmp_mem, 16777216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_101
// Description:	Constant
// Input:
// Output:
//	- name: Constant_101_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_101_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: Constant_61_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_65
// Description:	Constant
// Input:
// Output:
//	- name: Constant_65_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_65_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: Constant_93_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_85
// Description:	Constant
// Input:
// Output:
//	- name: Constant_85_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_85_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: Constant_37_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: Constant_84_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: Constant_87_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_87_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_86
// Description:	Constant
// Input:
// Output:
//	- name: Constant_86_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_86_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: Constant_67_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_67_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: Constant_91_0	type: float	shape: Shape{1024, 4096}
void Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16777216];
    bin_file.read(tmp_mem, 16777216);
    cudaMemcpyAsync(output0, tmp_mem, 16777216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: Constant_100_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_100_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_98
// Description:	Constant
// Input:
// Output:
//	- name: Constant_98_0	type: float	shape: Shape{4096, 1024}
void Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_98_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16777216];
    bin_file.read(tmp_mem, 16777216);
    cudaMemcpyAsync(output0, tmp_mem, 16777216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Broadcast_391
// Description:	Broadcast
// Input:
//	- name: Constant_112_0	type: float	shape: Shape{1001}
// Output:
//	- name: Broadcast_391_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_391(float* input0, float* output0)
{
    size_t nthreads = 64064;uint32_t strides0 = 1001;
    uint32_t strides1 = 1;
    int stride_magic0 = 1098413215;
    int stride_magic1 = 1;
    int stride_shift0 = 8;
    int stride_shift1 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void Broadcast_float_float_cuda_Broadcast_391_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_391<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_357
// Description:	Constant
// Input:
// Output:
//	- name: Constant_357_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_357(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_357_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_357_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_158
// Description:	Constant
// Input:
// Output:
//	- name: Constant_158_0	type: float	shape: Shape{64, 128}
void Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: Constant_92_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_92_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_318
// Description:	Constant
// Input:
// Output:
//	- name: Constant_318_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_318(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_318_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_318_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: Constant_68_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_68_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_104_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_95
// Description:	Constant
// Input:
// Output:
//	- name: Constant_95_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_95_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: Constant_19_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: Constant_96_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_96_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_97
// Description:	Constant
// Input:
// Output:
//	- name: Constant_97_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_97_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_112_0	type: float	shape: Shape{1001}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_90
// Description:	Constant
// Input:
// Output:
//	- name: Constant_90_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: Constant_47_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: Constant_57_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{512, 1024}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: Constant_52_0	type: float	shape: Shape{1024, 4096}
void Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16777216];
    bin_file.read(tmp_mem, 16777216);
    cudaMemcpyAsync(output0, tmp_mem, 16777216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_110
// Description:	Constant
// Input:
// Output:
//	- name: Constant_110_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_110_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: Constant_76_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: Constant_4_0	type: float	shape: Shape{30522, 1024}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[125018112];
    bin_file.read(tmp_mem, 125018112);
    cudaMemcpyAsync(output0, tmp_mem, 125018112, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_22
// Description:	Constant
// Input:
// Output:
//	- name: Constant_22_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_22_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_66
// Description:	Constant
// Input:
// Output:
//	- name: Constant_66_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_66_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_150
// Description:	Constant
// Input:
// Output:
//	- name: Constant_150_0	type: float	shape: Shape{64, 128}
void Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_69
// Description:	Constant
// Input:
// Output:
//	- name: Constant_69_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_69_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_215
// Description:	Constant
// Input:
// Output:
//	- name: Constant_215_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_215(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_215_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_215_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_111
// Description:	Constant
// Input:
// Output:
//	- name: Constant_111_0	type: float	shape: Shape{1024, 1001}
void Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_111_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4100096];
    bin_file.read(tmp_mem, 4100096);
    cudaMemcpyAsync(output0, tmp_mem, 4100096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_70
// Description:	Constant
// Input:
// Output:
//	- name: Constant_70_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_70_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_18_0	type: float	shape: Shape{1024}
//	- name: Reshape_153_0	type: float	shape: Shape{64, 128}
//	- name: Reshape_165_0	type: float	shape: Shape{64, 128}
//	- name: Constant_19_0	type: float	shape: Shape{1024}
//	- name: Add_142_0	type: float	shape: Shape{64, 128, 1024}
// Output:
//	- name: Add_175_0	type: float	shape: Shape{64, 128, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_173<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_18_0, Broadcast_173_0);
// Broadcast_float_float_cuda_Broadcast_154<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_153_0, Broadcast_154_0);
// Broadcast_float_float_cuda_Broadcast_166<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_165_0, Broadcast_166_0);
// Broadcast_float_float_cuda_Broadcast_167<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_19_0, Broadcast_167_0);
// Multiply_float_float_float_cuda_Multiply_168<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_166_0, Broadcast_167_0, Multiply_168_0);
// Multiply_float_float_float_cuda_Multiply_172<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_154_0, Multiply_168_0, Multiply_172_0);
// Subtract_float_float_float_cuda_Subtract_174<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_173_0, Multiply_172_0, Subtract_174_0);
// Multiply_float_float_float_cuda_Multiply_169<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_142_0, Multiply_168_0, Multiply_169_0);
// Add_float_float_float_cuda_Add_175<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_169_0, Subtract_174_0, Add_175_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = input1[tid / 1024];
    float temp2 = input2[tid / 1024];
    float temp3 = input3[tid % 1024];
    float temp4 = mul(temp2, temp3);
    float temp5 = mul(temp1, temp4);
    float temp6 = subtractf(temp0, temp5);
    float temp7 = mul(input4[tid], temp4);
    float temp8 = add(temp7, temp6);
    output0[tid] = temp8;

}
extern void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: Constant_58_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_79
// Description:	Constant
// Input:
// Output:
//	- name: Constant_79_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_79_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: Constant_109_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_109_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: Constant_45_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_207
// Description:	Constant
// Input:
// Output:
//	- name: Constant_207_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_207(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_207_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_207_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: Constant_48_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_48_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_99
// Description:	Constant
// Input:
// Output:
//	- name: Constant_99_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_99_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_262
// Description:	Constant
// Input:
// Output:
//	- name: Constant_262_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_262(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_262_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_262_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: Constant_32_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: Constant_51_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_60
// Description:	Constant
// Input:
// Output:
//	- name: Constant_60_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_60_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_71
// Description:	Constant
// Input:
// Output:
//	- name: Constant_71_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_71_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_254
// Description:	Constant
// Input:
// Output:
//	- name: Constant_254_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_254_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: Constant_78_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_78_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_56
// Description:	Constant
// Input:
// Output:
//	- name: Constant_56_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_56_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_94
// Description:	Constant
// Input:
// Output:
//	- name: Constant_94_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_37_0	type: float	shape: Shape{}
//	- name: BatchMatMul_192_0	type: float	shape: Shape{64, 16, 128, 128}
//	- name: Broadcast_196_0	type: float	shape: Shape{64, 16, 128, 128}
// Output:
//	- name: Add_197_0	type: float	shape: Shape{64, 16, 128, 128}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_193<<<dim3(262144, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_37_0, Broadcast_193_0);
// Multiply_float_float_float_cuda_Multiply_194<<<dim3(32768, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchMatMul_192_0, Broadcast_193_0, Multiply_194_0);
// Add_float_float_float_cuda_Add_197<<<dim3(32768, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_194_0, Broadcast_196_0, Add_197_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = mul(input1[tid], temp0);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: Constant_55_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_55_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: Constant_53_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_310
// Description:	Constant
// Input:
// Output:
//	- name: Constant_310_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_310(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_310_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_310_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: Constant_40_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	OneHot_118
// Description:	OneHot
// Input:
//	- name: Reshape_114_0	type: int32_t	shape: Shape{8192}
// Output:
//	- name: OneHot_118_0	type: float	shape: Shape{8192, 2}
extern "C" __launch_bounds__(64) __global__ void OneHot_int32_t_float_cuda_OneHot_118(int32_t* input0, float* output0)
{
    {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= 8192)
                return;
            for (int i = 0; i < 2; ++i)
                output0[idx * 2 + i] = 0.000000000000000000000000e+00;
            output0[idx * 2 + (int)input0[idx]] = 1.000000000000000000000000e+00;

    }

}
extern void OneHot_int32_t_float_cuda_OneHot_118_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    OneHot_int32_t_float_cuda_OneHot_118<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Reshape_153_0	type: float	shape: Shape{64, 128}
//	- name: Add_142_0	type: float	shape: Shape{64, 128, 1024}
// Output:
//	- name: Multiply_156_0	type: float	shape: Shape{64, 128, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_154<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_153_0, Broadcast_154_0);
// Subtract_float_float_float_cuda_Subtract_155<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_142_0, Broadcast_154_0, Subtract_155_0);
// Multiply_float_float_float_cuda_Multiply_156<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_155_0, Subtract_155_0, Multiply_156_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid / 1024];
    float temp1 = subtractf(input1[tid], temp0);
    float temp2 = mul(temp1, temp1);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Softmax_393
// Description:	Softmax
// Input:
//	- name: Add_392_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Softmax_393_0	type: float	shape: Shape{64, 1001}
void Softmax_float_float_cuda_lib_Softmax_393(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1, 1, 1001));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1, 1, 1001));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor_desc, input0, &beta, output_tensor_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));

}
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: Constant_25_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	BatchMatMul_192
// Description:	BatchMatMul
// Input:
//	- name: Reshape_189_0	type: float	shape: Shape{64, 16, 128, 64}
//	- name: Reshape_190_0	type: float	shape: Shape{64, 16, 128, 64}
// Output:
//	- name: BatchMatMul_192_0	type: float	shape: Shape{64, 16, 128, 128}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 128, 128, 64,
                                    &alpha, input1, 64, 8192, input0, 64, 8192,
                                    &beta, output0, 128, 16384, 1024));
                            
    }

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: GatherV2_121_0	type: float	shape: Shape{8192, 1024}
//	- name: Dot_122_0	type: float	shape: Shape{8192, 1024}
//	- name: Reshape_140_0	type: float	shape: Shape{128, 1024}
// Output:
//	- name: Add_142_0	type: float	shape: Shape{64, 128, 1024}
// Fused functions:
// Reshape_float_float_cuda_lib_Reshape_129(GatherV2_121_0, Reshape_129_0);
// Reshape_float_float_cuda_lib_Reshape_130(Dot_122_0, Reshape_130_0);
// Add_float_float_float_cuda_Add_133<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_129_0, Reshape_130_0, Add_133_0);
// Broadcast_float_float_cuda_Broadcast_141<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_140_0, Broadcast_141_0);
// Add_float_float_float_cuda_Add_142<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_133_0, Broadcast_141_0, Add_142_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = input2[tid % 131072];
    float temp2 = add(temp0, temp1);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Dot_179
// Description:	Dot
// Input:
//	- name: Reshape_176_0	type: float	shape: Shape{8192, 1024}
//	- name: Constant_31_0	type: float	shape: Shape{1024, 1024}
// Output:
//	- name: Dot_179_0	type: float	shape: Shape{8192, 1024}
void Dot_float_float_float_cuda_lib_Dot_179(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 8192, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: Constant_31_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Result_394
// Description:	Result
// Input:
//	- name: Softmax_393_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Result_394_0	type: float	shape: Shape{64, 1001}
void Result_float_float_cuda_lib_Result_394(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Reshape_191
// Description:	Reshape
// Input:
//	- name: Reshape_188_0	type: float	shape: Shape{64, 128, 16, 64}
// Output:
//	- name: Reshape_191_0	type: float	shape: Shape{64, 16, 128, 64}
extern "C" __launch_bounds__(64) __global__ void Reshape_float_float_cuda_Reshape_191(float* input0, float* output0)
{
    uint32_t input_strides0 = 131072;
    uint32_t input_strides1 = 1024;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 131072;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 8192;
    uint32_t trans_strides3 = 1;
    size_t n = 8388608;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        uint32_t input_idx = tid;
        uint32_t output_idx = 0;
        output_idx += (input_idx / input_strides0) * trans_strides0;
        input_idx %= input_strides0;
        output_idx += (input_idx / input_strides1) * trans_strides1;
        input_idx %= input_strides1;
        output_idx += (input_idx / input_strides2) * trans_strides2;
        input_idx %= input_strides2;
        output_idx += (input_idx / input_strides3) * trans_strides3;
        output0[output_idx] = input0[tid];
    }

}
extern void Reshape_float_float_cuda_Reshape_191_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_191<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Divide_151
// Description:	Divide
// Input:
//	- name: Sum_149_0	type: float	shape: Shape{64, 128}
//	- name: Constant_150_0	type: float	shape: Shape{64, 128}
// Output:
//	- name: Divide_151_0	type: float	shape: Shape{64, 128}
extern "C" __launch_bounds__(512) __global__ void Divide_float_float_float_cuda_Divide_151(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = fdividef(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void Divide_float_float_float_cuda_Divide_151_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Divide_float_float_float_cuda_Divide_151<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	GatherV2_121
// Description:	GatherV2
// Input:
//	- name: Constant_4_0	type: float	shape: Shape{30522, 1024}
//	- name: Reshape_117_0	type: int32_t	shape: Shape{8192}
// Output:
//	- name: GatherV2_121_0	type: float	shape: Shape{8192, 1024}
extern "C" __launch_bounds__(64) __global__ void GatherV2_float_int32_t_float_cuda_GatherV2_121(float* input0, int32_t* input1, float* output0)
{
    float* params = input0;
    int32_t* indices = input1;
    float* out = output0;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8388608)
    {
        uint32_t batch_i = 0;
        uint32_t indices_i = 0;
        uint32_t slice_i = 0;
        indices_i = i / 1024;
        slice_i = i - indices_i * 1024;
        uint32_t gather_i = __ldg(indices + indices_i);
        if (gather_i >= 30522)
           out[i] = 0;
        else
        {
            uint32_t params_i = (batch_i * 30522 + gather_i) * 1024 + slice_i;
            out[i] = __ldg(params + params_i);
        }
    }

}
extern void GatherV2_float_int32_t_float_cuda_GatherV2_121_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int32_t* input1, float* output0) {
    GatherV2_float_int32_t_float_cuda_GatherV2_121<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_58_0	type: float	shape: Shape{}
//	- name: Constant_57_0	type: float	shape: Shape{}
//	- name: Constant_56_0	type: float	shape: Shape{}
//	- name: Constant_55_0	type: float	shape: Shape{}
//	- name: Constant_53_0	type: float	shape: Shape{4096}
//	- name: Dot_233_0	type: float	shape: Shape{8192, 4096}
//	- name: Constant_54_0	type: float	shape: Shape{}
// Output:
//	- name: Multiply_248_0	type: float	shape: Shape{8192, 4096}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_246<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_58_0, Broadcast_246_0);
// Broadcast_float_float_cuda_Broadcast_244<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_57_0, Broadcast_244_0);
// Broadcast_float_float_cuda_Broadcast_241<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_56_0, Broadcast_241_0);
// Broadcast_float_float_cuda_Broadcast_238<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_55_0, Broadcast_238_0);
// Broadcast_float_float_cuda_Broadcast_234<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_53_0, Broadcast_234_0);
// Add_float_float_float_cuda_Add_235<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Dot_233_0, Broadcast_234_0, Add_235_0);
// Broadcast_float_float_cuda_Broadcast_236<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_54_0, Broadcast_236_0);
// Power_float_float_float_cuda_Power_237<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Broadcast_236_0, Power_237_0);
// Multiply_float_float_float_cuda_Multiply_239<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_238_0, Power_237_0, Multiply_239_0);
// Add_float_float_float_cuda_Add_240<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Multiply_239_0, Add_240_0);
// Multiply_float_float_float_cuda_Multiply_242<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_241_0, Add_240_0, Multiply_242_0);
// Tanh_float_float_cuda_Tanh_243<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_242_0, Tanh_243_0);
// Add_float_float_float_cuda_Add_245<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_244_0, Tanh_243_0, Add_245_0);
// Multiply_float_float_float_cuda_Multiply_247<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_246_0, Add_245_0, Multiply_247_0);
// Multiply_float_float_float_cuda_Multiply_248<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Multiply_247_0, Multiply_248_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid % 1];
    float temp2 = input2[tid % 1];
    float temp3 = input3[tid % 1];
    float temp4 = input4[tid % 4096];
    float temp5 = add(input5[tid], temp4);
    float temp6 = input6[tid % 1];
    float temp7 = powf(temp5, temp6);
    float temp8 = mul(temp3, temp7);
    float temp9 = add(temp5, temp8);
    float temp10 = mul(temp2, temp9);
    float temp11 = tanhf(temp10);
    float temp12 = add(temp1, temp11);
    float temp13 = mul(temp0, temp12);
    float temp14 = mul(temp5, temp13);
    output0[tid] = temp14;

}
extern void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0) {
    FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0);
}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: Constant_8_0	type: float	shape: Shape{2, 1024}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_46_0	type: float	shape: Shape{1024}
//	- name: Dot_202_0	type: float	shape: Shape{8192, 1024}
//	- name: Reshape_176_0	type: float	shape: Shape{8192, 1024}
// Output:
//	- name: Add_205_0	type: float	shape: Shape{8192, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_203<<<dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_46_0, Broadcast_203_0);
// Add_float_float_float_cuda_Add_204<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Dot_202_0, Broadcast_203_0, Add_204_0);
// Add_float_float_float_cuda_Add_205<<<dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_204_0, Reshape_176_0, Add_205_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: Constant_39_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_110_0	type: float	shape: Shape{1024}
//	- name: Dot_386_0	type: float	shape: Shape{64, 1024}
// Output:
//	- name: Tanh_389_0	type: float	shape: Shape{64, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_387<<<dim3(1024, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_110_0, Broadcast_387_0);
// Add_float_float_float_cuda_Add_388<<<dim3(128, 1, 1), dim3(512, 1, 1), 0, 0>>>(Dot_386_0, Broadcast_387_0, Add_388_0);
// Tanh_float_float_cuda_Tanh_389<<<dim3(128, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_388_0, Tanh_389_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    float temp2 = tanhf(temp1);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convert_120
// Description:	Convert
// Input:
//	- name: Reshape_115_0	type: int32_t	shape: Shape{64, 1, 128}
// Output:
//	- name: Convert_120_0	type: float	shape: Shape{64, 1, 128}
extern "C" __launch_bounds__(512) __global__ void Convert_int32_t_float_cuda_Convert_120(int32_t* input0, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = convert(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern void Convert_int32_t_float_cuda_Convert_120_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    Convert_int32_t_float_cuda_Convert_120<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Reshape_134_0	type: float	shape: Shape{1}
//	- name: Reshape_124_0	type: float	shape: Shape{64, 128}
//	- name: Broadcast_127_0	type: float	shape: Shape{64, 128, 128}
//	- name: Reshape_143_0	type: float	shape: Shape{1}
//	- name: Reshape_137_0	type: float	shape: Shape{1}
//	- name: Reshape_146_0	type: float	shape: Shape{1}
// Output:
//	- name: Multiply_148_0	type: float	shape: Shape{64, 1, 128, 128}
//	- name: Multiply_145_0	type: float	shape: Shape{64, 1, 128, 128}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_135<<<dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_134_0, Broadcast_135_0);
// Broadcast_float_float_cuda_Broadcast_125<<<dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_124_0, Broadcast_125_0);
// Multiply_float_float_float_cuda_Multiply_128<<<dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_125_0, Broadcast_127_0, Multiply_128_0);
// Reshape_float_float_cuda_lib_Reshape_131(Multiply_128_0, Reshape_131_0);
// Subtract_float_float_float_cuda_Subtract_136<<<dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_135_0, Reshape_131_0, Subtract_136_0);
// Broadcast_float_float_cuda_Broadcast_144<<<dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_143_0, Broadcast_144_0);
// Multiply_float_float_float_cuda_Multiply_145<<<dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_136_0, Broadcast_144_0, Multiply_145_0);
// Broadcast_float_float_cuda_Broadcast_138<<<dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_137_0, Broadcast_138_0);
// Subtract_float_float_float_cuda_Subtract_139<<<dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_138_0, Reshape_131_0, Subtract_139_0);
// Broadcast_float_float_cuda_Broadcast_147<<<dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_146_0, Broadcast_147_0);
// Multiply_float_float_float_cuda_Multiply_148<<<dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_139_0, Broadcast_147_0, Multiply_148_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid / 128];
    float temp2 = mul(temp1, input2[tid]);
    float temp3 = subtractf(temp0, temp2);
    float temp4 = input3[tid % 1];
    float temp5 = mul(temp3, temp4);
    float temp6 = input4[tid % 1];
    float temp7 = subtractf(temp6, temp2);
    float temp8 = input5[tid % 1];
    float temp9 = mul(temp7, temp8);
    output1[tid] = temp5;
    output0[tid] = temp9;

}
extern void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: Constant_54_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Dot_386
// Description:	Dot
// Input:
//	- name: Reshape_385_0	type: float	shape: Shape{64, 1024}
//	- name: Constant_109_0	type: float	shape: Shape{1024, 1024}
// Output:
//	- name: Dot_386_0	type: float	shape: Shape{64, 1024}
void Dot_float_float_float_cuda_lib_Dot_386(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 64, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: Constant_29_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Softmax_198
// Description:	Softmax
// Input:
//	- name: Add_197_0	type: float	shape: Shape{64, 16, 128, 128}
// Output:
//	- name: Softmax_198_0	type: float	shape: Shape{64, 16, 128, 128}
void Softmax_float_float_cuda_lib_Softmax_198(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 131072, 1, 1, 128));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 131072, 1, 1, 128));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor_desc, input0, &beta, output_tensor_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));

}
// Node name:	Dot_122
// Description:	Dot
// Input:
//	- name: OneHot_118_0	type: float	shape: Shape{8192, 2}
//	- name: Constant_8_0	type: float	shape: Shape{2, 1024}
// Output:
//	- name: Dot_122_0	type: float	shape: Shape{8192, 1024}
void Dot_float_float_float_cuda_lib_Dot_122(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 8192, 2, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 2, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Sum_157_0	type: float	shape: Shape{64, 128}
//	- name: Constant_158_0	type: float	shape: Shape{64, 128}
//	- name: Reshape_161_0	type: float	shape: Shape{1}
// Output:
//	- name: Rsqrt_164_0	type: float	shape: Shape{64, 128, 1}
// Fused functions:
// Divide_float_float_float_cuda_Divide_159<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Sum_157_0, Constant_158_0, Divide_159_0);
// Reshape_float_float_cuda_lib_Reshape_160(Divide_159_0, Reshape_160_0);
// Broadcast_float_float_cuda_Broadcast_162<<<dim3(128, 1, 1), dim3(64, 1, 1), 0, 0>>>(Reshape_161_0, Broadcast_162_0);
// Add_float_float_float_cuda_Add_163<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_160_0, Broadcast_162_0, Add_163_0);
// Rsqrt_float_float_cuda_Rsqrt_164<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_163_0, Rsqrt_164_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = fdividef(input0[tid], input1[tid]);
    float temp1 = input2[tid % 1];
    float temp2 = add(temp0, temp1);
    float temp3 = rsqrtf(temp2);
    output0[tid] = temp3;

}
extern void FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Reshape_114
// Description:	Reshape
// Input:
//	- name: Parameter_2_0	type: int32_t	shape: Shape{64, 128}
// Output:
//	- name: Reshape_114_0	type: int32_t	shape: Shape{8192}
void Reshape_int32_t_int32_t_cuda_lib_Reshape_114(cudaStream_t stream, int32_t* input0, int32_t* output0)
{
    if (input0 != output0) {
       cudaMemcpyAsync(output0, input0, 8192 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    }

}
// Node name:	Slice_384
// Description:	Slice
// Input:
//	- name: Reshape_383_0	type: float	shape: Shape{64, 128, 1024}
// Output:
//	- name: Slice_384_0	type: float	shape: Shape{64, 1, 1024}
extern "C" __launch_bounds__(64) __global__ void Slice_float_float_cuda_Slice_384(float* input0, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 65536)
    {
        uint32_t input_strides[] = {131072, 1024, 1};
        uint32_t output_strides[] = {1024, 1024, 1};
        uint32_t lower_bounds[] = {0, 0, 0};
        uint32_t slice_strides[] = {1, 1, 1};
        uint32_t input_idx = 0;
        uint32_t output_idx = tid;
        input_idx += (((output_idx / output_strides[0]) * slice_strides[0]) + lower_bounds[0]) * input_strides[0];
        output_idx %= output_strides[0];
        input_idx += (((output_idx / output_strides[1]) * slice_strides[1]) + lower_bounds[1]) * input_strides[1];
        output_idx %= output_strides[1];
        input_idx += (((output_idx / output_strides[2]) * slice_strides[2]) + lower_bounds[2]) * input_strides[2];
        output0[tid] = input0[input_idx];
    }

}
extern void Slice_float_float_cuda_Slice_384_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Slice_float_float_cuda_Slice_384<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_365
// Description:	Constant
// Input:
// Output:
//	- name: Constant_365_0	type: float	shape: Shape{8192}
void Constant_float_cuda_Constant_365(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_365_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_365_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Dot_249
// Description:	Dot
// Input:
//	- name: Multiply_248_0	type: float	shape: Shape{8192, 4096}
//	- name: Constant_59_0	type: float	shape: Shape{4096, 1024}
// Output:
//	- name: Dot_249_0	type: float	shape: Shape{8192, 1024}
void Dot_float_float_float_cuda_lib_Dot_249(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 8192, 4096, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	BatchMatMul_199
// Description:	BatchMatMul
// Input:
//	- name: Softmax_198_0	type: float	shape: Shape{64, 16, 128, 128}
//	- name: Reshape_191_0	type: float	shape: Shape{64, 16, 128, 64}
// Output:
//	- name: BatchMatMul_199_0	type: float	shape: Shape{64, 16, 128, 64}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 128, 128,
                                    &alpha, input1, 64, 8192, input0, 128, 16384,
                                    &beta, output0, 64, 8192, 1024));
                            
    }

}
// Node name:	Reshape_200
// Description:	Reshape
// Input:
//	- name: BatchMatMul_199_0	type: float	shape: Shape{64, 16, 128, 64}
// Output:
//	- name: Reshape_200_0	type: float	shape: Shape{64, 128, 16, 64}
extern "C" __launch_bounds__(64) __global__ void Reshape_float_float_cuda_Reshape_200(float* input0, float* output0)
{
    uint32_t input_strides0 = 131072;
    uint32_t input_strides1 = 8192;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 131072;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 1024;
    uint32_t trans_strides3 = 1;
    size_t n = 8388608;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        uint32_t input_idx = tid;
        uint32_t output_idx = 0;
        output_idx += (input_idx / input_strides0) * trans_strides0;
        input_idx %= input_strides0;
        output_idx += (input_idx / input_strides1) * trans_strides1;
        input_idx %= input_strides1;
        output_idx += (input_idx / input_strides2) * trans_strides2;
        input_idx %= input_strides2;
        output_idx += (input_idx / input_strides3) * trans_strides3;
        output0[output_idx] = input0[tid];
    }

}
extern void Reshape_float_float_cuda_Reshape_200_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_200<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_62
// Description:	Constant
// Input:
// Output:
//	- name: Constant_62_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_62_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Add_392
// Description:	Add
// Input:
//	- name: Dot_390_0	type: float	shape: Shape{64, 1001}
//	- name: Broadcast_391_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Add_392_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void Add_float_float_float_cuda_Add_392(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 64 + threadIdx.x] = add(input0[blockIdx.x * 64 + threadIdx.x], input1[blockIdx.x * 64 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_392<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Dot_233
// Description:	Dot
// Input:
//	- name: Add_232_0	type: float	shape: Shape{8192, 1024}
//	- name: Constant_52_0	type: float	shape: Shape{1024, 4096}
// Output:
//	- name: Dot_233_0	type: float	shape: Shape{8192, 4096}
void Dot_float_float_float_cuda_lib_Dot_233(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 8192, 1024, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 4096));

}

extern "C" void cuda_init()
{
CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:584758592
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,348225536));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 348225536));
Reshape_114_0 = (int32_t*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
OneHot_118_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Dot_122_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+98304);
Reshape_113_0 = (int32_t*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_117_0 = (int32_t*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
GatherV2_121_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33652736);
Add_142_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67207168);
Sum_149_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_151_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_152_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_153_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Multiply_156_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Sum_157_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33587200);
Rsqrt_164_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Reshape_165_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Add_175_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Reshape_176_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Dot_179_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Add_185_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Reshape_188_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Reshape_191_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Dot_178_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Add_183_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100728832);
Reshape_187_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100728832);
Reshape_190_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Dot_177_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100728832);
Add_181_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+134283264);
Reshape_186_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+134283264);
Reshape_189_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100728832);
BatchMatMul_192_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+134283264);
Broadcast_116_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_124_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_115_0 = (int32_t*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Convert_120_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Reshape_126_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Broadcast_127_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67207168);
Multiply_145_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+71401472);
Multiply_148_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75595776);
Reshape_195_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+71401472);
Broadcast_196_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+201392128);
Add_197_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+268500992);
Softmax_198_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+268500992);
BatchMatMul_199_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Reshape_200_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Reshape_201_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Dot_202_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Add_205_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Sum_206_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_208_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_209_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_210_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Multiply_213_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Sum_214_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33587200);
Rsqrt_221_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Reshape_222_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Add_232_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Dot_233_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Multiply_248_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+214007808);
Dot_249_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Add_252_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Sum_253_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_255_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_256_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_257_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Multiply_260_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Sum_261_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33587200);
Rsqrt_268_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Reshape_269_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Add_279_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Dot_282_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Add_288_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Reshape_291_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Reshape_294_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Dot_281_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Add_286_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+113344512);
Reshape_290_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+113344512);
Reshape_293_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Dot_280_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+113344512);
Add_284_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+146898944);
Reshape_289_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+146898944);
Reshape_292_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+113344512);
BatchMatMul_295_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+146898944);
Reshape_298_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75595776);
Broadcast_299_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+79790080);
Add_300_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+214007808);
Softmax_301_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+214007808);
BatchMatMul_302_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Reshape_303_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Reshape_304_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Dot_305_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Add_308_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Sum_309_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_311_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_312_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_313_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Multiply_316_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Sum_317_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33587200);
Rsqrt_324_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Reshape_325_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Add_335_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Dot_336_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Multiply_351_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+167837696);
Dot_352_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Add_355_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+67174400);
Sum_356_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_358_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_359_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_360_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Multiply_363_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Sum_364_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33587200);
Rsqrt_371_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Reshape_372_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32768);
Add_382_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Reshape_383_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Slice_384_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Reshape_385_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33619968);
Dot_386_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Tanh_389_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+262144);
Dot_390_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_391_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+256256);
Add_392_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Softmax_393_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+256256);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,236533056));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 236533056));
Constant_110_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_109_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4096);
Constant_46_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4198400);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4202496);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4206592);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8400896);
Constant_4_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8409088);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
Slice_119_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
Reshape_123_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
Reshape_140_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
Constant_19_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135524352);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135528448);
Constant_150_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135532544);
Constant_22_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135565312);
Reshape_161_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135565312);
Constant_158_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135565376);
Constant_37_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135598144);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135598208);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135602304);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+139796608);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+139800704);
Constant_79_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995008);
Reshape_146_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995008);
Constant_40_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995072);
Reshape_143_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995072);
Constant_78_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995136);
Reshape_137_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995136);
Constant_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995200);
Reshape_134_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995200);
Constant_25_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995264);
Constant_45_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+143995328);
Constant_48_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148189632);
Constant_47_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148193728);
Constant_207_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148197824);
Constant_51_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148230592);
Reshape_218_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148230592);
Constant_215_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148230656);
Constant_60_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148263424);
Constant_58_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148267520);
Constant_56_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148267584);
Constant_55_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148267648);
Constant_53_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148267712);
Constant_57_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148284096);
Constant_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148284160);
Constant_52_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+148284224);
Constant_59_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+165061440);
Constant_62_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181838656);
Constant_61_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181842752);
Constant_254_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181846848);
Constant_65_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181879616);
Reshape_265_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181879616);
Constant_262_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181879680);
Constant_85_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181912448);
Constant_71_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181916544);
Constant_70_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+181920640);
Constant_76_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+186114944);
Constant_69_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+186115008);
Constant_68_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+186119104);
Constant_67_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+190313408);
Constant_66_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+190317504);
Constant_84_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+194511808);
Constant_87_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198706112);
Constant_86_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198710208);
Constant_310_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198714304);
Constant_90_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198747072);
Reshape_321_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198747072);
Constant_318_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198747136);
Constant_99_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198779904);
Constant_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198784000);
Constant_95_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198784064);
Constant_94_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198784128);
Constant_92_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198784192);
Constant_96_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198800576);
Constant_93_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198800640);
Constant_91_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+198800704);
Constant_98_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+215577920);
Constant_101_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232355136);
Constant_100_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232359232);
Constant_357_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232363328);
Constant_104_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232396096);
Reshape_368_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232396096);
Constant_365_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232396160);
Constant_111_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232428928);
Constant_112_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+236529024);
// create streams/handles
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
 // name=bert/pooler/dense/bias
Constant_float_cuda_Constant_110(0, Constant_110_0);
 // name=bert/pooler/dense/kernel
Constant_float_cuda_Constant_109(0, Constant_109_0);
 // name=bert/encoder/layer_0/attention/output/dense/bias
Constant_float_cuda_Constant_46(0, Constant_46_0);
 // name=bert/encoder/layer_0/attention/self/value/bias
Constant_float_cuda_Constant_32(0, Constant_32_0);
 // name=bert/encoder/layer_0/attention/self/value/kernel
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=bert/embeddings/token_type_embeddings
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=bert/embeddings/word_embeddings
Constant_float_cuda_Constant_4(0, Constant_4_0);
 // name=bert/embeddings/position_embeddings
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=bert/embeddings/LayerNorm/gamma
Constant_float_cuda_Constant_19(0, Constant_19_0);
 // name=bert/embeddings/LayerNorm/beta
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=Constant_150
Constant_float_cuda_Constant_150(0, Constant_150_0);
 // name=bert/embeddings/LayerNorm/batchnorm/add/y
Constant_float_cuda_Constant_22(0, Constant_22_0);
 // name=Constant_158
Constant_float_cuda_Constant_158(0, Constant_158_0);
 // name=bert/encoder/layer_0/attention/self/Mul/y
Constant_float_cuda_Constant_37(0, Constant_37_0);
 // name=bert/encoder/layer_0/attention/self/key/bias
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=bert/encoder/layer_0/attention/self/key/kernel
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=bert/encoder/layer_0/attention/self/query/bias
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=bert/encoder/layer_0/attention/self/query/kernel
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=bert/encoder/layer_1/attention/self/mul_1/y
Constant_float_cuda_Constant_79(0, Constant_79_0);
 // name=bert/encoder/layer_0/attention/self/mul_1/y
Constant_float_cuda_Constant_40(0, Constant_40_0);
 // name=bert/encoder/layer_1/attention/self/sub/x
Constant_float_cuda_Constant_78(0, Constant_78_0);
 // name=bert/encoder/layer_0/attention/self/sub/x
Constant_float_cuda_Constant_39(0, Constant_39_0);
 // name=bert/encoder/ones/Const
Constant_float_cuda_Constant_25(0, Constant_25_0);
 // name=bert/encoder/layer_0/attention/output/dense/kernel
Constant_float_cuda_Constant_45(0, Constant_45_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/gamma
Constant_float_cuda_Constant_48(0, Constant_48_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/beta
Constant_float_cuda_Constant_47(0, Constant_47_0);
 // name=Constant_207
Constant_float_cuda_Constant_207(0, Constant_207_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y
Constant_float_cuda_Constant_51(0, Constant_51_0);
 // name=Constant_215
Constant_float_cuda_Constant_215(0, Constant_215_0);
 // name=bert/encoder/layer_0/output/dense/bias
Constant_float_cuda_Constant_60(0, Constant_60_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul_2/x
Constant_float_cuda_Constant_58(0, Constant_58_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul_1/x
Constant_float_cuda_Constant_56(0, Constant_56_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul/x
Constant_float_cuda_Constant_55(0, Constant_55_0);
 // name=bert/encoder/layer_0/intermediate/dense/bias
Constant_float_cuda_Constant_53(0, Constant_53_0);
 // name=bert/encoder/layer_0/intermediate/dense/add_1/x
Constant_float_cuda_Constant_57(0, Constant_57_0);
 // name=bert/encoder/layer_0/intermediate/dense/Pow/y
Constant_float_cuda_Constant_54(0, Constant_54_0);
 // name=bert/encoder/layer_0/intermediate/dense/kernel
Constant_float_cuda_Constant_52(0, Constant_52_0);
 // name=bert/encoder/layer_0/output/dense/kernel
Constant_float_cuda_Constant_59(0, Constant_59_0);
 // name=bert/encoder/layer_0/output/LayerNorm/gamma
Constant_float_cuda_Constant_62(0, Constant_62_0);
 // name=bert/encoder/layer_0/output/LayerNorm/beta
Constant_float_cuda_Constant_61(0, Constant_61_0);
 // name=Constant_254
Constant_float_cuda_Constant_254(0, Constant_254_0);
 // name=bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y
Constant_float_cuda_Constant_65(0, Constant_65_0);
 // name=Constant_262
Constant_float_cuda_Constant_262(0, Constant_262_0);
 // name=bert/encoder/layer_1/attention/output/dense/bias
Constant_float_cuda_Constant_85(0, Constant_85_0);
 // name=bert/encoder/layer_1/attention/self/value/bias
Constant_float_cuda_Constant_71(0, Constant_71_0);
 // name=bert/encoder/layer_1/attention/self/value/kernel
Constant_float_cuda_Constant_70(0, Constant_70_0);
 // name=bert/encoder/layer_1/attention/self/Mul/y
Constant_float_cuda_Constant_76(0, Constant_76_0);
 // name=bert/encoder/layer_1/attention/self/key/bias
Constant_float_cuda_Constant_69(0, Constant_69_0);
 // name=bert/encoder/layer_1/attention/self/key/kernel
Constant_float_cuda_Constant_68(0, Constant_68_0);
 // name=bert/encoder/layer_1/attention/self/query/bias
Constant_float_cuda_Constant_67(0, Constant_67_0);
 // name=bert/encoder/layer_1/attention/self/query/kernel
Constant_float_cuda_Constant_66(0, Constant_66_0);
 // name=bert/encoder/layer_1/attention/output/dense/kernel
Constant_float_cuda_Constant_84(0, Constant_84_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/gamma
Constant_float_cuda_Constant_87(0, Constant_87_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/beta
Constant_float_cuda_Constant_86(0, Constant_86_0);
 // name=Constant_310
Constant_float_cuda_Constant_310(0, Constant_310_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y
Constant_float_cuda_Constant_90(0, Constant_90_0);
 // name=Constant_318
Constant_float_cuda_Constant_318(0, Constant_318_0);
 // name=bert/encoder/layer_1/output/dense/bias
Constant_float_cuda_Constant_99(0, Constant_99_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul_2/x
Constant_float_cuda_Constant_97(0, Constant_97_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul_1/x
Constant_float_cuda_Constant_95(0, Constant_95_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul/x
Constant_float_cuda_Constant_94(0, Constant_94_0);
 // name=bert/encoder/layer_1/intermediate/dense/bias
Constant_float_cuda_Constant_92(0, Constant_92_0);
 // name=bert/encoder/layer_1/intermediate/dense/add_1/x
Constant_float_cuda_Constant_96(0, Constant_96_0);
 // name=bert/encoder/layer_1/intermediate/dense/Pow/y
Constant_float_cuda_Constant_93(0, Constant_93_0);
 // name=bert/encoder/layer_1/intermediate/dense/kernel
Constant_float_cuda_Constant_91(0, Constant_91_0);
 // name=bert/encoder/layer_1/output/dense/kernel
Constant_float_cuda_Constant_98(0, Constant_98_0);
 // name=bert/encoder/layer_1/output/LayerNorm/gamma
Constant_float_cuda_Constant_101(0, Constant_101_0);
 // name=bert/encoder/layer_1/output/LayerNorm/beta
Constant_float_cuda_Constant_100(0, Constant_100_0);
 // name=Constant_357
Constant_float_cuda_Constant_357(0, Constant_357_0);
 // name=bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y
Constant_float_cuda_Constant_104(0, Constant_104_0);
 // name=Constant_365
Constant_float_cuda_Constant_365(0, Constant_365_0);
 // name=dense/kernel
Constant_float_cuda_Constant_111(0, Constant_111_0);
 // name=dense/bias
Constant_float_cuda_Constant_112(0, Constant_112_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
}


extern "C" int kernel_entry(int32_t* Parameter_0_0, int32_t* Parameter_1_0, int32_t* Parameter_2_0, float** Result_394_0)
{
// kernel_entry_init
 // name=bert/embeddings/Reshape_2
Reshape_int32_t_int32_t_cuda_lib_Reshape_114(0, Parameter_2_0, Reshape_114_0);
 // name=bert/embeddings/one_hot
OneHot_int32_t_float_cuda_OneHot_118_Call(dim3(128, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_114_0, OneHot_118_0);
 // name=bert/embeddings/MatMul
Dot_float_float_float_cuda_lib_Dot_122(cublas_handle_0, OneHot_118_0, Constant_8_0, Dot_122_0);
 // name=bert/embeddings/ExpandDims
Reshape_int32_t_int32_t_cuda_lib_Reshape_114(0, Parameter_0_0, Reshape_113_0);
 // name=bert/embeddings/Reshape
// eliminated
 // name=bert/embeddings/GatherV2
GatherV2_float_int32_t_float_cuda_GatherV2_121_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Constant_4_0, Reshape_117_0, GatherV2_121_0);
 // name=bert/embeddings/Slice
// eliminated
 // name=bert/embeddings/Reshape_4
// eliminated
 // name=Reshape_140
// eliminated
 // name=fused_kernel_395
FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, GatherV2_121_0, Dot_122_0, Reshape_140_0, Add_142_0);
 // name=Sum_149
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Add_142_0, Sum_149_0);
 // name=Divide_151
Divide_float_float_float_cuda_Divide_151_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_149_0, Constant_150_0, Divide_151_0);
 // name=bert/embeddings/LayerNorm/moments/mean
// eliminated
 // name=Reshape_153
// eliminated
 // name=fused_kernel_397
FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_153_0, Add_142_0, Multiply_156_0);
 // name=Sum_157
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Multiply_156_0, Sum_157_0);
 // name=Reshape_161
// eliminated
 // name=fused_kernel_398
FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_157_0, Constant_158_0, Reshape_161_0, Rsqrt_164_0);
 // name=Reshape_165
// eliminated
 // name=fused_kernel_399
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_18_0, Reshape_153_0, Reshape_165_0, Constant_19_0, Add_142_0, Add_175_0);
 // name=bert/encoder/Reshape_1
// eliminated
 // name=bert/encoder/layer_0/attention/self/value/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Reshape_176_0, Constant_31_0, Dot_179_0);
 // name=fused_kernel_402
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_32_0, Dot_179_0, Add_185_0);
 // name=bert/encoder/layer_0/attention/self/Reshape_2
// eliminated
 // name=bert/encoder/layer_0/attention/self/transpose_2
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_188_0, Reshape_191_0);
 // name=bert/encoder/layer_0/attention/self/key/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Reshape_176_0, Constant_29_0, Dot_178_0);
 // name=fused_kernel_401
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_30_0, Dot_178_0, Add_183_0);
 // name=bert/encoder/layer_0/attention/self/Reshape_1
// eliminated
 // name=bert/encoder/layer_0/attention/self/transpose_1
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_187_0, Reshape_190_0);
 // name=bert/encoder/layer_0/attention/self/query/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Reshape_176_0, Constant_27_0, Dot_177_0);
 // name=fused_kernel_400
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_28_0, Dot_177_0, Add_181_0);
 // name=bert/encoder/layer_0/attention/self/Reshape
// eliminated
 // name=bert/encoder/layer_0/attention/self/transpose
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_186_0, Reshape_189_0);
 // name=bert/encoder/layer_0/attention/self/MatMul
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublas_handle_0, Reshape_189_0, Reshape_190_0, BatchMatMul_192_0);
 // name=Reshape_146
// eliminated
 // name=Reshape_143
// eliminated
 // name=Reshape_137
// eliminated
 // name=Reshape_134
// eliminated
 // name=bert/encoder/ones
Broadcast_float_float_cuda_Broadcast_116_Call(dim3(128, 1, 1), dim3(64, 1, 1), 0, 0, Constant_25_0, Broadcast_116_0);
 // name=Reshape_124
// eliminated
 // name=bert/encoder/Reshape
Reshape_int32_t_int32_t_cuda_lib_Reshape_114(0, Parameter_1_0, Reshape_115_0);
 // name=bert/encoder/Cast
Convert_int32_t_float_cuda_Convert_120_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_115_0, Convert_120_0);
 // name=Reshape_126
// eliminated
 // name=Broadcast_127
Broadcast_float_float_cuda_Broadcast_127_Call(dim3(16384, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_126_0, Broadcast_127_0);
 // name=fused_kernel_396
FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_134_0, Reshape_124_0, Broadcast_127_0, Reshape_143_0, Reshape_137_0, Reshape_146_0, Multiply_148_0, Multiply_145_0);
 // name=Reshape_195
// eliminated
 // name=Broadcast_196
Broadcast_float_float_cuda_Broadcast_196_Call(dim3(262144, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_195_0, Broadcast_196_0);
 // name=fused_kernel_403
FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(dim3(32768, 1, 1), dim3(512, 1, 1), 0, 0, Constant_37_0, BatchMatMul_192_0, Broadcast_196_0, Add_197_0);
 // name=bert/encoder/layer_0/attention/self/Softmax
Softmax_float_float_cuda_lib_Softmax_198(cudnn_handle_0, Add_197_0, Softmax_198_0);
 // name=bert/encoder/layer_0/attention/self/MatMul_1
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublas_handle_0, Softmax_198_0, Reshape_191_0, BatchMatMul_199_0);
 // name=bert/encoder/layer_0/attention/self/transpose_3
Reshape_float_float_cuda_Reshape_200_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, BatchMatMul_199_0, Reshape_200_0);
 // name=bert/encoder/layer_0/attention/self/Reshape_3
// eliminated
 // name=bert/encoder/layer_0/attention/output/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Reshape_201_0, Constant_45_0, Dot_202_0);
 // name=fused_kernel_404
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_46_0, Dot_202_0, Reshape_176_0, Add_205_0);
 // name=Sum_206
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Add_205_0, Sum_206_0);
 // name=Divide_208
Divide_float_float_float_cuda_Divide_151_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_206_0, Constant_207_0, Divide_208_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/moments/mean
// eliminated
 // name=Reshape_210
// eliminated
 // name=fused_kernel_405
FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_210_0, Add_205_0, Multiply_213_0);
 // name=Sum_214
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Multiply_213_0, Sum_214_0);
 // name=Reshape_218
// eliminated
 // name=fused_kernel_406
FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_214_0, Constant_215_0, Reshape_218_0, Rsqrt_221_0);
 // name=Reshape_222
// eliminated
 // name=fused_kernel_407
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_47_0, Reshape_210_0, Reshape_222_0, Constant_48_0, Add_205_0, Add_232_0);
 // name=bert/encoder/layer_0/intermediate/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_233(cublas_handle_0, Add_232_0, Constant_52_0, Dot_233_0);
 // name=fused_kernel_408
FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0, Constant_58_0, Constant_57_0, Constant_56_0, Constant_55_0, Constant_53_0, Dot_233_0, Constant_54_0, Multiply_248_0);
 // name=bert/encoder/layer_0/output/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_249(cublas_handle_0, Multiply_248_0, Constant_59_0, Dot_249_0);
 // name=fused_kernel_409
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_60_0, Dot_249_0, Add_232_0, Add_252_0);
 // name=Sum_253
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Add_252_0, Sum_253_0);
 // name=Divide_255
Divide_float_float_float_cuda_Divide_151_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_253_0, Constant_254_0, Divide_255_0);
 // name=bert/encoder/layer_0/output/LayerNorm/moments/mean
// eliminated
 // name=Reshape_257
// eliminated
 // name=fused_kernel_410
FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_257_0, Add_252_0, Multiply_260_0);
 // name=Sum_261
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Multiply_260_0, Sum_261_0);
 // name=Reshape_265
// eliminated
 // name=fused_kernel_411
FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_261_0, Constant_262_0, Reshape_265_0, Rsqrt_268_0);
 // name=Reshape_269
// eliminated
 // name=fused_kernel_412
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_61_0, Reshape_257_0, Reshape_269_0, Constant_62_0, Add_252_0, Add_279_0);
 // name=bert/encoder/layer_1/attention/self/value/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Add_279_0, Constant_70_0, Dot_282_0);
 // name=fused_kernel_415
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_71_0, Dot_282_0, Add_288_0);
 // name=bert/encoder/layer_1/attention/self/Reshape_2
// eliminated
 // name=bert/encoder/layer_1/attention/self/transpose_2
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_291_0, Reshape_294_0);
 // name=bert/encoder/layer_1/attention/self/key/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Add_279_0, Constant_68_0, Dot_281_0);
 // name=fused_kernel_414
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_69_0, Dot_281_0, Add_286_0);
 // name=bert/encoder/layer_1/attention/self/Reshape_1
// eliminated
 // name=bert/encoder/layer_1/attention/self/transpose_1
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_290_0, Reshape_293_0);
 // name=bert/encoder/layer_1/attention/self/query/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Add_279_0, Constant_66_0, Dot_280_0);
 // name=fused_kernel_413
FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_67_0, Dot_280_0, Add_284_0);
 // name=bert/encoder/layer_1/attention/self/Reshape
// eliminated
 // name=bert/encoder/layer_1/attention/self/transpose
Reshape_float_float_cuda_Reshape_191_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_289_0, Reshape_292_0);
 // name=bert/encoder/layer_1/attention/self/MatMul
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublas_handle_0, Reshape_292_0, Reshape_293_0, BatchMatMul_295_0);
 // name=Reshape_298
// eliminated
 // name=Broadcast_299
Broadcast_float_float_cuda_Broadcast_196_Call(dim3(262144, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_298_0, Broadcast_299_0);
 // name=fused_kernel_416
FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(dim3(32768, 1, 1), dim3(512, 1, 1), 0, 0, Constant_76_0, BatchMatMul_295_0, Broadcast_299_0, Add_300_0);
 // name=bert/encoder/layer_1/attention/self/Softmax
Softmax_float_float_cuda_lib_Softmax_198(cudnn_handle_0, Add_300_0, Softmax_301_0);
 // name=bert/encoder/layer_1/attention/self/MatMul_1
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublas_handle_0, Softmax_301_0, Reshape_294_0, BatchMatMul_302_0);
 // name=bert/encoder/layer_1/attention/self/transpose_3
Reshape_float_float_cuda_Reshape_200_Call(dim3(131072, 1, 1), dim3(64, 1, 1), 0, 0, BatchMatMul_302_0, Reshape_303_0);
 // name=bert/encoder/layer_1/attention/self/Reshape_3
// eliminated
 // name=bert/encoder/layer_1/attention/output/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_179(cublas_handle_0, Reshape_304_0, Constant_84_0, Dot_305_0);
 // name=fused_kernel_417
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_85_0, Dot_305_0, Add_279_0, Add_308_0);
 // name=Sum_309
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Add_308_0, Sum_309_0);
 // name=Divide_311
Divide_float_float_float_cuda_Divide_151_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_309_0, Constant_310_0, Divide_311_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/moments/mean
// eliminated
 // name=Reshape_313
// eliminated
 // name=fused_kernel_418
FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_313_0, Add_308_0, Multiply_316_0);
 // name=Sum_317
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Multiply_316_0, Sum_317_0);
 // name=Reshape_321
// eliminated
 // name=fused_kernel_419
FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_317_0, Constant_318_0, Reshape_321_0, Rsqrt_324_0);
 // name=Reshape_325
// eliminated
 // name=fused_kernel_420
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_86_0, Reshape_313_0, Reshape_325_0, Constant_87_0, Add_308_0, Add_335_0);
 // name=bert/encoder/layer_1/intermediate/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_233(cublas_handle_0, Add_335_0, Constant_91_0, Dot_336_0);
 // name=fused_kernel_421
FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0, Constant_97_0, Constant_96_0, Constant_95_0, Constant_94_0, Constant_92_0, Dot_336_0, Constant_93_0, Multiply_351_0);
 // name=bert/encoder/layer_1/output/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_249(cublas_handle_0, Multiply_351_0, Constant_98_0, Dot_352_0);
 // name=fused_kernel_422
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_99_0, Dot_352_0, Add_335_0, Add_355_0);
 // name=Sum_356
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Add_355_0, Sum_356_0);
 // name=Divide_358
Divide_float_float_float_cuda_Divide_151_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_356_0, Constant_357_0, Divide_358_0);
 // name=bert/encoder/layer_1/output/LayerNorm/moments/mean
// eliminated
 // name=Reshape_360
// eliminated
 // name=fused_kernel_423
FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_360_0, Add_355_0, Multiply_363_0);
 // name=Sum_364
Sum_float_float_cuda_Sum_149_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Multiply_363_0, Sum_364_0);
 // name=Reshape_368
// eliminated
 // name=fused_kernel_424
FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(dim3(16, 1, 1), dim3(512, 1, 1), 0, 0, Sum_364_0, Constant_365_0, Reshape_368_0, Rsqrt_371_0);
 // name=Reshape_372
// eliminated
 // name=fused_kernel_425
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(dim3(16384, 1, 1), dim3(512, 1, 1), 0, 0, Constant_100_0, Reshape_360_0, Reshape_372_0, Constant_101_0, Add_355_0, Add_382_0);
 // name=bert/encoder/Reshape_3
// eliminated
 // name=bert/pooler/strided_slice
Slice_float_float_cuda_Slice_384_Call(dim3(1024, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_383_0, Slice_384_0);
 // name=bert/pooler/Squeeze
// eliminated
 // name=bert/pooler/dense/MatMul
Dot_float_float_float_cuda_lib_Dot_386(cublas_handle_0, Reshape_385_0, Constant_109_0, Dot_386_0);
 // name=fused_kernel_426
FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call(dim3(128, 1, 1), dim3(512, 1, 1), 0, 0, Constant_110_0, Dot_386_0, Tanh_389_0);
 // name=dense/MatMul
Dot_float_float_float_cuda_lib_Dot_390(cublas_handle_0, Tanh_389_0, Constant_111_0, Dot_390_0);
 // name=Broadcast_391
Broadcast_float_float_cuda_Broadcast_391_Call(dim3(1001, 1, 1), dim3(64, 1, 1), 0, 0, Constant_112_0, Broadcast_391_0);
 // name=dense/BiasAdd
Add_float_float_float_cuda_Add_392_Call(dim3(1001, 1, 1), dim3(64, 1, 1), 0, 0, Dot_390_0, Broadcast_391_0, Add_392_0);
 // name=dense/Softmax
Softmax_float_float_cuda_lib_Softmax_393(cudnn_handle_0, Add_392_0, Softmax_393_0);
 // name=Result_394
Result_float_float_cuda_lib_Result_394(Softmax_393_0, Result_394_0);
return 0;
}


extern "C" void cuda_free()
{
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
}

