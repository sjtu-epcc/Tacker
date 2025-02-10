#pragma once
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
char* bert_group_0_CUDA_GPU0_allocator_memory_pool;
int32_t* bert_Reshape_114_0;
float* bert_OneHot_118_0;
float* bert_Dot_122_0;
int32_t* bert_Reshape_113_0;
int32_t* bert_Reshape_117_0;
float* bert_GatherV2_121_0;
float* bert_Add_142_0;
float* bert_Sum_149_0;
float* bert_Divide_151_0;
float* bert_Reshape_152_0;
float* bert_Reshape_153_0;
float* bert_Multiply_156_0;
float* bert_Sum_157_0;
float* bert_Rsqrt_164_0;
float* bert_Reshape_165_0;
float* bert_Add_175_0;
float* bert_Reshape_176_0;
float* bert_Dot_179_0;
float* bert_Add_185_0;
float* bert_Reshape_188_0;
float* bert_Reshape_191_0;
float* bert_Dot_178_0;
float* bert_Add_183_0;
float* bert_Reshape_187_0;
float* bert_Reshape_190_0;
float* bert_Dot_177_0;
float* bert_Add_181_0;
float* bert_Reshape_186_0;
float* bert_Reshape_189_0;
float* bert_BatchMatMul_192_0;
float* bert_Broadcast_116_0;
float* bert_Reshape_124_0;
int32_t* bert_Reshape_115_0;
float* bert_Convert_120_0;
float* bert_Reshape_126_0;
float* bert_Broadcast_127_0;
float* bert_Multiply_145_0;
float* bert_Multiply_148_0;
float* bert_Reshape_195_0;
float* bert_Broadcast_196_0;
float* bert_Add_197_0;
float* bert_Softmax_198_0;
float* bert_BatchMatMul_199_0;
float* bert_Reshape_200_0;
float* bert_Reshape_201_0;
float* bert_Dot_202_0;
float* bert_Add_205_0;
float* bert_Sum_206_0;
float* bert_Divide_208_0;
float* bert_Reshape_209_0;
float* bert_Reshape_210_0;
float* bert_Multiply_213_0;
float* bert_Sum_214_0;
float* bert_Rsqrt_221_0;
float* bert_Reshape_222_0;
float* bert_Add_232_0;
float* bert_Dot_233_0;
float* bert_Multiply_248_0;
float* bert_Dot_249_0;
float* bert_Add_252_0;
float* bert_Sum_253_0;
float* bert_Divide_255_0;
float* bert_Reshape_256_0;
float* bert_Reshape_257_0;
float* bert_Multiply_260_0;
float* bert_Sum_261_0;
float* bert_Rsqrt_268_0;
float* bert_Reshape_269_0;
float* bert_Add_279_0;
float* bert_Dot_282_0;
float* bert_Add_288_0;
float* bert_Reshape_291_0;
float* bert_Reshape_294_0;
float* bert_Dot_281_0;
float* bert_Add_286_0;
float* bert_Reshape_290_0;
float* bert_Reshape_293_0;
float* bert_Dot_280_0;
float* bert_Add_284_0;
float* bert_Reshape_289_0;
float* bert_Reshape_292_0;
float* bert_BatchMatMul_295_0;
float* bert_Reshape_298_0;
float* bert_Broadcast_299_0;
float* bert_Add_300_0;
float* bert_Softmax_301_0;
float* bert_BatchMatMul_302_0;
float* bert_Reshape_303_0;
float* bert_Reshape_304_0;
float* bert_Dot_305_0;
float* bert_Add_308_0;
float* bert_Sum_309_0;
float* bert_Divide_311_0;
float* bert_Reshape_312_0;
float* bert_Reshape_313_0;
float* bert_Multiply_316_0;
float* bert_Sum_317_0;
float* bert_Rsqrt_324_0;
float* bert_Reshape_325_0;
float* bert_Add_335_0;
float* bert_Dot_336_0;
float* bert_Multiply_351_0;
float* bert_Dot_352_0;
float* bert_Add_355_0;
float* bert_Sum_356_0;
float* bert_Divide_358_0;
float* bert_Reshape_359_0;
float* bert_Reshape_360_0;
float* bert_Multiply_363_0;
float* bert_Sum_364_0;
float* bert_Rsqrt_371_0;
float* bert_Reshape_372_0;
float* bert_Add_382_0;
float* bert_Reshape_383_0;
float* bert_Slice_384_0;
float* bert_Reshape_385_0;
float* bert_Dot_386_0;
float* bert_Tanh_389_0;
float* bert_Dot_390_0;
float* bert_Broadcast_391_0;
float* bert_Add_392_0;
float* bert_Softmax_393_0;
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
cudnnHandle_t bert_cudnn_handle_0;
cublasHandle_t bert_cublas_handle_0;
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
int bert_num_SMs;
char* bert_group_persist_CUDA_GPU0_allocator_memory_pool;
float* bert_Constant_110_0;
float* bert_Constant_109_0;
float* bert_Constant_46_0;
float* bert_Constant_32_0;
float* bert_Constant_31_0;
float* bert_Constant_8_0;
float* bert_Constant_4_0;
float* bert_Constant_14_0;
float* bert_Slice_119_0;
float* bert_Reshape_123_0;
float* bert_Reshape_140_0;
float* bert_Constant_19_0;
float* bert_Constant_18_0;
float* bert_Constant_150_0;
float* bert_Constant_22_0;
float* bert_Reshape_161_0;
float* bert_Constant_158_0;
float* bert_Constant_37_0;
float* bert_Constant_30_0;
float* bert_Constant_29_0;
float* bert_Constant_28_0;
float* bert_Constant_27_0;
float* bert_Constant_79_0;
float* bert_Reshape_146_0;
float* bert_Constant_40_0;
float* bert_Reshape_143_0;
float* bert_Constant_78_0;
float* bert_Reshape_137_0;
float* bert_Constant_39_0;
float* bert_Reshape_134_0;
float* bert_Constant_25_0;
float* bert_Constant_45_0;
float* bert_Constant_48_0;
float* bert_Constant_47_0;
float* bert_Constant_207_0;
float* bert_Constant_51_0;
float* bert_Reshape_218_0;
float* bert_Constant_215_0;
float* bert_Constant_60_0;
float* bert_Constant_58_0;
float* bert_Constant_56_0;
float* bert_Constant_55_0;
float* bert_Constant_53_0;
float* bert_Constant_57_0;
float* bert_Constant_54_0;
float* bert_Constant_52_0;
float* bert_Constant_59_0;
float* bert_Constant_62_0;
float* bert_Constant_61_0;
float* bert_Constant_254_0;
float* bert_Constant_65_0;
float* bert_Reshape_265_0;
float* bert_Constant_262_0;
float* bert_Constant_85_0;
float* bert_Constant_71_0;
float* bert_Constant_70_0;
float* bert_Constant_76_0;
float* bert_Constant_69_0;
float* bert_Constant_68_0;
float* bert_Constant_67_0;
float* bert_Constant_66_0;
float* bert_Constant_84_0;
float* bert_Constant_87_0;
float* bert_Constant_86_0;
float* bert_Constant_310_0;
float* bert_Constant_90_0;
float* bert_Reshape_321_0;
float* bert_Constant_318_0;
float* bert_Constant_99_0;
float* bert_Constant_97_0;
float* bert_Constant_95_0;
float* bert_Constant_94_0;
float* bert_Constant_92_0;
float* bert_Constant_96_0;
float* bert_Constant_93_0;
float* bert_Constant_91_0;
float* bert_Constant_98_0;
float* bert_Constant_101_0;
float* bert_Constant_100_0;
float* bert_Constant_357_0;
float* bert_Constant_104_0;
float* bert_Reshape_368_0;
float* bert_Constant_365_0;
float* bert_Constant_111_0;
float* bert_Constant_112_0;

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 3
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {256, 128}
#define NNFUSION_GRAPH_INPUT_DTYPE_1 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_1 {256, 128}
#define NNFUSION_GRAPH_INPUT_DTYPE_2 int32_t
#define NNFUSION_GRAPH_INPUT_SHAPE_2 {256, 128}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {256, 1001}
#endif

// Node name:	Dot_390
// Description:	Dot
// Input:
//	- name: bert_Tanh_389_0	type: float	shape: Shape{256, 1024}
//	- name: bert_Constant_111_0	type: float	shape: Shape{1024, 1001}
// Output:
//	- name: bert_Dot_390_0	type: float	shape: Shape{256, 1001}
void Dot_float_float_float_cuda_lib_Dot_390(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 256, 1024, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_27_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_32_0	type: float	shape: Shape{1024}
//	- name: bert_Dot_179_0	type: float	shape: Shape{32768, 1024}
// Output:
//	- name: bert_Add_185_0	type: float	shape: Shape{32768, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_184<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_32_0, Broadcast_184_0);
// Add_float_float_float_cuda_Add_185<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Dot_179_0, Broadcast_184_0, bert_Add_185_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Sum_149
// Description:	Sum
// Input:
//	- name: bert_Add_142_0	type: float	shape: Shape{256, 128, 1024}
// Output:
//	- name: bert_Sum_149_0	type: float	shape: Shape{256, 128}
extern "C" __launch_bounds__(512) __global__ void bert_Sum_float_float_cuda_Sum_149(float* input0, float* output0)
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
extern void bert_Sum_float_float_cuda_Sum_149_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Sum_float_float_cuda_Sum_149<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_127
// Description:	Broadcast
// Input:
//	- name: bert_Reshape_126_0	type: float	shape: Shape{256, 128}
// Output:
//	- name: bert_Broadcast_127_0	type: float	shape: Shape{256, 128, 128}
extern "C" __launch_bounds__(64) __global__ void bert_Broadcast_float_float_cuda_Broadcast_127(float* input0, float* output0)
{
    size_t nthreads = 4194304;uint32_t strides0 = 16384;
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
extern void bert_Broadcast_float_float_cuda_Broadcast_127_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_127<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_196
// Description:	Broadcast
// Input:
//	- name: bert_Reshape_195_0	type: float	shape: Shape{256, 128, 128}
// Output:
//	- name: bert_Broadcast_196_0	type: float	shape: Shape{256, 16, 128, 128}
extern "C" __launch_bounds__(64) __global__ void bert_Broadcast_float_float_cuda_Broadcast_196(float* input0, float* output0)
{
    size_t nthreads = 67108864;uint32_t strides0 = 262144;
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
extern void bert_Broadcast_float_float_cuda_Broadcast_196_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_196<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_116
// Description:	Broadcast
// Input:
//	- name: bert_Constant_25_0	type: float	shape: Shape{}
// Output:
//	- name: bert_Broadcast_116_0	type: float	shape: Shape{256, 128, 1}
extern "C" __launch_bounds__(64) __global__ void bert_Broadcast_float_float_cuda_Broadcast_116(float* input0, float* output0)
{
    size_t nthreads = 32768;uint32_t strides0 = 128;
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
extern void bert_Broadcast_float_float_cuda_Broadcast_116_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_116<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_59_0	type: float	shape: Shape{4096, 1024}
void bert_Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_59_0 failed.\n");
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
//	- name: bert_Constant_101_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_101_0 failed.\n");
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
//	- name: bert_Constant_61_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_61_0 failed.\n");
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
//	- name: bert_Constant_65_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_65_0 failed.\n");
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
//	- name: bert_Constant_93_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_93_0 failed.\n");
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
//	- name: bert_Constant_85_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_85_0 failed.\n");
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
//	- name: bert_Constant_37_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_37_0 failed.\n");
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
//	- name: bert_Constant_84_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_84_0 failed.\n");
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
//	- name: bert_Constant_87_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_87_0 failed.\n");
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
//	- name: bert_Constant_86_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_86_0 failed.\n");
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
//	- name: bert_Constant_67_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_67_0 failed.\n");
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
//	- name: bert_Constant_91_0	type: float	shape: Shape{1024, 4096}
void bert_Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_91_0 failed.\n");
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
//	- name: bert_Constant_100_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_100_0 failed.\n");
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
//	- name: bert_Constant_98_0	type: float	shape: Shape{4096, 1024}
void bert_Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_98_0 failed.\n");
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
//	- name: bert_Constant_112_0	type: float	shape: Shape{1001}
// Output:
//	- name: bert_Broadcast_391_0	type: float	shape: Shape{256, 1001}
extern "C" __launch_bounds__(64) __global__ void bert_Broadcast_float_float_cuda_Broadcast_391(float* input0, float* output0)
{
    size_t nthreads = 256256;uint32_t strides0 = 1001;
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
extern void bert_Broadcast_float_float_cuda_Broadcast_391_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_391<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_357
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_357_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_357(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_357_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_357_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_158
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_158_0	type: float	shape: Shape{256, 128}
void bert_Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_92_0	type: float	shape: Shape{4096}
void bert_Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_92_0 failed.\n");
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
//	- name: bert_Constant_318_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_318(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_318_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_318_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_68_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_68_0 failed.\n");
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
//	- name: bert_Constant_104_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_104_0 failed.\n");
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
//	- name: bert_Constant_95_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_95_0 failed.\n");
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
//	- name: bert_Constant_19_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_19_0 failed.\n");
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
//	- name: bert_Constant_96_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_96_0 failed.\n");
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
//	- name: bert_Constant_97_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_97_0 failed.\n");
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
//	- name: bert_Constant_112_0	type: float	shape: Shape{1001}
void bert_Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_112_0 failed.\n");
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
//	- name: bert_Constant_90_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_90_0 failed.\n");
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
//	- name: bert_Constant_47_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_47_0 failed.\n");
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
//	- name: bert_Constant_57_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_57_0 failed.\n");
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
//	- name: bert_Constant_14_0	type: float	shape: Shape{512, 1024}
void bert_Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_14_0 failed.\n");
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
//	- name: bert_Constant_52_0	type: float	shape: Shape{1024, 4096}
void bert_Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_52_0 failed.\n");
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
//	- name: bert_Constant_110_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_110_0 failed.\n");
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
//	- name: bert_Constant_76_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_76_0 failed.\n");
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
//	- name: bert_Constant_4_0	type: float	shape: Shape{30522, 1024}
void bert_Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_4_0 failed.\n");
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
//	- name: bert_Constant_22_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_22_0 failed.\n");
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
//	- name: bert_Constant_66_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_66_0 failed.\n");
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
//	- name: bert_Constant_150_0	type: float	shape: Shape{256, 128}
void bert_Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_46_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_46_0 failed.\n");
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
//	- name: bert_Constant_69_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_69_0 failed.\n");
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
//	- name: bert_Constant_215_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_215(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_215_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_215_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_111
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_111_0	type: float	shape: Shape{1024, 1001}
void bert_Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_111_0 failed.\n");
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
//	- name: bert_Constant_70_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_70_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_18_0	type: float	shape: Shape{1024}
//	- name: bert_Reshape_153_0	type: float	shape: Shape{256, 128}
//	- name: bert_Reshape_165_0	type: float	shape: Shape{256, 128}
//	- name: bert_Constant_19_0	type: float	shape: Shape{1024}
//	- name: bert_Add_142_0	type: float	shape: Shape{256, 128, 1024}
// Output:
//	- name: bert_Add_175_0	type: float	shape: Shape{256, 128, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_173<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_18_0, Broadcast_173_0);
// Broadcast_float_float_cuda_Broadcast_154<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_153_0, Broadcast_154_0);
// Broadcast_float_float_cuda_Broadcast_166<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_165_0, Broadcast_166_0);
// Broadcast_float_float_cuda_Broadcast_167<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_19_0, Broadcast_167_0);
// Multiply_float_float_float_cuda_Multiply_168<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_166_0, Broadcast_167_0, Multiply_168_0);
// Multiply_float_float_float_cuda_Multiply_172<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_154_0, Multiply_168_0, Multiply_172_0);
// Subtract_float_float_float_cuda_Subtract_174<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_173_0, Multiply_172_0, Subtract_174_0);
// Multiply_float_float_float_cuda_Multiply_169<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Add_142_0, Multiply_168_0, Multiply_169_0);
// Add_float_float_float_cuda_Add_175<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_169_0, Subtract_174_0, bert_Add_175_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
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
extern void bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_58_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_58_0 failed.\n");
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
//	- name: bert_Constant_28_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_28_0 failed.\n");
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
//	- name: bert_Constant_79_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_79_0 failed.\n");
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
//	- name: bert_Constant_109_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_109_0 failed.\n");
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
//	- name: bert_Constant_45_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_45_0 failed.\n");
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
//	- name: bert_Constant_207_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_207(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_207_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_207_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_48_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_48_0 failed.\n");
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
//	- name: bert_Constant_99_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_99_0 failed.\n");
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
//	- name: bert_Constant_262_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_262(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_262_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_262_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_32_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_32_0 failed.\n");
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
//	- name: bert_Constant_30_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_30_0 failed.\n");
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
//	- name: bert_Constant_51_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_51_0 failed.\n");
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
//	- name: bert_Constant_60_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_60_0 failed.\n");
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
//	- name: bert_Constant_71_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_71_0 failed.\n");
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
//	- name: bert_Constant_254_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_254_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_78_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_78_0 failed.\n");
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
//	- name: bert_Constant_56_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_56_0 failed.\n");
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
//	- name: bert_Constant_94_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_37_0	type: float	shape: Shape{}
//	- name: bert_BatchMatMul_192_0	type: float	shape: Shape{256, 16, 128, 128}
//	- name: bert_Broadcast_196_0	type: float	shape: Shape{256, 16, 128, 128}
// Output:
//	- name: bert_Add_197_0	type: float	shape: Shape{256, 16, 128, 128}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_193<<<dim3(1048576, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_37_0, Broadcast_193_0);
// Multiply_float_float_float_cuda_Multiply_194<<<dim3(131072, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_BatchMatMul_192_0, Broadcast_193_0, Multiply_194_0);
// Add_float_float_float_cuda_Add_197<<<dim3(131072, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_194_0, bert_Broadcast_196_0, bert_Add_197_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = mul(input1[tid], temp0);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_55_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_55_0 failed.\n");
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
//	- name: bert_Constant_53_0	type: float	shape: Shape{4096}
void bert_Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_53_0 failed.\n");
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
//	- name: bert_Constant_310_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_310(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_310_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_310_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_40_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_40_0 failed.\n");
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
//	- name: bert_Reshape_114_0	type: int32_t	shape: Shape{32768}
// Output:
//	- name: bert_OneHot_118_0	type: float	shape: Shape{32768, 2}
extern "C" __launch_bounds__(64) __global__ void bert_OneHot_int32_t_float_cuda_OneHot_118(int32_t* input0, float* output0)
{
    {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= 32768)
                return;
            for (int i = 0; i < 2; ++i)
                output0[idx * 2 + i] = 0.000000000000000000000000e+00;
            output0[idx * 2 + (int)input0[idx]] = 1.000000000000000000000000e+00;

    }

}
extern void bert_OneHot_int32_t_float_cuda_OneHot_118_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    bert_OneHot_int32_t_float_cuda_OneHot_118<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Reshape_153_0	type: float	shape: Shape{256, 128}
//	- name: bert_Add_142_0	type: float	shape: Shape{256, 128, 1024}
// Output:
//	- name: bert_Multiply_156_0	type: float	shape: Shape{256, 128, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_154<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_153_0, Broadcast_154_0);
// Subtract_float_float_float_cuda_Subtract_155<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Add_142_0, Broadcast_154_0, Subtract_155_0);
// Multiply_float_float_float_cuda_Multiply_156<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_155_0, Subtract_155_0, bert_Multiply_156_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid / 1024];
    float temp1 = subtractf(input1[tid], temp0);
    float temp2 = mul(temp1, temp1);
    output0[tid] = temp2;

}
extern void bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Softmax_393
// Description:	Softmax
// Input:
//	- name: bert_Add_392_0	type: float	shape: Shape{256, 1001}
// Output:
//	- name: bert_Softmax_393_0	type: float	shape: Shape{256, 1001}
void Softmax_float_float_cuda_lib_Softmax_393(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 256, 1, 1, 1001));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 256, 1, 1, 1001));
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
//	- name: bert_Constant_25_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_25_0 failed.\n");
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
//	- name: bert_Reshape_189_0	type: float	shape: Shape{256, 16, 128, 64}
//	- name: bert_Reshape_190_0	type: float	shape: Shape{256, 16, 128, 64}
// Output:
//	- name: bert_BatchMatMul_192_0	type: float	shape: Shape{256, 16, 128, 128}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 128, 128, 64,
                                    &alpha, input1, 64, 8192, input0, 64, 8192,
                                    &beta, output0, 128, 16384, 4096));
                            
    }

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_GatherV2_121_0	type: float	shape: Shape{32768, 1024}
//	- name: bert_Dot_122_0	type: float	shape: Shape{32768, 1024}
//	- name: bert_Reshape_140_0	type: float	shape: Shape{128, 1024}
// Output:
//	- name: bert_Add_142_0	type: float	shape: Shape{256, 128, 1024}
// Fused functions:
// Reshape_float_float_cuda_lib_Reshape_129(bert_GatherV2_121_0, Reshape_129_0);
// Reshape_float_float_cuda_lib_Reshape_130(bert_Dot_122_0, Reshape_130_0);
// Add_float_float_float_cuda_Add_133<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_129_0, Reshape_130_0, Add_133_0);
// Broadcast_float_float_cuda_Broadcast_141<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_140_0, Broadcast_141_0);
// Add_float_float_float_cuda_Add_142<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_133_0, Broadcast_141_0, bert_Add_142_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = input2[tid % 131072];
    float temp2 = add(temp0, temp1);
    output0[tid] = temp2;

}
extern void bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Dot_179
// Description:	Dot
// Input:
//	- name: bert_Reshape_176_0	type: float	shape: Shape{32768, 1024}
//	- name: bert_Constant_31_0	type: float	shape: Shape{1024, 1024}
// Output:
//	- name: bert_Dot_179_0	type: float	shape: Shape{32768, 1024}
void Dot_float_float_float_cuda_lib_Dot_179(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_31_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_31_0 failed.\n");
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
//	- name: bert_Softmax_393_0	type: float	shape: Shape{256, 1001}
// Output:
//	- name: Result_394_0	type: float	shape: Shape{256, 1001}
void Result_float_float_cuda_lib_Result_394(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_18_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_18_0 failed.\n");
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
//	- name: bert_Reshape_188_0	type: float	shape: Shape{256, 128, 16, 64}
// Output:
//	- name: bert_Reshape_191_0	type: float	shape: Shape{256, 16, 128, 64}
extern "C" __launch_bounds__(64) __global__ void bert_Reshape_float_float_cuda_Reshape_191(float* input0, float* output0)
{
    uint32_t input_strides0 = 131072;
    uint32_t input_strides1 = 1024;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 131072;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 8192;
    uint32_t trans_strides3 = 1;
    size_t n = 33554432;
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
extern void bert_Reshape_float_float_cuda_Reshape_191_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Reshape_float_float_cuda_Reshape_191<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Divide_151
// Description:	Divide
// Input:
//	- name: bert_Sum_149_0	type: float	shape: Shape{256, 128}
//	- name: bert_Constant_150_0	type: float	shape: Shape{256, 128}
// Output:
//	- name: bert_Divide_151_0	type: float	shape: Shape{256, 128}
extern "C" __launch_bounds__(512) __global__ void bert_Divide_float_float_float_cuda_Divide_151(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = fdividef(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void bert_Divide_float_float_float_cuda_Divide_151_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_Divide_float_float_float_cuda_Divide_151<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	GatherV2_121
// Description:	GatherV2
// Input:
//	- name: bert_Constant_4_0	type: float	shape: Shape{30522, 1024}
//	- name: bert_Reshape_117_0	type: int32_t	shape: Shape{32768}
// Output:
//	- name: bert_GatherV2_121_0	type: float	shape: Shape{32768, 1024}
extern "C" __launch_bounds__(64) __global__ void bert_GatherV2_float_int32_t_float_cuda_GatherV2_121(float* input0, int32_t* input1, float* output0)
{
    float* params = input0;
    int32_t* indices = input1;
    float* out = output0;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 33554432)
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
extern void bert_GatherV2_float_int32_t_float_cuda_GatherV2_121_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int32_t* input1, float* output0) {
    bert_GatherV2_float_int32_t_float_cuda_GatherV2_121<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_58_0	type: float	shape: Shape{}
//	- name: bert_Constant_57_0	type: float	shape: Shape{}
//	- name: bert_Constant_56_0	type: float	shape: Shape{}
//	- name: bert_Constant_55_0	type: float	shape: Shape{}
//	- name: bert_Constant_53_0	type: float	shape: Shape{4096}
//	- name: bert_Dot_233_0	type: float	shape: Shape{32768, 4096}
//	- name: bert_Constant_54_0	type: float	shape: Shape{}
// Output:
//	- name: bert_Multiply_248_0	type: float	shape: Shape{32768, 4096}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_246<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_58_0, Broadcast_246_0);
// Broadcast_float_float_cuda_Broadcast_244<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_57_0, Broadcast_244_0);
// Broadcast_float_float_cuda_Broadcast_241<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_56_0, Broadcast_241_0);
// Broadcast_float_float_cuda_Broadcast_238<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_55_0, Broadcast_238_0);
// Broadcast_float_float_cuda_Broadcast_234<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_53_0, Broadcast_234_0);
// Add_float_float_float_cuda_Add_235<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Dot_233_0, Broadcast_234_0, Add_235_0);
// Broadcast_float_float_cuda_Broadcast_236<<<dim3(2097152, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_54_0, Broadcast_236_0);
// Power_float_float_float_cuda_Power_237<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Broadcast_236_0, Power_237_0);
// Multiply_float_float_float_cuda_Multiply_239<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_238_0, Power_237_0, Multiply_239_0);
// Add_float_float_float_cuda_Add_240<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Multiply_239_0, Add_240_0);
// Multiply_float_float_float_cuda_Multiply_242<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_241_0, Add_240_0, Multiply_242_0);
// Tanh_float_float_cuda_Tanh_243<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Multiply_242_0, Tanh_243_0);
// Add_float_float_float_cuda_Add_245<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_244_0, Tanh_243_0, Add_245_0);
// Multiply_float_float_float_cuda_Multiply_247<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_246_0, Add_245_0, Multiply_247_0);
// Multiply_float_float_float_cuda_Multiply_248<<<dim3(262144, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_235_0, Multiply_247_0, bert_Multiply_248_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0)
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
extern void bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0) {
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0);
}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_8_0	type: float	shape: Shape{2, 1024}
void bert_Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_46_0	type: float	shape: Shape{1024}
//	- name: bert_Dot_202_0	type: float	shape: Shape{32768, 1024}
//	- name: bert_Reshape_176_0	type: float	shape: Shape{32768, 1024}
// Output:
//	- name: bert_Add_205_0	type: float	shape: Shape{32768, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_203<<<dim3(524288, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_46_0, Broadcast_203_0);
// Add_float_float_float_cuda_Add_204<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Dot_202_0, Broadcast_203_0, Add_204_0);
// Add_float_float_float_cuda_Add_205<<<dim3(65536, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_204_0, bert_Reshape_176_0, bert_Add_205_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_39_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Constant_110_0	type: float	shape: Shape{1024}
//	- name: bert_Dot_386_0	type: float	shape: Shape{256, 1024}
// Output:
//	- name: bert_Tanh_389_0	type: float	shape: Shape{256, 1024}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_387<<<dim3(4096, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Constant_110_0, Broadcast_387_0);
// Add_float_float_float_cuda_Add_388<<<dim3(512, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Dot_386_0, Broadcast_387_0, Add_388_0);
// Tanh_float_float_cuda_Tanh_389<<<dim3(512, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_388_0, bert_Tanh_389_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1024];
    float temp1 = add(input1[tid], temp0);
    float temp2 = tanhf(temp1);
    output0[tid] = temp2;

}
extern void bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convert_120
// Description:	Convert
// Input:
//	- name: bert_Reshape_115_0	type: int32_t	shape: Shape{256, 1, 128}
// Output:
//	- name: bert_Convert_120_0	type: float	shape: Shape{256, 1, 128}
extern "C" __launch_bounds__(512) __global__ void bert_Convert_int32_t_float_cuda_Convert_120(int32_t* input0, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = convert(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern void bert_Convert_int32_t_float_cuda_Convert_120_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    bert_Convert_int32_t_float_cuda_Convert_120<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Reshape_134_0	type: float	shape: Shape{1}
//	- name: bert_Reshape_124_0	type: float	shape: Shape{256, 128}
//	- name: bert_Broadcast_127_0	type: float	shape: Shape{256, 128, 128}
//	- name: bert_Reshape_143_0	type: float	shape: Shape{1}
//	- name: bert_Reshape_137_0	type: float	shape: Shape{1}
//	- name: bert_Reshape_146_0	type: float	shape: Shape{1}
// Output:
//	- name: bert_Multiply_148_0	type: float	shape: Shape{256, 1, 128, 128}
//	- name: bert_Multiply_145_0	type: float	shape: Shape{256, 1, 128, 128}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_135<<<dim3(65536, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_134_0, Broadcast_135_0);
// Broadcast_float_float_cuda_Broadcast_125<<<dim3(65536, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_124_0, Broadcast_125_0);
// Multiply_float_float_float_cuda_Multiply_128<<<dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_125_0, bert_Broadcast_127_0, Multiply_128_0);
// Reshape_float_float_cuda_lib_Reshape_131(Multiply_128_0, Reshape_131_0);
// Subtract_float_float_float_cuda_Subtract_136<<<dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_135_0, Reshape_131_0, Subtract_136_0);
// Broadcast_float_float_cuda_Broadcast_144<<<dim3(65536, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_143_0, Broadcast_144_0);
// Multiply_float_float_float_cuda_Multiply_145<<<dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_136_0, Broadcast_144_0, bert_Multiply_145_0);
// Broadcast_float_float_cuda_Broadcast_138<<<dim3(65536, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_137_0, Broadcast_138_0);
// Subtract_float_float_float_cuda_Subtract_139<<<dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0>>>(Broadcast_138_0, Reshape_131_0, Subtract_139_0);
// Broadcast_float_float_cuda_Broadcast_147<<<dim3(65536, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_146_0, Broadcast_147_0);
// Multiply_float_float_float_cuda_Multiply_148<<<dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0>>>(Subtract_139_0, Broadcast_147_0, bert_Multiply_148_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
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
extern void bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_54_0	type: float	shape: Shape{}
void bert_Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_54_0 failed.\n");
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
//	- name: bert_Reshape_385_0	type: float	shape: Shape{256, 1024}
//	- name: bert_Constant_109_0	type: float	shape: Shape{1024, 1024}
// Output:
//	- name: bert_Dot_386_0	type: float	shape: Shape{256, 1024}
void Dot_float_float_float_cuda_lib_Dot_386(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 256, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_29_0	type: float	shape: Shape{1024, 1024}
void bert_Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_29_0 failed.\n");
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
//	- name: bert_Add_197_0	type: float	shape: Shape{256, 16, 128, 128}
// Output:
//	- name: bert_Softmax_198_0	type: float	shape: Shape{256, 16, 128, 128}
void Softmax_float_float_cuda_lib_Softmax_198(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 524288, 1, 1, 128));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 524288, 1, 1, 128));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor_desc, input0, &beta, output_tensor_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));

}
// Node name:	Dot_122
// Description:	Dot
// Input:
//	- name: bert_OneHot_118_0	type: float	shape: Shape{32768, 2}
//	- name: bert_Constant_8_0	type: float	shape: Shape{2, 1024}
// Output:
//	- name: bert_Dot_122_0	type: float	shape: Shape{32768, 1024}
void Dot_float_float_float_cuda_lib_Dot_122(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 2, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 2, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: bert_Sum_157_0	type: float	shape: Shape{256, 128}
//	- name: bert_Constant_158_0	type: float	shape: Shape{256, 128}
//	- name: bert_Reshape_161_0	type: float	shape: Shape{1}
// Output:
//	- name: bert_Rsqrt_164_0	type: float	shape: Shape{256, 128, 1}
// Fused functions:
// Divide_float_float_float_cuda_Divide_159<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(bert_Sum_157_0, bert_Constant_158_0, Divide_159_0);
// Reshape_float_float_cuda_lib_Reshape_160(Divide_159_0, Reshape_160_0);
// Broadcast_float_float_cuda_Broadcast_162<<<dim3(512, 1, 1), dim3(64, 1, 1), 0, 0>>>(bert_Reshape_161_0, Broadcast_162_0);
// Add_float_float_float_cuda_Add_163<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_160_0, Broadcast_162_0, Add_163_0);
// Rsqrt_float_float_cuda_Rsqrt_164<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_163_0, bert_Rsqrt_164_0);
extern "C" __launch_bounds__(512) __global__ void bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = fdividef(input0[tid], input1[tid]);
    float temp1 = input2[tid % 1];
    float temp2 = add(temp0, temp1);
    float temp3 = rsqrtf(temp2);
    output0[tid] = temp3;

}
extern void bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Reshape_114
// Description:	Reshape
// Input:
//	- name: Parameter_2_0	type: int32_t	shape: Shape{256, 128}
// Output:
//	- name: bert_Reshape_114_0	type: int32_t	shape: Shape{32768}
void Reshape_int32_t_int32_t_cuda_lib_Reshape_114(cudaStream_t stream, int32_t* input0, int32_t* output0)
{
    if (input0 != output0) {
       cudaMemcpyAsync(output0, input0, 32768 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    }

}
// Node name:	Slice_384
// Description:	Slice
// Input:
//	- name: bert_Reshape_383_0	type: float	shape: Shape{256, 128, 1024}
// Output:
//	- name: bert_Slice_384_0	type: float	shape: Shape{256, 1, 1024}
extern "C" __launch_bounds__(64) __global__ void bert_Slice_float_float_cuda_Slice_384(float* input0, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 262144)
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
extern void bert_Slice_float_float_cuda_Slice_384_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Slice_float_float_cuda_Slice_384<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_365
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_365_0	type: float	shape: Shape{32768}
void bert_Constant_float_cuda_Constant_365(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_365_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_365_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Dot_249
// Description:	Dot
// Input:
//	- name: bert_Multiply_248_0	type: float	shape: Shape{32768, 4096}
//	- name: bert_Constant_59_0	type: float	shape: Shape{4096, 1024}
// Output:
//	- name: bert_Dot_249_0	type: float	shape: Shape{32768, 1024}
void Dot_float_float_float_cuda_lib_Dot_249(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 4096, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1024));

}
// Node name:	BatchMatMul_199
// Description:	BatchMatMul
// Input:
//	- name: bert_Softmax_198_0	type: float	shape: Shape{256, 16, 128, 128}
//	- name: bert_Reshape_191_0	type: float	shape: Shape{256, 16, 128, 64}
// Output:
//	- name: bert_BatchMatMul_199_0	type: float	shape: Shape{256, 16, 128, 64}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 128, 128,
                                    &alpha, input1, 64, 8192, input0, 128, 16384,
                                    &beta, output0, 64, 8192, 4096));
                            
    }

}
// Node name:	Reshape_200
// Description:	Reshape
// Input:
//	- name: bert_BatchMatMul_199_0	type: float	shape: Shape{256, 16, 128, 64}
// Output:
//	- name: bert_Reshape_200_0	type: float	shape: Shape{256, 128, 16, 64}
extern "C" __launch_bounds__(64) __global__ void bert_Reshape_float_float_cuda_Reshape_200(float* input0, float* output0)
{
    uint32_t input_strides0 = 131072;
    uint32_t input_strides1 = 8192;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 131072;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 1024;
    uint32_t trans_strides3 = 1;
    size_t n = 33554432;
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
extern void bert_Reshape_float_float_cuda_Reshape_200_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Reshape_float_float_cuda_Reshape_200<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_62
// Description:	Constant
// Input:
// Output:
//	- name: bert_Constant_62_0	type: float	shape: Shape{1024}
void bert_Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/bert/Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load bert_Constant_62_0 failed.\n");
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
//	- name: bert_Dot_390_0	type: float	shape: Shape{256, 1001}
//	- name: bert_Broadcast_391_0	type: float	shape: Shape{256, 1001}
// Output:
//	- name: bert_Add_392_0	type: float	shape: Shape{256, 1001}
extern "C" __launch_bounds__(256) __global__ void bert_Add_float_float_float_cuda_Add_392(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 256 + threadIdx.x] = add(input0[blockIdx.x * 256 + threadIdx.x], input1[blockIdx.x * 256 + threadIdx.x]);

}
extern void bert_Add_float_float_float_cuda_Add_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_Add_float_float_float_cuda_Add_392<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Dot_233
// Description:	Dot
// Input:
//	- name: bert_Add_232_0	type: float	shape: Shape{32768, 1024}
//	- name: bert_Constant_52_0	type: float	shape: Shape{1024, 4096}
// Output:
//	- name: bert_Dot_233_0	type: float	shape: Shape{32768, 4096}
void Dot_float_float_float_cuda_lib_Dot_233(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32768, 1024, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 4096));

}

extern "C" void bert_cuda_init()
{
// total memory:1630418240

CUDA_SAFE_CALL(cudaMalloc((void**)&bert_group_0_CUDA_GPU0_allocator_memory_pool,1392902144));
CUDA_SAFE_CALL(cudaMemset((void*)bert_group_0_CUDA_GPU0_allocator_memory_pool, 0, 1392902144));
bert_Reshape_114_0 = (int32_t*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_OneHot_118_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Dot_122_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+393216);
bert_Reshape_113_0 = (int32_t*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_117_0 = (int32_t*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_GatherV2_121_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134610944);
bert_Add_142_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268828672);
bert_Sum_149_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Divide_151_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_152_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_153_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Multiply_156_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Sum_157_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134348800);
bert_Rsqrt_164_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Reshape_165_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Add_175_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Reshape_176_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Dot_179_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Add_185_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Reshape_188_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Reshape_191_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Dot_178_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Add_183_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+402915328);
bert_Reshape_187_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+402915328);
bert_Reshape_190_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Dot_177_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+402915328);
bert_Add_181_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+537133056);
bert_Reshape_186_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+537133056);
bert_Reshape_189_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+402915328);
bert_BatchMatMul_192_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+537133056);
bert_Broadcast_116_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_124_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_115_0 = (int32_t*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Convert_120_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Reshape_126_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Broadcast_127_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268828672);
bert_Multiply_145_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+285605888);
bert_Multiply_148_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+302383104);
bert_Reshape_195_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+285605888);
bert_Broadcast_196_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+805568512);
bert_Add_197_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+1074003968);
bert_Softmax_198_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+1074003968);
bert_BatchMatMul_199_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Reshape_200_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Reshape_201_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Dot_202_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Add_205_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Sum_206_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Divide_208_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_209_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_210_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Multiply_213_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Sum_214_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134348800);
bert_Rsqrt_221_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Reshape_222_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Add_232_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Dot_233_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Multiply_248_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+856031232);
bert_Dot_249_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Add_252_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Sum_253_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Divide_255_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_256_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_257_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Multiply_260_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Sum_261_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134348800);
bert_Rsqrt_268_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Reshape_269_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Add_279_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Dot_282_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Add_288_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Reshape_291_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Reshape_294_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Dot_281_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Add_286_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+453378048);
bert_Reshape_290_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+453378048);
bert_Reshape_293_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Dot_280_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+453378048);
bert_Add_284_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+587595776);
bert_Reshape_289_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+587595776);
bert_Reshape_292_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+453378048);
bert_BatchMatMul_295_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+587595776);
bert_Reshape_298_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+302383104);
bert_Broadcast_299_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+319160320);
bert_Add_300_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+856031232);
bert_Softmax_301_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+856031232);
bert_BatchMatMul_302_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Reshape_303_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Reshape_304_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Dot_305_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Add_308_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Sum_309_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Divide_311_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_312_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_313_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Multiply_316_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Sum_317_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134348800);
bert_Rsqrt_324_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Reshape_325_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Add_335_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Dot_336_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Multiply_351_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+671350784);
bert_Dot_352_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Add_355_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+268697600);
bert_Sum_356_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Divide_358_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_359_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Reshape_360_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Multiply_363_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Sum_364_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134348800);
bert_Rsqrt_371_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Reshape_372_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+131072);
bert_Add_382_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Reshape_383_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+262144);
bert_Slice_384_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Reshape_385_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+134479872);
bert_Dot_386_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Tanh_389_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+1048576);
bert_Dot_390_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Broadcast_391_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+1025024);
bert_Add_392_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+0);
bert_Softmax_393_0 = (float*)(bert_group_0_CUDA_GPU0_allocator_memory_pool+1025024);

CUDA_SAFE_CALL(cudaMalloc((void**)&bert_group_persist_CUDA_GPU0_allocator_memory_pool,237516096));
CUDA_SAFE_CALL(cudaMemset((void*)bert_group_persist_CUDA_GPU0_allocator_memory_pool, 0, 237516096));
bert_Constant_110_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+0);
bert_Constant_109_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+4096);
bert_Constant_46_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+4198400);
bert_Constant_32_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+4202496);
bert_Constant_31_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+4206592);
bert_Constant_8_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+8400896);
bert_Constant_4_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+8409088);
bert_Constant_14_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
bert_Slice_119_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
bert_Reshape_123_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
bert_Reshape_140_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+133427200);
bert_Constant_19_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135524352);
bert_Constant_18_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135528448);
bert_Constant_150_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135532544);
bert_Constant_22_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135663616);
bert_Reshape_161_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135663616);
bert_Constant_158_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135663680);
bert_Constant_37_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135794752);
bert_Constant_30_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135794816);
bert_Constant_29_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+135798912);
bert_Constant_28_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+139993216);
bert_Constant_27_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+139997312);
bert_Constant_79_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191616);
bert_Reshape_146_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191616);
bert_Constant_40_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191680);
bert_Reshape_143_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191680);
bert_Constant_78_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191744);
bert_Reshape_137_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191744);
bert_Constant_39_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191808);
bert_Reshape_134_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191808);
bert_Constant_25_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191872);
bert_Constant_45_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+144191936);
bert_Constant_48_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148386240);
bert_Constant_47_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148390336);
bert_Constant_207_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148394432);
bert_Constant_51_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148525504);
bert_Reshape_218_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148525504);
bert_Constant_215_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148525568);
bert_Constant_60_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148656640);
bert_Constant_58_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148660736);
bert_Constant_56_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148660800);
bert_Constant_55_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148660864);
bert_Constant_53_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148660928);
bert_Constant_57_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148677312);
bert_Constant_54_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148677376);
bert_Constant_52_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+148677440);
bert_Constant_59_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+165454656);
bert_Constant_62_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182231872);
bert_Constant_61_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182235968);
bert_Constant_254_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182240064);
bert_Constant_65_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182371136);
bert_Reshape_265_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182371136);
bert_Constant_262_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182371200);
bert_Constant_85_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182502272);
bert_Constant_71_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182506368);
bert_Constant_70_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+182510464);
bert_Constant_76_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+186704768);
bert_Constant_69_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+186704832);
bert_Constant_68_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+186708928);
bert_Constant_67_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+190903232);
bert_Constant_66_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+190907328);
bert_Constant_84_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+195101632);
bert_Constant_87_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199295936);
bert_Constant_86_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199300032);
bert_Constant_310_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199304128);
bert_Constant_90_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199435200);
bert_Reshape_321_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199435200);
bert_Constant_318_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199435264);
bert_Constant_99_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199566336);
bert_Constant_97_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199570432);
bert_Constant_95_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199570496);
bert_Constant_94_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199570560);
bert_Constant_92_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199570624);
bert_Constant_96_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199587008);
bert_Constant_93_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199587072);
bert_Constant_91_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+199587136);
bert_Constant_98_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+216364352);
bert_Constant_101_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233141568);
bert_Constant_100_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233145664);
bert_Constant_357_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233149760);
bert_Constant_104_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233280832);
bert_Reshape_368_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233280832);
bert_Constant_365_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233280896);
bert_Constant_111_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+233411968);
bert_Constant_112_0 = (float*)(bert_group_persist_CUDA_GPU0_allocator_memory_pool+237512064);
// create streams/handles
CUDNN_SAFE_CALL(cudnnCreate(&bert_cudnn_handle_0));
CUBLAS_SAFE_CALL(cublasCreate(&bert_cublas_handle_0));
 // name=bert/pooler/dense/bias
bert_Constant_float_cuda_Constant_110(0, bert_Constant_110_0);
 // name=bert/pooler/dense/kernel
bert_Constant_float_cuda_Constant_109(0, bert_Constant_109_0);
 // name=bert/encoder/layer_0/attention/output/dense/bias
bert_Constant_float_cuda_Constant_46(0, bert_Constant_46_0);
 // name=bert/encoder/layer_0/attention/self/value/bias
bert_Constant_float_cuda_Constant_32(0, bert_Constant_32_0);
 // name=bert/encoder/layer_0/attention/self/value/kernel
bert_Constant_float_cuda_Constant_31(0, bert_Constant_31_0);
 // name=bert/embeddings/token_type_embeddings
bert_Constant_float_cuda_Constant_8(0, bert_Constant_8_0);
 // name=bert/embeddings/word_embeddings
bert_Constant_float_cuda_Constant_4(0, bert_Constant_4_0);
 // name=bert/embeddings/position_embeddings
bert_Constant_float_cuda_Constant_14(0, bert_Constant_14_0);
 // name=bert/embeddings/LayerNorm/gamma
bert_Constant_float_cuda_Constant_19(0, bert_Constant_19_0);
 // name=bert/embeddings/LayerNorm/beta
bert_Constant_float_cuda_Constant_18(0, bert_Constant_18_0);
 // name=Constant_150
bert_Constant_float_cuda_Constant_150(0, bert_Constant_150_0);
 // name=bert/embeddings/LayerNorm/batchnorm/add/y
bert_Constant_float_cuda_Constant_22(0, bert_Constant_22_0);
 // name=Constant_158
bert_Constant_float_cuda_Constant_158(0, bert_Constant_158_0);
 // name=bert/encoder/layer_0/attention/self/Mul/y
bert_Constant_float_cuda_Constant_37(0, bert_Constant_37_0);
 // name=bert/encoder/layer_0/attention/self/key/bias
bert_Constant_float_cuda_Constant_30(0, bert_Constant_30_0);
 // name=bert/encoder/layer_0/attention/self/key/kernel
bert_Constant_float_cuda_Constant_29(0, bert_Constant_29_0);
 // name=bert/encoder/layer_0/attention/self/query/bias
bert_Constant_float_cuda_Constant_28(0, bert_Constant_28_0);
 // name=bert/encoder/layer_0/attention/self/query/kernel
bert_Constant_float_cuda_Constant_27(0, bert_Constant_27_0);
 // name=bert/encoder/layer_1/attention/self/mul_1/y
bert_Constant_float_cuda_Constant_79(0, bert_Constant_79_0);
 // name=bert/encoder/layer_0/attention/self/mul_1/y
bert_Constant_float_cuda_Constant_40(0, bert_Constant_40_0);
 // name=bert/encoder/layer_1/attention/self/sub/x
bert_Constant_float_cuda_Constant_78(0, bert_Constant_78_0);
 // name=bert/encoder/layer_0/attention/self/sub/x
bert_Constant_float_cuda_Constant_39(0, bert_Constant_39_0);
 // name=bert/encoder/ones/Const
bert_Constant_float_cuda_Constant_25(0, bert_Constant_25_0);
 // name=bert/encoder/layer_0/attention/output/dense/kernel
bert_Constant_float_cuda_Constant_45(0, bert_Constant_45_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/gamma
bert_Constant_float_cuda_Constant_48(0, bert_Constant_48_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/beta
bert_Constant_float_cuda_Constant_47(0, bert_Constant_47_0);
 // name=Constant_207
bert_Constant_float_cuda_Constant_207(0, bert_Constant_207_0);
 // name=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y
bert_Constant_float_cuda_Constant_51(0, bert_Constant_51_0);
 // name=Constant_215
bert_Constant_float_cuda_Constant_215(0, bert_Constant_215_0);
 // name=bert/encoder/layer_0/output/dense/bias
bert_Constant_float_cuda_Constant_60(0, bert_Constant_60_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul_2/x
bert_Constant_float_cuda_Constant_58(0, bert_Constant_58_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul_1/x
bert_Constant_float_cuda_Constant_56(0, bert_Constant_56_0);
 // name=bert/encoder/layer_0/intermediate/dense/mul/x
bert_Constant_float_cuda_Constant_55(0, bert_Constant_55_0);
 // name=bert/encoder/layer_0/intermediate/dense/bias
bert_Constant_float_cuda_Constant_53(0, bert_Constant_53_0);
 // name=bert/encoder/layer_0/intermediate/dense/add_1/x
bert_Constant_float_cuda_Constant_57(0, bert_Constant_57_0);
 // name=bert/encoder/layer_0/intermediate/dense/Pow/y
bert_Constant_float_cuda_Constant_54(0, bert_Constant_54_0);
 // name=bert/encoder/layer_0/intermediate/dense/kernel
bert_Constant_float_cuda_Constant_52(0, bert_Constant_52_0);
 // name=bert/encoder/layer_0/output/dense/kernel
bert_Constant_float_cuda_Constant_59(0, bert_Constant_59_0);
 // name=bert/encoder/layer_0/output/LayerNorm/gamma
bert_Constant_float_cuda_Constant_62(0, bert_Constant_62_0);
 // name=bert/encoder/layer_0/output/LayerNorm/beta
bert_Constant_float_cuda_Constant_61(0, bert_Constant_61_0);
 // name=Constant_254
bert_Constant_float_cuda_Constant_254(0, bert_Constant_254_0);
 // name=bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y
bert_Constant_float_cuda_Constant_65(0, bert_Constant_65_0);
 // name=Constant_262
bert_Constant_float_cuda_Constant_262(0, bert_Constant_262_0);
 // name=bert/encoder/layer_1/attention/output/dense/bias
bert_Constant_float_cuda_Constant_85(0, bert_Constant_85_0);
 // name=bert/encoder/layer_1/attention/self/value/bias
bert_Constant_float_cuda_Constant_71(0, bert_Constant_71_0);
 // name=bert/encoder/layer_1/attention/self/value/kernel
bert_Constant_float_cuda_Constant_70(0, bert_Constant_70_0);
 // name=bert/encoder/layer_1/attention/self/Mul/y
bert_Constant_float_cuda_Constant_76(0, bert_Constant_76_0);
 // name=bert/encoder/layer_1/attention/self/key/bias
bert_Constant_float_cuda_Constant_69(0, bert_Constant_69_0);
 // name=bert/encoder/layer_1/attention/self/key/kernel
bert_Constant_float_cuda_Constant_68(0, bert_Constant_68_0);
 // name=bert/encoder/layer_1/attention/self/query/bias
bert_Constant_float_cuda_Constant_67(0, bert_Constant_67_0);
 // name=bert/encoder/layer_1/attention/self/query/kernel
bert_Constant_float_cuda_Constant_66(0, bert_Constant_66_0);
 // name=bert/encoder/layer_1/attention/output/dense/kernel
bert_Constant_float_cuda_Constant_84(0, bert_Constant_84_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/gamma
bert_Constant_float_cuda_Constant_87(0, bert_Constant_87_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/beta
bert_Constant_float_cuda_Constant_86(0, bert_Constant_86_0);
 // name=Constant_310
bert_Constant_float_cuda_Constant_310(0, bert_Constant_310_0);
 // name=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y
bert_Constant_float_cuda_Constant_90(0, bert_Constant_90_0);
 // name=Constant_318
bert_Constant_float_cuda_Constant_318(0, bert_Constant_318_0);
 // name=bert/encoder/layer_1/output/dense/bias
bert_Constant_float_cuda_Constant_99(0, bert_Constant_99_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul_2/x
bert_Constant_float_cuda_Constant_97(0, bert_Constant_97_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul_1/x
bert_Constant_float_cuda_Constant_95(0, bert_Constant_95_0);
 // name=bert/encoder/layer_1/intermediate/dense/mul/x
bert_Constant_float_cuda_Constant_94(0, bert_Constant_94_0);
 // name=bert/encoder/layer_1/intermediate/dense/bias
bert_Constant_float_cuda_Constant_92(0, bert_Constant_92_0);
 // name=bert/encoder/layer_1/intermediate/dense/add_1/x
bert_Constant_float_cuda_Constant_96(0, bert_Constant_96_0);
 // name=bert/encoder/layer_1/intermediate/dense/Pow/y
bert_Constant_float_cuda_Constant_93(0, bert_Constant_93_0);
 // name=bert/encoder/layer_1/intermediate/dense/kernel
bert_Constant_float_cuda_Constant_91(0, bert_Constant_91_0);
 // name=bert/encoder/layer_1/output/dense/kernel
bert_Constant_float_cuda_Constant_98(0, bert_Constant_98_0);
 // name=bert/encoder/layer_1/output/LayerNorm/gamma
bert_Constant_float_cuda_Constant_101(0, bert_Constant_101_0);
 // name=bert/encoder/layer_1/output/LayerNorm/beta
bert_Constant_float_cuda_Constant_100(0, bert_Constant_100_0);
 // name=Constant_357
bert_Constant_float_cuda_Constant_357(0, bert_Constant_357_0);
 // name=bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y
bert_Constant_float_cuda_Constant_104(0, bert_Constant_104_0);
 // name=Constant_365
bert_Constant_float_cuda_Constant_365(0, bert_Constant_365_0);
 // name=dense/kernel
bert_Constant_float_cuda_Constant_111(0, bert_Constant_111_0);
 // name=dense/bias
bert_Constant_float_cuda_Constant_112(0, bert_Constant_112_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&bert_num_SMs, cudaDevAttrMultiProcessorCount, 0));
}




extern "C" void bert_cuda_free()
{

CUDA_SAFE_CALL(cudaFree(bert_group_0_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(bert_group_persist_CUDA_GPU0_allocator_memory_pool));
CUDNN_SAFE_CALL(cudnnDestroy(bert_cudnn_handle_0));
CUBLAS_SAFE_CALL(cublasDestroy(bert_cublas_handle_0));
}

#include "./include/dnn.h"

class bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114Kernel : public Kernel {
public:
    bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114Kernel(cudaStream_t  stream, int32_t*  input0, int32_t*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cudaStream_t  stream; int32_t*  input0; int32_t*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Reshape_int32_t_int32_t_cuda_lib_Reshape_114(cudaStream_t stream, int32_t* input0, int32_t* output0)
{
    if (input0 != output0) {
       cudaMemcpyAsync(output0, input0, 32768 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    }

}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_int32_t_int32_t_cuda_lib_Reshape_114(stream, input0, output0);
    }
};


class bert_OneHot_int32_t_float_cuda_OneHot_118_CallKernel : public Kernel {
public:
    bert_OneHot_int32_t_float_cuda_OneHot_118_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, int32_t*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_OneHot_int32_t_float_cuda_OneHot_118_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; int32_t*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void OneHot_int32_t_float_cuda_OneHot_118_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    bert_OneHot_int32_t_float_cuda_OneHot_118<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->OneHot_int32_t_float_cuda_OneHot_118_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_122Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_122Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_122";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1024;
    ret[1] = 32768;
    ret[2] = 2;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_122(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 2, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 2, &beta, static_cast<float*>(output0), 1024));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_122(cublas_handle, input0, input1, output0);
    }
};


class bert_GatherV2_float_int32_t_float_cuda_GatherV2_121_CallKernel : public Kernel {
public:
    bert_GatherV2_float_int32_t_float_cuda_GatherV2_121_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, int32_t*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_GatherV2_float_int32_t_float_cuda_GatherV2_121_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; int32_t*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void GatherV2_float_int32_t_float_cuda_GatherV2_121_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int32_t* input1, float* output0) {
    bert_GatherV2_float_int32_t_float_cuda_GatherV2_121<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->GatherV2_float_int32_t_float_cuda_GatherV2_121_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class bert_Sum_float_float_cuda_Sum_149_CallKernel : public Kernel {
public:
    bert_Sum_float_float_cuda_Sum_149_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Sum_float_float_cuda_Sum_149_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Sum_float_float_cuda_Sum_149_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Sum_float_float_cuda_Sum_149<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Sum_float_float_cuda_Sum_149_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Divide_float_float_float_cuda_Divide_151_CallKernel : public Kernel {
public:
    bert_Divide_float_float_float_cuda_Divide_151_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Divide_float_float_float_cuda_Divide_151_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Divide_float_float_float_cuda_Divide_151_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_Divide_float_float_float_cuda_Divide_151<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Divide_float_float_float_cuda_Divide_151_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  input3; float*  input4; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_179Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_179";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1024;
    ret[1] = 32768;
    ret[2] = 1024;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_179(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_179(cublas_handle, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Add_7_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_Reshape_float_float_cuda_Reshape_191_CallKernel : public Kernel {
public:
    bert_Reshape_float_float_cuda_Reshape_191_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Reshape_float_float_cuda_Reshape_191_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_191_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Reshape_float_float_cuda_Reshape_191<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_191_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192Kernel : public Kernel {
public:
    bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 128, 128, 64,
                                    &alpha, input1, 64, 8192, input0, 64, 8192,
                                    &beta, output0, 128, 16384, 4096));
                            
    }

}

    void executeImpl(cudaStream_t stream) {
        this->BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192(cublas_handle, input0, input1, output0);
    }
};


class bert_Broadcast_float_float_cuda_Broadcast_116_CallKernel : public Kernel {
public:
    bert_Broadcast_float_float_cuda_Broadcast_116_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Broadcast_float_float_cuda_Broadcast_116_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_116_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_116<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_116_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Convert_int32_t_float_cuda_Convert_120_CallKernel : public Kernel {
public:
    bert_Convert_int32_t_float_cuda_Convert_120_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, int32_t*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Convert_int32_t_float_cuda_Convert_120_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; int32_t*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Convert_int32_t_float_cuda_Convert_120_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int32_t* input0, float* output0) {
    bert_Convert_int32_t_float_cuda_Convert_120<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Convert_int32_t_float_cuda_Convert_120_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Broadcast_float_float_cuda_Broadcast_127_CallKernel : public Kernel {
public:
    bert_Broadcast_float_float_cuda_Broadcast_127_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Broadcast_float_float_cuda_Broadcast_127_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_127_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_127<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_127_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  input5, float*  output0, float*  output1, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->input5 = input5, this->output0 = output0, this->output1 = output1, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  input3; float*  input4; float*  input5; float*  output0; float*  output1;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, input5, output0, output1);
    }
};


class bert_Broadcast_float_float_cuda_Broadcast_196_CallKernel : public Kernel {
public:
    bert_Broadcast_float_float_cuda_Broadcast_196_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Broadcast_float_float_cuda_Broadcast_196_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_196_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_196<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_196_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class bert_Softmax_float_float_cuda_lib_Softmax_198Kernel : public Kernel {
public:
    bert_Softmax_float_float_cuda_lib_Softmax_198Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Softmax_float_float_cuda_lib_Softmax_198";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cudnnHandle_t  cudnn_handle; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Softmax_float_float_cuda_lib_Softmax_198(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 524288, 1, 1, 128));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 524288, 1, 1, 128));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor_desc, input0, &beta, output_tensor_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));

}

    void executeImpl(cudaStream_t stream) {
        this->Softmax_float_float_cuda_lib_Softmax_198(cudnn_handle, input0, output0);
    }
};


class bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199Kernel : public Kernel {
public:
    bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 128, 128,
                                    &alpha, input1, 64, 8192, input0, 128, 16384,
                                    &beta, output0, 64, 8192, 4096));
                            
    }

}

    void executeImpl(cudaStream_t stream) {
        this->BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199(cublas_handle, input0, input1, output0);
    }
};


class bert_Reshape_float_float_cuda_Reshape_200_CallKernel : public Kernel {
public:
    bert_Reshape_float_float_cuda_Reshape_200_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Reshape_float_float_cuda_Reshape_200_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_200_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Reshape_float_float_cuda_Reshape_200<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_200_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_233Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_233Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_233";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 4096;
    ret[1] = 32768;
    ret[2] = 1024;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_233(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32768, 1024, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 4096));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_233(cublas_handle, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  input5, float*  input6, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->input5 = input5, this->input6 = input6, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  input3; float*  input4; float*  input5; float*  input6; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0) {
    bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, input5, input6, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_249Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_249Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_249";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1024;
    ret[1] = 32768;
    ret[2] = 4096;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_249(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 32768, 4096, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1024));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_249(cublas_handle, input0, input1, output0);
    }
};


class bert_Slice_float_float_cuda_Slice_384_CallKernel : public Kernel {
public:
    bert_Slice_float_float_cuda_Slice_384_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Slice_float_float_cuda_Slice_384_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Slice_float_float_cuda_Slice_384_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Slice_float_float_cuda_Slice_384<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Slice_float_float_cuda_Slice_384_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_386Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_386Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_386";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1024;
    ret[1] = 256;
    ret[2] = 1024;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_386(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 256, 1024, &alpha, static_cast<const float*>(input1), 1024, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1024));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_386(cublas_handle, input0, input1, output0);
    }
};


class bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_CallKernel : public Kernel {
public:
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_Dot_float_float_float_cuda_lib_Dot_390Kernel : public Kernel {
public:
    bert_Dot_float_float_float_cuda_lib_Dot_390Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Dot_float_float_float_cuda_lib_Dot_390";
        this->Id = 0;
        this->mixable = 1;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cublasHandle_t  cublas_handle; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1001;
    ret[1] = 256;
    ret[2] = 1024;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_390(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 256, 1024, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 1024, &beta, static_cast<float*>(output0), 1001));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_390(cublas_handle, input0, input1, output0);
    }
};


class bert_Broadcast_float_float_cuda_Broadcast_391_CallKernel : public Kernel {
public:
    bert_Broadcast_float_float_cuda_Broadcast_391_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Broadcast_float_float_cuda_Broadcast_391_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_391_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    bert_Broadcast_float_float_cuda_Broadcast_391<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_391_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class bert_Add_float_float_float_cuda_Add_392_CallKernel : public Kernel {
public:
    bert_Add_float_float_float_cuda_Add_392_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Add_float_float_float_cuda_Add_392_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    bert_Add_float_float_float_cuda_Add_392<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Add_float_float_float_cuda_Add_392_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class bert_Softmax_float_float_cuda_lib_Softmax_393Kernel : public Kernel {
public:
    bert_Softmax_float_float_cuda_lib_Softmax_393Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Softmax_float_float_cuda_lib_Softmax_393";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cudnnHandle_t  cudnn_handle; float*  input0; float*  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Softmax_float_float_cuda_lib_Softmax_393(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 256, 1, 1, 1001));
    cudnnTensorDescriptor_t output_tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 256, 1, 1, 1001));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor_desc, input0, &beta, output_tensor_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));

}

    void executeImpl(cudaStream_t stream) {
        this->Softmax_float_float_cuda_lib_Softmax_393(cudnn_handle, input0, output0);
    }
};


class bert_Result_float_float_cuda_lib_Result_394Kernel : public Kernel {
public:
    bert_Result_float_float_cuda_lib_Result_394Kernel(float*  input0, float**  output0, int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Parameter_1_0 = Parameter_1_0, this->Parameter_2_0 = Parameter_2_0, this->Result_394_0 = Result_394_0;
        this->kernelName = "bert_Result_float_float_cuda_lib_Result_394";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        float*  input0; float**  output0;
    int32_t*  Parameter_0_0; int32_t*  Parameter_1_0; int32_t*  Parameter_2_0; float**  Result_394_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_394(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        this->Result_float_float_cuda_lib_Result_394(input0, output0);
    }
};
void Bert::gen_vector(int32_t*  Parameter_0_0, int32_t*  Parameter_1_0, int32_t*  Parameter_2_0, float**  Result_394_0) {
    kernels.emplace_back(new bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114Kernel(0, std::move(Parameter_2_0), std::move(bert_Reshape_114_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_OneHot_int32_t_float_cuda_OneHot_118_CallKernel(dim3(512, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_114_0), std::move(bert_OneHot_118_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_122Kernel(std::move(bert_cublas_handle_0), std::move(bert_OneHot_118_0), std::move(bert_Constant_8_0), std::move(bert_Dot_122_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114Kernel(0, std::move(Parameter_0_0), std::move(bert_Reshape_113_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_GatherV2_float_int32_t_float_cuda_GatherV2_121_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Constant_4_0), std::move(bert_Reshape_117_0), std::move(bert_GatherV2_121_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Reshape_Reshape_Add_Broadcast_Add_0_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_GatherV2_121_0), std::move(bert_Dot_122_0), std::move(bert_Reshape_140_0), std::move(bert_Add_142_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Add_142_0), std::move(bert_Sum_149_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Divide_float_float_float_cuda_Divide_151_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_149_0), std::move(bert_Constant_150_0), std::move(bert_Divide_151_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_153_0), std::move(bert_Add_142_0), std::move(bert_Multiply_156_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Multiply_156_0), std::move(bert_Sum_157_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_157_0), std::move(bert_Constant_158_0), std::move(bert_Reshape_161_0), std::move(bert_Rsqrt_164_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_18_0), std::move(bert_Reshape_153_0), std::move(bert_Reshape_165_0), std::move(bert_Constant_19_0), std::move(bert_Add_142_0), std::move(bert_Add_175_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_176_0), std::move(bert_Constant_31_0), std::move(bert_Dot_179_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_32_0), std::move(bert_Dot_179_0), std::move(bert_Add_185_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_188_0), std::move(bert_Reshape_191_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_176_0), std::move(bert_Constant_29_0), std::move(bert_Dot_178_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_30_0), std::move(bert_Dot_178_0), std::move(bert_Add_183_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_187_0), std::move(bert_Reshape_190_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_176_0), std::move(bert_Constant_27_0), std::move(bert_Dot_177_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_28_0), std::move(bert_Dot_177_0), std::move(bert_Add_181_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_186_0), std::move(bert_Reshape_189_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_189_0), std::move(bert_Reshape_190_0), std::move(bert_BatchMatMul_192_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Broadcast_float_float_cuda_Broadcast_116_CallKernel(dim3(512, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Constant_25_0), std::move(bert_Broadcast_116_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_int32_t_int32_t_cuda_lib_Reshape_114Kernel(0, std::move(Parameter_1_0), std::move(bert_Reshape_115_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Convert_int32_t_float_cuda_Convert_120_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_115_0), std::move(bert_Convert_120_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Broadcast_float_float_cuda_Broadcast_127_CallKernel(dim3(65536, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_126_0), std::move(bert_Broadcast_127_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Multiply_Reshape_Subtract_Broadcast_Multiply_Broadcast_Subtract_Broadcast_Multiply_1_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_134_0), std::move(bert_Reshape_124_0), std::move(bert_Broadcast_127_0), std::move(bert_Reshape_143_0), std::move(bert_Reshape_137_0), std::move(bert_Reshape_146_0), std::move(bert_Multiply_148_0), std::move(bert_Multiply_145_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Broadcast_float_float_cuda_Broadcast_196_CallKernel(dim3(1048576, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_195_0), std::move(bert_Broadcast_196_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_CallKernel(dim3(131072, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_37_0), std::move(bert_BatchMatMul_192_0), std::move(bert_Broadcast_196_0), std::move(bert_Add_197_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Softmax_float_float_cuda_lib_Softmax_198Kernel(std::move(bert_cudnn_handle_0), std::move(bert_Add_197_0), std::move(bert_Softmax_198_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199Kernel(std::move(bert_cublas_handle_0), std::move(bert_Softmax_198_0), std::move(bert_Reshape_191_0), std::move(bert_BatchMatMul_199_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_200_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_BatchMatMul_199_0), std::move(bert_Reshape_200_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_201_0), std::move(bert_Constant_45_0), std::move(bert_Dot_202_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_46_0), std::move(bert_Dot_202_0), std::move(bert_Reshape_176_0), std::move(bert_Add_205_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Add_205_0), std::move(bert_Sum_206_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Divide_float_float_float_cuda_Divide_151_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_206_0), std::move(bert_Constant_207_0), std::move(bert_Divide_208_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_210_0), std::move(bert_Add_205_0), std::move(bert_Multiply_213_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Multiply_213_0), std::move(bert_Sum_214_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_214_0), std::move(bert_Constant_215_0), std::move(bert_Reshape_218_0), std::move(bert_Rsqrt_221_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_47_0), std::move(bert_Reshape_210_0), std::move(bert_Reshape_222_0), std::move(bert_Constant_48_0), std::move(bert_Add_205_0), std::move(bert_Add_232_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_233Kernel(std::move(bert_cublas_handle_0), std::move(bert_Add_232_0), std::move(bert_Constant_52_0), std::move(bert_Dot_233_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_CallKernel(dim3(262144, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_58_0), std::move(bert_Constant_57_0), std::move(bert_Constant_56_0), std::move(bert_Constant_55_0), std::move(bert_Constant_53_0), std::move(bert_Dot_233_0), std::move(bert_Constant_54_0), std::move(bert_Multiply_248_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_249Kernel(std::move(bert_cublas_handle_0), std::move(bert_Multiply_248_0), std::move(bert_Constant_59_0), std::move(bert_Dot_249_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_60_0), std::move(bert_Dot_249_0), std::move(bert_Add_232_0), std::move(bert_Add_252_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Add_252_0), std::move(bert_Sum_253_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Divide_float_float_float_cuda_Divide_151_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_253_0), std::move(bert_Constant_254_0), std::move(bert_Divide_255_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_257_0), std::move(bert_Add_252_0), std::move(bert_Multiply_260_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Multiply_260_0), std::move(bert_Sum_261_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_261_0), std::move(bert_Constant_262_0), std::move(bert_Reshape_265_0), std::move(bert_Rsqrt_268_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_61_0), std::move(bert_Reshape_257_0), std::move(bert_Reshape_269_0), std::move(bert_Constant_62_0), std::move(bert_Add_252_0), std::move(bert_Add_279_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Add_279_0), std::move(bert_Constant_70_0), std::move(bert_Dot_282_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_71_0), std::move(bert_Dot_282_0), std::move(bert_Add_288_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_291_0), std::move(bert_Reshape_294_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Add_279_0), std::move(bert_Constant_68_0), std::move(bert_Dot_281_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_69_0), std::move(bert_Dot_281_0), std::move(bert_Add_286_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_290_0), std::move(bert_Reshape_293_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Add_279_0), std::move(bert_Constant_66_0), std::move(bert_Dot_280_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_7_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_67_0), std::move(bert_Dot_280_0), std::move(bert_Add_284_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_191_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_289_0), std::move(bert_Reshape_292_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_192Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_292_0), std::move(bert_Reshape_293_0), std::move(bert_BatchMatMul_295_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Broadcast_float_float_cuda_Broadcast_196_CallKernel(dim3(1048576, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_298_0), std::move(bert_Broadcast_299_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Multiply_Add_8_CallKernel(dim3(131072, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_76_0), std::move(bert_BatchMatMul_295_0), std::move(bert_Broadcast_299_0), std::move(bert_Add_300_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Softmax_float_float_cuda_lib_Softmax_198Kernel(std::move(bert_cudnn_handle_0), std::move(bert_Add_300_0), std::move(bert_Softmax_301_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_199Kernel(std::move(bert_cublas_handle_0), std::move(bert_Softmax_301_0), std::move(bert_Reshape_294_0), std::move(bert_BatchMatMul_302_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Reshape_float_float_cuda_Reshape_200_CallKernel(dim3(524288, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_BatchMatMul_302_0), std::move(bert_Reshape_303_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_179Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_304_0), std::move(bert_Constant_84_0), std::move(bert_Dot_305_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_85_0), std::move(bert_Dot_305_0), std::move(bert_Add_279_0), std::move(bert_Add_308_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Add_308_0), std::move(bert_Sum_309_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Divide_float_float_float_cuda_Divide_151_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_309_0), std::move(bert_Constant_310_0), std::move(bert_Divide_311_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_313_0), std::move(bert_Add_308_0), std::move(bert_Multiply_316_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Multiply_316_0), std::move(bert_Sum_317_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_317_0), std::move(bert_Constant_318_0), std::move(bert_Reshape_321_0), std::move(bert_Rsqrt_324_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_86_0), std::move(bert_Reshape_313_0), std::move(bert_Reshape_325_0), std::move(bert_Constant_87_0), std::move(bert_Add_308_0), std::move(bert_Add_335_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_233Kernel(std::move(bert_cublas_handle_0), std::move(bert_Add_335_0), std::move(bert_Constant_91_0), std::move(bert_Dot_336_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Broadcast_Add_Broadcast_Power_Multiply_Add_Multiply_Tanh_Add_Multiply_Multiply_13_CallKernel(dim3(262144, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_97_0), std::move(bert_Constant_96_0), std::move(bert_Constant_95_0), std::move(bert_Constant_94_0), std::move(bert_Constant_92_0), std::move(bert_Dot_336_0), std::move(bert_Constant_93_0), std::move(bert_Multiply_351_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_249Kernel(std::move(bert_cublas_handle_0), std::move(bert_Multiply_351_0), std::move(bert_Constant_98_0), std::move(bert_Dot_352_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_99_0), std::move(bert_Dot_352_0), std::move(bert_Add_335_0), std::move(bert_Add_355_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Add_355_0), std::move(bert_Sum_356_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Divide_float_float_float_cuda_Divide_151_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_356_0), std::move(bert_Constant_357_0), std::move(bert_Divide_358_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Subtract_Multiply_2_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Reshape_360_0), std::move(bert_Add_355_0), std::move(bert_Multiply_363_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Sum_float_float_cuda_Sum_149_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Multiply_363_0), std::move(bert_Sum_364_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_cuda_Divide_Reshape_Broadcast_Add_Rsqrt_3_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Sum_364_0), std::move(bert_Constant_365_0), std::move(bert_Reshape_368_0), std::move(bert_Rsqrt_371_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Multiply_Multiply_Subtract_Multiply_Add_4_CallKernel(dim3(65536, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_100_0), std::move(bert_Reshape_360_0), std::move(bert_Reshape_372_0), std::move(bert_Constant_101_0), std::move(bert_Add_355_0), std::move(bert_Add_382_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Slice_float_float_cuda_Slice_384_CallKernel(dim3(4096, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Reshape_383_0), std::move(bert_Slice_384_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_386Kernel(std::move(bert_cublas_handle_0), std::move(bert_Reshape_385_0), std::move(bert_Constant_109_0), std::move(bert_Dot_386_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_FusedKernel_float_float_float_cuda_Broadcast_Add_Tanh_31_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(bert_Constant_110_0), std::move(bert_Dot_386_0), std::move(bert_Tanh_389_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Dot_float_float_float_cuda_lib_Dot_390Kernel(std::move(bert_cublas_handle_0), std::move(bert_Tanh_389_0), std::move(bert_Constant_111_0), std::move(bert_Dot_390_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Broadcast_float_float_cuda_Broadcast_391_CallKernel(dim3(4004, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(bert_Constant_112_0), std::move(bert_Broadcast_391_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Add_float_float_float_cuda_Add_392_CallKernel(dim3(1001, 1, 1), dim3(256, 1, 1), 0, nullptr, std::move(bert_Dot_390_0), std::move(bert_Broadcast_391_0), std::move(bert_Add_392_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Softmax_float_float_cuda_lib_Softmax_393Kernel(std::move(bert_cudnn_handle_0), std::move(bert_Add_392_0), std::move(bert_Softmax_393_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
    kernels.emplace_back(new bert_Result_float_float_cuda_lib_Result_394Kernel(std::move(bert_Softmax_393_0), std::move(Result_394_0), std::move(Parameter_0_0), std::move(Parameter_1_0), std::move(Parameter_2_0), std::move(Result_394_0)));
}
