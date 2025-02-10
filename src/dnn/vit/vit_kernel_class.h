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
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <fstream>

#include <vector>
#include <cudnn.h>
#include <limits>
#define MIN(a,b) ((a)>(b)?(b):(a))
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
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif
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
   __device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}

// Check compute capability
const int GPU_WARP_SIZE = 32;
const uint64_t MAX_GRID_Y = 65535;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}
__device__ __forceinline__ float subtractf(float x0, float x1)
{
    return x0-x1;
}
cublasHandle_t vit_cublas_handle_0;
cudnnHandle_t vit_cudnn_handle_0;
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
__device__ __forceinline__ half  load(const half*  __restrict__ in, int i=0, bool b=true)
{
    half v = 0.0f;
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

inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/* Modifications Copyright (c) Microsoft. */

// The code below(from the definition of softmax_warp_forward to the definition of dispatch_softmax_forward) 
// is mostly copied from Pytorch PersistentSoftmax.cuh

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "GPU_WARP_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architecures, but there is no guarantee this will not change for future arch.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_log_softmax) {
        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
      } else {
        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    if (is_log_softmax) sum[i] = max_value[i] + std::log((float)(sum[i]));
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(cudaStream_t stream, output_t* dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_forward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_forward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_forward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_forward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_forward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_forward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_forward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_forward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_forward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_forward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_forward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
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

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
int vit_num_SMs;
char* vit_group_0_CUDA_GPU0_allocator_memory_pool;
float* vit_Broadcast_215_0;
float* vit_Broadcast_209_0;
float* vit_Convolution_208_0;
float* vit_Add_210_0;
float* vit_Reshape_211_0;
float* vit_Reshape_212_0;
float* vit_Concat_213_0;
float* vit_Add_216_0;
float* vit_Sum_217_0;
float* vit_Divide_220_0;
float* vit_Reshape_221_0;
float* vit_Reshape_222_0;
float* vit_Power_226_0;
float* vit_Subtract_224_0;
float* vit_Sum_227_0;
float* vit_Sqrt_235_0;
float* vit_Reshape_236_0;
float* vit_Add_242_0;
float* vit_Dot_266_0;
float* vit_Add_268_0;
float* vit_Reshape_269_0;
float* vit_Reshape_270_0;
float* vit_Reshape_272_0;
float* vit_Broadcast_274_0;
float* vit_Dot_243_0;
float* vit_Add_245_0;
float* vit_Reshape_246_0;
float* vit_Reshape_247_0;
float* vit_Multiply_250_0;
float* vit_Reshape_260_0;
float* vit_Broadcast_262_0;
float* vit_Dot_251_0;
float* vit_Add_253_0;
float* vit_Reshape_254_0;
float* vit_Reshape_255_0;
float* vit_Multiply_258_0;
float* vit_Reshape_259_0;
float* vit_Broadcast_261_0;
float* vit_BatchMatMul_263_0;
float* vit_Reshape_264_0;
float* vit_Softmax_265_0;
float* vit_Reshape_271_0;
float* vit_Broadcast_273_0;
float* vit_BatchMatMul_275_0;
float* vit_Reshape_276_0;
float* vit_Reshape_277_0;
float* vit_Reshape_278_0;
float* vit_Dot_279_0;
float* vit_Add_282_0;
float* vit_Sum_283_0;
float* vit_Divide_286_0;
float* vit_Reshape_287_0;
float* vit_Reshape_288_0;
float* vit_Power_292_0;
float* vit_Subtract_290_0;
float* vit_Sum_293_0;
float* vit_Sqrt_301_0;
float* vit_Reshape_302_0;
float* vit_Add_308_0;
float* vit_Dot_309_0;
float* vit_Multiply_319_0;
float* vit_Dot_320_0;
float* vit_Add_323_0;
float* vit_Sum_324_0;
float* vit_Divide_327_0;
float* vit_Reshape_328_0;
float* vit_Reshape_329_0;
float* vit_Power_333_0;
float* vit_Subtract_331_0;
float* vit_Sum_334_0;
float* vit_Sqrt_342_0;
float* vit_Reshape_343_0;
float* vit_Add_349_0;
float* vit_Dot_373_0;
float* vit_Add_375_0;
float* vit_Reshape_376_0;
float* vit_Reshape_377_0;
float* vit_Reshape_379_0;
float* vit_Broadcast_381_0;
float* vit_Dot_350_0;
float* vit_Add_352_0;
float* vit_Reshape_353_0;
float* vit_Reshape_354_0;
float* vit_Multiply_357_0;
float* vit_Reshape_367_0;
float* vit_Broadcast_369_0;
float* vit_Dot_358_0;
float* vit_Add_360_0;
float* vit_Reshape_361_0;
float* vit_Reshape_362_0;
float* vit_Multiply_365_0;
float* vit_Reshape_366_0;
float* vit_Broadcast_368_0;
float* vit_BatchMatMul_370_0;
float* vit_Reshape_371_0;
float* vit_Softmax_372_0;
float* vit_Reshape_378_0;
float* vit_Broadcast_380_0;
float* vit_BatchMatMul_382_0;
float* vit_Reshape_383_0;
float* vit_Reshape_384_0;
float* vit_Reshape_385_0;
float* vit_Dot_386_0;
float* vit_Add_389_0;
float* vit_Sum_390_0;
float* vit_Divide_393_0;
float* vit_Reshape_394_0;
float* vit_Reshape_395_0;
float* vit_Power_399_0;
float* vit_Subtract_397_0;
float* vit_Sum_400_0;
float* vit_Sqrt_408_0;
float* vit_Reshape_409_0;
float* vit_Add_415_0;
float* vit_Dot_416_0;
float* vit_Multiply_426_0;
float* vit_Dot_427_0;
float* vit_Add_430_0;
float* vit_Sum_431_0;
float* vit_Divide_434_0;
float* vit_Reshape_435_0;
float* vit_Reshape_436_0;
float* vit_Power_440_0;
float* vit_Subtract_438_0;
float* vit_Sum_441_0;
float* vit_Sqrt_449_0;
float* vit_Reshape_450_0;
float* vit_Add_456_0;
float* vit_Dot_480_0;
float* vit_Add_482_0;
float* vit_Reshape_483_0;
float* vit_Reshape_484_0;
float* vit_Reshape_486_0;
float* vit_Broadcast_488_0;
float* vit_Dot_457_0;
float* vit_Add_459_0;
float* vit_Reshape_460_0;
float* vit_Reshape_461_0;
float* vit_Multiply_464_0;
float* vit_Reshape_474_0;
float* vit_Broadcast_476_0;
float* vit_Dot_465_0;
float* vit_Add_467_0;
float* vit_Reshape_468_0;
float* vit_Reshape_469_0;
float* vit_Multiply_472_0;
float* vit_Reshape_473_0;
float* vit_Broadcast_475_0;
float* vit_BatchMatMul_477_0;
float* vit_Reshape_478_0;
float* vit_Softmax_479_0;
float* vit_Reshape_485_0;
float* vit_Broadcast_487_0;
float* vit_BatchMatMul_489_0;
float* vit_Reshape_490_0;
float* vit_Reshape_491_0;
float* vit_Reshape_492_0;
float* vit_Dot_493_0;
float* vit_Add_496_0;
float* vit_Sum_497_0;
float* vit_Divide_500_0;
float* vit_Reshape_501_0;
float* vit_Reshape_502_0;
float* vit_Power_506_0;
float* vit_Subtract_504_0;
float* vit_Sum_507_0;
float* vit_Sqrt_515_0;
float* vit_Reshape_516_0;
float* vit_Add_522_0;
float* vit_Dot_523_0;
float* vit_Multiply_533_0;
float* vit_Dot_534_0;
float* vit_Add_537_0;
float* vit_Sum_538_0;
float* vit_Divide_541_0;
float* vit_Reshape_542_0;
float* vit_Reshape_543_0;
float* vit_Power_547_0;
float* vit_Subtract_545_0;
float* vit_Sum_548_0;
float* vit_Sqrt_556_0;
float* vit_Reshape_557_0;
float* vit_Add_563_0;
float* vit_Dot_587_0;
float* vit_Add_589_0;
float* vit_Reshape_590_0;
float* vit_Reshape_591_0;
float* vit_Reshape_593_0;
float* vit_Broadcast_595_0;
float* vit_Dot_564_0;
float* vit_Add_566_0;
float* vit_Reshape_567_0;
float* vit_Reshape_568_0;
float* vit_Multiply_571_0;
float* vit_Reshape_581_0;
float* vit_Broadcast_583_0;
float* vit_Dot_572_0;
float* vit_Add_574_0;
float* vit_Reshape_575_0;
float* vit_Reshape_576_0;
float* vit_Multiply_579_0;
float* vit_Reshape_580_0;
float* vit_Broadcast_582_0;
float* vit_BatchMatMul_584_0;
float* vit_Reshape_585_0;
float* vit_Softmax_586_0;
float* vit_Reshape_592_0;
float* vit_Broadcast_594_0;
float* vit_BatchMatMul_596_0;
float* vit_Reshape_597_0;
float* vit_Reshape_598_0;
float* vit_Reshape_599_0;
float* vit_Dot_600_0;
float* vit_Add_603_0;
float* vit_Sum_604_0;
float* vit_Divide_607_0;
float* vit_Reshape_608_0;
float* vit_Reshape_609_0;
float* vit_Power_613_0;
float* vit_Subtract_611_0;
float* vit_Sum_614_0;
float* vit_Sqrt_622_0;
float* vit_Reshape_623_0;
float* vit_Add_629_0;
float* vit_Dot_630_0;
float* vit_Multiply_640_0;
float* vit_Dot_641_0;
float* vit_Add_644_0;
float* vit_Sum_645_0;
float* vit_Divide_648_0;
float* vit_Reshape_649_0;
float* vit_Reshape_650_0;
float* vit_Power_654_0;
float* vit_Subtract_652_0;
float* vit_Sum_655_0;
float* vit_Sqrt_663_0;
float* vit_Reshape_664_0;
float* vit_Add_670_0;
float* vit_Dot_694_0;
float* vit_Add_696_0;
float* vit_Reshape_697_0;
float* vit_Reshape_698_0;
float* vit_Reshape_700_0;
float* vit_Broadcast_702_0;
float* vit_Dot_671_0;
float* vit_Add_673_0;
float* vit_Reshape_674_0;
float* vit_Reshape_675_0;
float* vit_Multiply_678_0;
float* vit_Reshape_688_0;
float* vit_Broadcast_690_0;
float* vit_Dot_679_0;
float* vit_Add_681_0;
float* vit_Reshape_682_0;
float* vit_Reshape_683_0;
float* vit_Multiply_686_0;
float* vit_Reshape_687_0;
float* vit_Broadcast_689_0;
float* vit_BatchMatMul_691_0;
float* vit_Reshape_692_0;
float* vit_Softmax_693_0;
float* vit_Reshape_699_0;
float* vit_Broadcast_701_0;
float* vit_BatchMatMul_703_0;
float* vit_Reshape_704_0;
float* vit_Reshape_705_0;
float* vit_Reshape_706_0;
float* vit_Dot_707_0;
float* vit_Add_710_0;
float* vit_Sum_711_0;
float* vit_Divide_714_0;
float* vit_Reshape_715_0;
float* vit_Reshape_716_0;
float* vit_Power_720_0;
float* vit_Subtract_718_0;
float* vit_Sum_721_0;
float* vit_Sqrt_729_0;
float* vit_Reshape_730_0;
float* vit_Add_736_0;
float* vit_Dot_737_0;
float* vit_Multiply_747_0;
float* vit_Dot_748_0;
float* vit_Add_751_0;
float* vit_Sum_752_0;
float* vit_Divide_755_0;
float* vit_Reshape_756_0;
float* vit_Reshape_757_0;
float* vit_Power_761_0;
float* vit_Subtract_759_0;
float* vit_Sum_762_0;
float* vit_Sqrt_770_0;
float* vit_Reshape_771_0;
float* vit_Add_777_0;
float* vit_Dot_801_0;
float* vit_Add_803_0;
float* vit_Reshape_804_0;
float* vit_Reshape_805_0;
float* vit_Reshape_807_0;
float* vit_Broadcast_809_0;
float* vit_Dot_778_0;
float* vit_Add_780_0;
float* vit_Reshape_781_0;
float* vit_Reshape_782_0;
float* vit_Multiply_785_0;
float* vit_Reshape_795_0;
float* vit_Broadcast_797_0;
float* vit_Dot_786_0;
float* vit_Add_788_0;
float* vit_Reshape_789_0;
float* vit_Reshape_790_0;
float* vit_Multiply_793_0;
float* vit_Reshape_794_0;
float* vit_Broadcast_796_0;
float* vit_BatchMatMul_798_0;
float* vit_Reshape_799_0;
float* vit_Softmax_800_0;
float* vit_Reshape_806_0;
float* vit_Broadcast_808_0;
float* vit_BatchMatMul_810_0;
float* vit_Reshape_811_0;
float* vit_Reshape_812_0;
float* vit_Reshape_813_0;
float* vit_Dot_814_0;
float* vit_Add_817_0;
float* vit_Sum_818_0;
float* vit_Divide_821_0;
float* vit_Reshape_822_0;
float* vit_Reshape_823_0;
float* vit_Power_827_0;
float* vit_Subtract_825_0;
float* vit_Sum_828_0;
float* vit_Sqrt_836_0;
float* vit_Reshape_837_0;
float* vit_Add_843_0;
float* vit_Dot_844_0;
float* vit_Multiply_854_0;
float* vit_Dot_855_0;
float* vit_Add_858_0;
float* vit_Sum_859_0;
float* vit_Divide_862_0;
float* vit_Reshape_863_0;
float* vit_Reshape_864_0;
float* vit_Power_868_0;
float* vit_Subtract_866_0;
float* vit_Sum_869_0;
float* vit_Sqrt_877_0;
float* vit_Reshape_878_0;
float* vit_Add_884_0;
float* vit_Dot_908_0;
float* vit_Add_910_0;
float* vit_Reshape_911_0;
float* vit_Reshape_912_0;
float* vit_Reshape_914_0;
float* vit_Broadcast_916_0;
float* vit_Dot_885_0;
float* vit_Add_887_0;
float* vit_Reshape_888_0;
float* vit_Reshape_889_0;
float* vit_Multiply_892_0;
float* vit_Reshape_902_0;
float* vit_Broadcast_904_0;
float* vit_Dot_893_0;
float* vit_Add_895_0;
float* vit_Reshape_896_0;
float* vit_Reshape_897_0;
float* vit_Multiply_900_0;
float* vit_Reshape_901_0;
float* vit_Broadcast_903_0;
float* vit_BatchMatMul_905_0;
float* vit_Reshape_906_0;
float* vit_Softmax_907_0;
float* vit_Reshape_913_0;
float* vit_Broadcast_915_0;
float* vit_BatchMatMul_917_0;
float* vit_Reshape_918_0;
float* vit_Reshape_919_0;
float* vit_Reshape_920_0;
float* vit_Dot_921_0;
float* vit_Add_924_0;
float* vit_Sum_925_0;
float* vit_Divide_928_0;
float* vit_Reshape_929_0;
float* vit_Reshape_930_0;
float* vit_Power_934_0;
float* vit_Subtract_932_0;
float* vit_Sum_935_0;
float* vit_Sqrt_943_0;
float* vit_Reshape_944_0;
float* vit_Add_950_0;
float* vit_Dot_951_0;
float* vit_Multiply_961_0;
float* vit_Dot_962_0;
float* vit_Add_965_0;
float* vit_Sum_966_0;
float* vit_Divide_969_0;
float* vit_Reshape_970_0;
float* vit_Reshape_971_0;
float* vit_Power_975_0;
float* vit_Subtract_973_0;
float* vit_Sum_976_0;
float* vit_Sqrt_984_0;
float* vit_Reshape_985_0;
float* vit_Add_991_0;
float* vit_Dot_1015_0;
float* vit_Add_1017_0;
float* vit_Reshape_1018_0;
float* vit_Reshape_1019_0;
float* vit_Reshape_1021_0;
float* vit_Broadcast_1023_0;
float* vit_Dot_992_0;
float* vit_Add_994_0;
float* vit_Reshape_995_0;
float* vit_Reshape_996_0;
float* vit_Multiply_999_0;
float* vit_Reshape_1009_0;
float* vit_Broadcast_1011_0;
float* vit_Dot_1000_0;
float* vit_Add_1002_0;
float* vit_Reshape_1003_0;
float* vit_Reshape_1004_0;
float* vit_Multiply_1007_0;
float* vit_Reshape_1008_0;
float* vit_Broadcast_1010_0;
float* vit_BatchMatMul_1012_0;
float* vit_Reshape_1013_0;
float* vit_Softmax_1014_0;
float* vit_Reshape_1020_0;
float* vit_Broadcast_1022_0;
float* vit_BatchMatMul_1024_0;
float* vit_Reshape_1025_0;
float* vit_Reshape_1026_0;
float* vit_Reshape_1027_0;
float* vit_Dot_1028_0;
float* vit_Add_1031_0;
float* vit_Sum_1032_0;
float* vit_Divide_1035_0;
float* vit_Reshape_1036_0;
float* vit_Reshape_1037_0;
float* vit_Power_1041_0;
float* vit_Subtract_1039_0;
float* vit_Sum_1042_0;
float* vit_Sqrt_1050_0;
float* vit_Reshape_1051_0;
float* vit_Add_1057_0;
float* vit_Dot_1058_0;
float* vit_Multiply_1068_0;
float* vit_Dot_1069_0;
float* vit_Add_1072_0;
float* vit_Sum_1073_0;
float* vit_Divide_1076_0;
float* vit_Reshape_1077_0;
float* vit_Reshape_1078_0;
float* vit_Power_1082_0;
float* vit_Subtract_1080_0;
float* vit_Sum_1083_0;
float* vit_Sqrt_1091_0;
float* vit_Reshape_1092_0;
float* vit_Add_1098_0;
float* vit_Dot_1122_0;
float* vit_Add_1124_0;
float* vit_Reshape_1125_0;
float* vit_Reshape_1126_0;
float* vit_Reshape_1128_0;
float* vit_Broadcast_1130_0;
float* vit_Dot_1099_0;
float* vit_Add_1101_0;
float* vit_Reshape_1102_0;
float* vit_Reshape_1103_0;
float* vit_Multiply_1106_0;
float* vit_Reshape_1116_0;
float* vit_Broadcast_1118_0;
float* vit_Dot_1107_0;
float* vit_Add_1109_0;
float* vit_Reshape_1110_0;
float* vit_Reshape_1111_0;
float* vit_Multiply_1114_0;
float* vit_Reshape_1115_0;
float* vit_Broadcast_1117_0;
float* vit_BatchMatMul_1119_0;
float* vit_Reshape_1120_0;
float* vit_Softmax_1121_0;
float* vit_Reshape_1127_0;
float* vit_Broadcast_1129_0;
float* vit_BatchMatMul_1131_0;
float* vit_Reshape_1132_0;
float* vit_Reshape_1133_0;
float* vit_Reshape_1134_0;
float* vit_Dot_1135_0;
float* vit_Add_1138_0;
float* vit_Sum_1139_0;
float* vit_Divide_1142_0;
float* vit_Reshape_1143_0;
float* vit_Reshape_1144_0;
float* vit_Power_1148_0;
float* vit_Subtract_1146_0;
float* vit_Sum_1149_0;
float* vit_Sqrt_1157_0;
float* vit_Reshape_1158_0;
float* vit_Add_1164_0;
float* vit_Dot_1165_0;
float* vit_Multiply_1175_0;
float* vit_Dot_1176_0;
float* vit_Add_1179_0;
float* vit_Sum_1180_0;
float* vit_Divide_1183_0;
float* vit_Reshape_1184_0;
float* vit_Reshape_1185_0;
float* vit_Power_1189_0;
float* vit_Subtract_1187_0;
float* vit_Sum_1190_0;
float* vit_Sqrt_1198_0;
float* vit_Reshape_1199_0;
float* vit_Add_1205_0;
float* vit_Dot_1229_0;
float* vit_Add_1231_0;
float* vit_Reshape_1232_0;
float* vit_Reshape_1233_0;
float* vit_Reshape_1235_0;
float* vit_Broadcast_1237_0;
float* vit_Dot_1206_0;
float* vit_Add_1208_0;
float* vit_Reshape_1209_0;
float* vit_Reshape_1210_0;
float* vit_Multiply_1213_0;
float* vit_Reshape_1223_0;
float* vit_Broadcast_1225_0;
float* vit_Dot_1214_0;
float* vit_Add_1216_0;
float* vit_Reshape_1217_0;
float* vit_Reshape_1218_0;
float* vit_Multiply_1221_0;
float* vit_Reshape_1222_0;
float* vit_Broadcast_1224_0;
float* vit_BatchMatMul_1226_0;
float* vit_Reshape_1227_0;
float* vit_Softmax_1228_0;
float* vit_Reshape_1234_0;
float* vit_Broadcast_1236_0;
float* vit_BatchMatMul_1238_0;
float* vit_Reshape_1239_0;
float* vit_Reshape_1240_0;
float* vit_Reshape_1241_0;
float* vit_Dot_1242_0;
float* vit_Add_1245_0;
float* vit_Sum_1246_0;
float* vit_Divide_1249_0;
float* vit_Reshape_1250_0;
float* vit_Reshape_1251_0;
float* vit_Power_1255_0;
float* vit_Subtract_1253_0;
float* vit_Sum_1256_0;
float* vit_Sqrt_1264_0;
float* vit_Reshape_1265_0;
float* vit_Add_1271_0;
float* vit_Dot_1272_0;
float* vit_Multiply_1282_0;
float* vit_Dot_1283_0;
float* vit_Add_1286_0;
float* vit_Sum_1287_0;
float* vit_Divide_1290_0;
float* vit_Reshape_1291_0;
float* vit_Reshape_1292_0;
float* vit_Power_1296_0;
float* vit_Subtract_1294_0;
float* vit_Sum_1297_0;
float* vit_Sqrt_1305_0;
float* vit_Reshape_1306_0;
float* vit_Add_1312_0;
float* vit_Dot_1336_0;
float* vit_Add_1338_0;
float* vit_Reshape_1339_0;
float* vit_Reshape_1340_0;
float* vit_Reshape_1342_0;
float* vit_Broadcast_1344_0;
float* vit_Dot_1313_0;
float* vit_Add_1315_0;
float* vit_Reshape_1316_0;
float* vit_Reshape_1317_0;
float* vit_Multiply_1320_0;
float* vit_Reshape_1330_0;
float* vit_Broadcast_1332_0;
float* vit_Dot_1321_0;
float* vit_Add_1323_0;
float* vit_Reshape_1324_0;
float* vit_Reshape_1325_0;
float* vit_Multiply_1328_0;
float* vit_Reshape_1329_0;
float* vit_Broadcast_1331_0;
float* vit_BatchMatMul_1333_0;
float* vit_Reshape_1334_0;
float* vit_Softmax_1335_0;
float* vit_Reshape_1341_0;
float* vit_Broadcast_1343_0;
float* vit_BatchMatMul_1345_0;
float* vit_Reshape_1346_0;
float* vit_Reshape_1347_0;
float* vit_Reshape_1348_0;
float* vit_Dot_1349_0;
float* vit_Add_1352_0;
float* vit_Sum_1353_0;
float* vit_Divide_1356_0;
float* vit_Reshape_1357_0;
float* vit_Reshape_1358_0;
float* vit_Power_1362_0;
float* vit_Subtract_1360_0;
float* vit_Sum_1363_0;
float* vit_Sqrt_1371_0;
float* vit_Reshape_1372_0;
float* vit_Add_1378_0;
float* vit_Dot_1379_0;
float* vit_Multiply_1389_0;
float* vit_Dot_1390_0;
float* vit_Add_1393_0;
float* vit_Sum_1394_0;
float* vit_Divide_1397_0;
float* vit_Reshape_1398_0;
float* vit_Reshape_1399_0;
float* vit_Power_1403_0;
float* vit_Subtract_1401_0;
float* vit_Sum_1404_0;
float* vit_Sqrt_1412_0;
float* vit_Reshape_1413_0;
float* vit_Add_1419_0;
float* vit_Dot_1443_0;
float* vit_Add_1445_0;
float* vit_Reshape_1446_0;
float* vit_Reshape_1447_0;
float* vit_Reshape_1449_0;
float* vit_Broadcast_1451_0;
float* vit_Dot_1420_0;
float* vit_Add_1422_0;
float* vit_Reshape_1423_0;
float* vit_Reshape_1424_0;
float* vit_Multiply_1427_0;
float* vit_Reshape_1437_0;
float* vit_Broadcast_1439_0;
float* vit_Dot_1428_0;
float* vit_Add_1430_0;
float* vit_Reshape_1431_0;
float* vit_Reshape_1432_0;
float* vit_Multiply_1435_0;
float* vit_Reshape_1436_0;
float* vit_Broadcast_1438_0;
float* vit_BatchMatMul_1440_0;
float* vit_Reshape_1441_0;
float* vit_Softmax_1442_0;
float* vit_Reshape_1448_0;
float* vit_Broadcast_1450_0;
float* vit_BatchMatMul_1452_0;
float* vit_Reshape_1453_0;
float* vit_Reshape_1454_0;
float* vit_Reshape_1455_0;
float* vit_Dot_1456_0;
float* vit_Add_1459_0;
float* vit_Sum_1460_0;
float* vit_Divide_1463_0;
float* vit_Reshape_1464_0;
float* vit_Reshape_1465_0;
float* vit_Power_1469_0;
float* vit_Subtract_1467_0;
float* vit_Sum_1470_0;
float* vit_Sqrt_1478_0;
float* vit_Reshape_1479_0;
float* vit_Add_1485_0;
float* vit_Dot_1486_0;
float* vit_Multiply_1496_0;
float* vit_Dot_1497_0;
float* vit_Add_1500_0;
float* vit_Sum_1501_0;
float* vit_Divide_1504_0;
float* vit_Reshape_1505_0;
float* vit_Reshape_1506_0;
float* vit_Power_1510_0;
float* vit_Subtract_1508_0;
float* vit_Sum_1511_0;
float* vit_Sqrt_1519_0;
float* vit_Reshape_1520_0;
char* vit_group_persist_CUDA_GPU0_allocator_memory_pool;
float* vit_Constant_1_0;
float* vit_Reshape_214_0;
float* vit_Constant_3_0;
float* vit_Constant_2_0;
float* vit_Constant_200_0;
float* vit_Constant_129_0;
float* vit_Constant_128_0;
float* vit_Constant_218_0;
float* vit_Constant_206_0;
float* vit_Constant_228_0;
float* vit_Constant_202_0;
float* vit_Reshape_1516_0;
float* vit_Constant_10_0;
float* vit_Constant_11_0;
float* vit_Constant_6_0;
float* vit_Constant_127_0;
float* vit_Constant_5_0;
float* vit_Constant_201_0;
float* vit_Reshape_248_0;
float* vit_Constant_126_0;
float* vit_Constant_4_0;
float* vit_Constant_7_0;
float* vit_Constant_131_0;
float* vit_Constant_130_0;
float* vit_Constant_284_0;
float* vit_Constant_294_0;
float* vit_Constant_12_0;
float* vit_Constant_13_0;
float* vit_Constant_8_0;
float* vit_Constant_205_0;
float* vit_Constant_203_0;
float* vit_Constant_204_0;
float* vit_Constant_9_0;
float* vit_Constant_135_0;
float* vit_Constant_134_0;
float* vit_Constant_325_0;
float* vit_Constant_335_0;
float* vit_Constant_20_0;
float* vit_Constant_21_0;
float* vit_Constant_16_0;
float* vit_Constant_133_0;
float* vit_Constant_15_0;
float* vit_Constant_132_0;
float* vit_Constant_14_0;
float* vit_Constant_17_0;
float* vit_Constant_137_0;
float* vit_Constant_136_0;
float* vit_Constant_391_0;
float* vit_Constant_401_0;
float* vit_Constant_22_0;
float* vit_Constant_23_0;
float* vit_Constant_18_0;
float* vit_Constant_19_0;
float* vit_Constant_141_0;
float* vit_Constant_140_0;
float* vit_Constant_432_0;
float* vit_Constant_442_0;
float* vit_Constant_30_0;
float* vit_Constant_31_0;
float* vit_Constant_26_0;
float* vit_Constant_139_0;
float* vit_Constant_25_0;
float* vit_Constant_138_0;
float* vit_Constant_24_0;
float* vit_Constant_27_0;
float* vit_Constant_143_0;
float* vit_Constant_142_0;
float* vit_Constant_498_0;
float* vit_Constant_508_0;
float* vit_Constant_32_0;
float* vit_Constant_33_0;
float* vit_Constant_28_0;
float* vit_Constant_29_0;
float* vit_Constant_147_0;
float* vit_Constant_146_0;
float* vit_Constant_539_0;
float* vit_Constant_549_0;
float* vit_Constant_40_0;
float* vit_Constant_41_0;
float* vit_Constant_36_0;
float* vit_Constant_145_0;
float* vit_Constant_35_0;
float* vit_Constant_144_0;
float* vit_Constant_34_0;
float* vit_Constant_37_0;
float* vit_Constant_149_0;
float* vit_Constant_148_0;
float* vit_Constant_605_0;
float* vit_Constant_615_0;
float* vit_Constant_42_0;
float* vit_Constant_43_0;
float* vit_Constant_38_0;
float* vit_Constant_39_0;
float* vit_Constant_153_0;
float* vit_Constant_152_0;
float* vit_Constant_646_0;
float* vit_Constant_656_0;
float* vit_Constant_50_0;
float* vit_Constant_51_0;
float* vit_Constant_46_0;
float* vit_Constant_151_0;
float* vit_Constant_45_0;
float* vit_Constant_150_0;
float* vit_Constant_44_0;
float* vit_Constant_47_0;
float* vit_Constant_155_0;
float* vit_Constant_154_0;
float* vit_Constant_712_0;
float* vit_Constant_722_0;
float* vit_Constant_52_0;
float* vit_Constant_53_0;
float* vit_Constant_48_0;
float* vit_Constant_49_0;
float* vit_Constant_159_0;
float* vit_Constant_158_0;
float* vit_Constant_753_0;
float* vit_Constant_763_0;
float* vit_Constant_60_0;
float* vit_Constant_61_0;
float* vit_Constant_56_0;
float* vit_Constant_157_0;
float* vit_Constant_55_0;
float* vit_Constant_156_0;
float* vit_Constant_54_0;
float* vit_Constant_57_0;
float* vit_Constant_161_0;
float* vit_Constant_160_0;
float* vit_Constant_819_0;
float* vit_Constant_829_0;
float* vit_Constant_62_0;
float* vit_Constant_63_0;
float* vit_Constant_58_0;
float* vit_Constant_59_0;
float* vit_Constant_165_0;
float* vit_Constant_164_0;
float* vit_Constant_860_0;
float* vit_Constant_870_0;
float* vit_Constant_70_0;
float* vit_Constant_71_0;
float* vit_Constant_66_0;
float* vit_Constant_163_0;
float* vit_Constant_65_0;
float* vit_Constant_162_0;
float* vit_Constant_64_0;
float* vit_Constant_67_0;
float* vit_Constant_167_0;
float* vit_Constant_166_0;
float* vit_Constant_926_0;
float* vit_Constant_936_0;
float* vit_Constant_72_0;
float* vit_Constant_73_0;
float* vit_Constant_68_0;
float* vit_Constant_69_0;
float* vit_Constant_171_0;
float* vit_Constant_170_0;
float* vit_Constant_967_0;
float* vit_Constant_977_0;
float* vit_Constant_80_0;
float* vit_Constant_81_0;
float* vit_Constant_76_0;
float* vit_Constant_169_0;
float* vit_Constant_75_0;
float* vit_Constant_168_0;
float* vit_Constant_74_0;
float* vit_Constant_77_0;
float* vit_Constant_173_0;
float* vit_Constant_172_0;
float* vit_Constant_1033_0;
float* vit_Constant_1043_0;
float* vit_Constant_82_0;
float* vit_Constant_83_0;
float* vit_Constant_78_0;
float* vit_Constant_79_0;
float* vit_Constant_177_0;
float* vit_Constant_176_0;
float* vit_Constant_1074_0;
float* vit_Constant_1084_0;
float* vit_Constant_90_0;
float* vit_Constant_91_0;
float* vit_Constant_86_0;
float* vit_Constant_175_0;
float* vit_Constant_85_0;
float* vit_Constant_174_0;
float* vit_Constant_84_0;
float* vit_Constant_87_0;
float* vit_Constant_179_0;
float* vit_Constant_178_0;
float* vit_Constant_1140_0;
float* vit_Constant_1150_0;
float* vit_Constant_92_0;
float* vit_Constant_93_0;
float* vit_Constant_88_0;
float* vit_Constant_89_0;
float* vit_Constant_183_0;
float* vit_Constant_182_0;
float* vit_Constant_1181_0;
float* vit_Constant_1191_0;
float* vit_Constant_100_0;
float* vit_Constant_101_0;
float* vit_Constant_96_0;
float* vit_Constant_181_0;
float* vit_Constant_95_0;
float* vit_Constant_180_0;
float* vit_Constant_94_0;
float* vit_Constant_97_0;
float* vit_Constant_185_0;
float* vit_Constant_184_0;
float* vit_Constant_1247_0;
float* vit_Constant_1257_0;
float* vit_Constant_102_0;
float* vit_Constant_103_0;
float* vit_Constant_98_0;
float* vit_Constant_99_0;
float* vit_Constant_189_0;
float* vit_Constant_188_0;
float* vit_Constant_1288_0;
float* vit_Constant_1298_0;
float* vit_Constant_110_0;
float* vit_Constant_111_0;
float* vit_Constant_106_0;
float* vit_Constant_187_0;
float* vit_Constant_105_0;
float* vit_Constant_186_0;
float* vit_Constant_104_0;
float* vit_Constant_107_0;
float* vit_Constant_191_0;
float* vit_Constant_190_0;
float* vit_Constant_1354_0;
float* vit_Constant_1364_0;
float* vit_Constant_112_0;
float* vit_Constant_113_0;
float* vit_Constant_108_0;
float* vit_Constant_109_0;
float* vit_Constant_195_0;
float* vit_Constant_194_0;
float* vit_Constant_1395_0;
float* vit_Constant_1405_0;
float* vit_Constant_120_0;
float* vit_Constant_121_0;
float* vit_Constant_116_0;
float* vit_Constant_193_0;
float* vit_Constant_115_0;
float* vit_Constant_192_0;
float* vit_Constant_114_0;
float* vit_Constant_117_0;
float* vit_Constant_197_0;
float* vit_Constant_196_0;
float* vit_Constant_1461_0;
float* vit_Constant_1471_0;
float* vit_Constant_122_0;
float* vit_Constant_123_0;
float* vit_Constant_118_0;
float* vit_Constant_119_0;
float* vit_Constant_1502_0;
float* vit_Constant_1512_0;
float* vit_Constant_124_0;
float* vit_Constant_125_0;
float* vit_last_hidden_state;
float* vit_Result_1527_0;
int64_t get_workspace_size()
{
    return 725516288;
}
// Node name:	Constant_168
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_168_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_168(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_168_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_168_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_162
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_162_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_162_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_162_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1395
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1395_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1395(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1395_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1395_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_91_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_109_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_109_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Add_210
// Description:	Add
// Input:
//	- name: vit_Convolution_208_0	type: float	shape: Shape{48, 768, 14, 14}
//	- name: vit_Broadcast_209_0	type: float	shape: Shape{48, 768, 14, 14}
// Output:
//	- name: vit_Add_210_0	type: float	shape: Shape{48, 768, 14, 14}
extern "C" __launch_bounds__(512) __global__ void vit_Add_float_float_float_cuda_Add_210(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void vit_Add_float_float_float_cuda_Add_210_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_Add_float_float_float_cuda_Add_210<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_270
// Description:	Reshape
// Input:
//	- name: vit_Reshape_269_0	type: float	shape: Shape{48, 197, 12, 64}
// Output:
//	- name: vit_Reshape_270_0	type: float	shape: Shape{48, 12, 197, 64}
extern "C" __launch_bounds__(64) __global__ void vit_Reshape_float_float_cuda_Reshape_270(float* input0, float* output0)
{
    uint32_t input_strides0 = 151296;
    uint32_t input_strides1 = 768;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 151296;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 12608;
    uint32_t trans_strides3 = 1;
    size_t n = 7262208;
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
extern void vit_Reshape_float_float_cuda_Reshape_270_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_270<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_208
// Description:	Convolution
// Input:
//	- name: Parameter_207_0	type: float	shape: Shape{48, 3, 224, 224}
//	- name: vit_Constant_2_0	type: float	shape: Shape{768, 3, 16, 16}
// Output:
//	- name: vit_Convolution_208_0	type: float	shape: Shape{48, 768, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_208(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 48, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 48, 768, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 768, 3, 16, 16));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 16, 16, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    static bool selected_algo = true;
    static cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    if (!selected_algo) {
        int num_algos;
        int max_algos = 0;
        // cudnnGetConvolutionForwardAlgorithm_v7;
        CUDNN_SAFE_CALL(
            cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
        std::vector<cudnnConvolutionFwdAlgoPerf_t> results(max_algos);
        CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                                tensor_desc_0,
                                                filter_desc,
                                                conv_desc,
                                                tensor_desc_1,
                                                static_cast<int>(results.size()),
                                                &num_algos,
                                                results.data()));
        results.resize(num_algos);
        for (size_t i = 0; i != results.size(); ++i) {
            cudnnConvolutionFwdAlgoPerf_t const& result = results[i];
            if (result.status == CUDNN_STATUS_SUCCESS) {
                conv_fwd_algo = result.algo;
                break;
            }
        }
        selected_algo = true;
    }
    const float alpha = 1.0;
    const float beta = 0.0;
    static void *workspace_ptr_0 = NULL;
    static size_t workspace_size_in_bytes = 0;
    if (!workspace_ptr_0)
    {
        CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, tensor_desc_0, filter_desc, conv_desc, tensor_desc_1, conv_fwd_algo, &workspace_size_in_bytes));
        CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr_0, workspace_size_in_bytes));
    }
    CUDNN_SAFE_CALL(cudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_206_0	type: float	shape: Shape{}
//	- name: vit_Reshape_222_0	type: float	shape: Shape{48, 197}
//	- name: vit_Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Subtract_224_0	type: float	shape: Shape{48, 197, 768}
//	- name: vit_Power_226_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_1528
// Broadcast, Broadcast_223
// Subtract, /encoder/layer.0/layernorm_before/Sub_output_0
// Power, /encoder/layer.0/layernorm_before/Pow_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid / 768];
    float temp2 = subtractf(input2[tid], temp1);
    float temp3 = powf(temp2, temp0);
    output1[tid] = temp3;
    output0[tid] = temp2;

}
extern void vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	BatchMatMul_263
// Description:	BatchMatMul
// Input:
//	- name: vit_Broadcast_261_0	type: float	shape: Shape{48, 12, 197, 64}
//	- name: vit_Broadcast_262_0	type: float	shape: Shape{48, 12, 64, 197}
// Output:
//	- name: vit_BatchMatMul_263_0	type: float	shape: Shape{48, 12, 197, 197}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 197, 197, 64,
                                    &alpha, input1, 197, 12608, input0, 64, 12608,
                                    &beta, output0, 197, 38809, 576));
                            
    }

}
// Node name:	Result_1527
// Description:	Result
// Input:
//	- name: vit_last_hidden_state	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Result_1527_0	type: float	shape: Shape{48, 197, 768}
void Result_float_float_cuda_lib_Result_1527(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Reshape_248_0	type: float	shape: Shape{}
//	- name: vit_Reshape_247_0	type: float	shape: Shape{48, 12, 64, 197}
// Output:
//	- name: vit_Multiply_250_0	type: float	shape: Shape{48, 12, 64, 197}
// Fused functions:
// Broadcast, Broadcast_1530
// Multiply, /encoder/layer.0/attention/attention/Mul_1_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = mul(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Softmax_265
// Description:	Softmax
// Input:
//	- name: vit_Reshape_264_0	type: float	shape: Shape{48, 12, 197, 197}
// Output:
//	- name: vit_Softmax_265_0	type: float	shape: Shape{48, 12, 197, 197}
void Softmax_float_float_cuda_lib_Softmax_265(cudaStream_t stream, float* input0, float* output0)
{

    dispatch_softmax_forward<float, float, float, false>(stream, output0, input0, 197, 197, 113472);
        

}
// Node name:	Dot_320
// Description:	Dot
// Input:
//	- name: vit_Multiply_319_0	type: float	shape: Shape{48, 197, 3072}
//	- name: vit_Constant_131_0	type: float	shape: Shape{3072, 768}
// Output:
//	- name: vit_Dot_320_0	type: float	shape: Shape{48, 197, 768}
void Dot_float_float_float_cuda_lib_Dot_320(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 3072, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 3072, &beta, static_cast<float*>(output0), 768));

}
// Node name:	BatchMatMul_275
// Description:	BatchMatMul
// Input:
//	- name: vit_Broadcast_273_0	type: float	shape: Shape{48, 12, 197, 197}
//	- name: vit_Broadcast_274_0	type: float	shape: Shape{48, 12, 197, 64}
// Output:
//	- name: vit_BatchMatMul_275_0	type: float	shape: Shape{48, 12, 197, 64}
void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 197, 197,
                                    &alpha, input1, 64, 12608, input0, 197, 38809,
                                    &beta, output0, 64, 12608, 576));
                            
    }

}
// Node name:	Reshape_212
// Description:	Reshape
// Input:
//	- name: vit_Reshape_211_0	type: float	shape: Shape{48, 768, 196}
// Output:
//	- name: vit_Reshape_212_0	type: float	shape: Shape{48, 196, 768}
extern "C" __launch_bounds__(256) __global__ void vit_Reshape_float_float_cuda_Reshape_212(float* input0, float* output0)
{
    uint32_t input_strides0 = 150528;
    uint32_t input_strides1 = 196;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 150528;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 768;
    size_t nx = 196;
    size_t ny = 768;
    size_t nz = 48;
    __shared__ float tile[1][16][17];
    uint32_t base2 = blockIdx.x * blockDim.x;
    uint32_t base1 = blockIdx.y * blockDim.y;
    uint32_t base0 = blockIdx.z * blockDim.z;
    uint32_t tid2 = threadIdx.x;
    uint32_t tid1 = threadIdx.y;
    uint32_t tid0 = threadIdx.z;
    uint32_t otid2 = tid2;
    uint32_t otid1 = tid1;
    uint32_t otid0 = tid0;
    uint32_t idx2 = base2 + tid2;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        input_idx += input_strides2* idx2;
        tile[tid0][tid1][tid2] = input0[input_idx];
    }
    otid2 = tid1;
    otid1 = tid2;
    idx2 = base2 + otid2;
    idx1 = base1 + otid1;
    idx0 = base0 + otid0;
    __syncthreads();
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output_idx += trans_strides2* idx2;
        output0[output_idx] = tile[otid0][otid1][otid2];
    }

}
extern void vit_Reshape_float_float_cuda_Reshape_212_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_212<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_7_0	type: float	shape: Shape{768}
//	- name: vit_Dot_279_0	type: float	shape: Shape{48, 197, 768}
//	- name: vit_Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Add_282_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_280
// Add, /encoder/layer.0/attention/output/dense/Add_output_0
// Add, /encoder/layer.0/Add_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 768];
    float temp1 = add(temp0, input1[tid]);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Concat_213
// Description:	Concat
// Input:
//	- name: vit_Constant_200_0	type: float	shape: Shape{48, 1, 768}
//	- name: vit_Reshape_212_0	type: float	shape: Shape{48, 196, 768}
// Output:
//	- name: vit_Concat_213_0	type: float	shape: Shape{48, 197, 768}
extern "C" __launch_bounds__(512) __global__ void vit_Concat_float_float_float_cuda_Concat_213(float* input0, float* input1, float* output0)
{
    uint32_t inputs_strides[] = {768, 150528};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 7262208)
    {
        uint32_t block_id = tid / 151296;
        uint32_t block_idx = tid % 151296;
        uint32_t output_idx = block_id * 151296 + block_idx;
        if(block_idx < inputs_strides[0])
        {
            output0[output_idx] = input0[block_id * inputs_strides[0] + block_idx];
            return;
        }
        block_idx -= inputs_strides[0];
        if(block_idx < inputs_strides[1])
        {
            output0[output_idx] = input1[block_id * inputs_strides[1] + block_idx];
            return;
        }
        block_idx -= inputs_strides[1];
    }

}
extern void vit_Concat_float_float_float_cuda_Concat_213_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_Concat_float_float_float_cuda_Concat_213<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Broadcast_215
// Description:	Broadcast
// Input:
//	- name: vit_Reshape_214_0	type: float	shape: Shape{197, 768}
// Output:
//	- name: vit_Broadcast_215_0	type: float	shape: Shape{48, 197, 768}
extern "C" __launch_bounds__(64) __global__ void vit_Broadcast_float_float_cuda_Broadcast_215(float* input0, float* output0)
{
    size_t nthreads = 7262208;
    uint32_t strides0 = 151296;
    uint32_t strides1 = 768;
    uint32_t strides2 = 1;
    int stride_magic0 = -574115763;
    int stride_magic1 = 715827883;
    int stride_magic2 = 1;
    int stride_shift0 = 17;
    int stride_shift1 = 7;
    int stride_shift2 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 768;
    uint32_t reduced_strides2 = 1;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        int coordinate2 = division_by_invariant_multiplication(coordinate_product, stride_magic2, stride_shift2);
        coordinate_product -= (coordinate2 * strides2);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void vit_Broadcast_float_float_cuda_Broadcast_215_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Broadcast_float_float_cuda_Broadcast_215<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_218_0	type: float	shape: Shape{}
//	- name: vit_Sum_217_0	type: float	shape: Shape{48, 197}
// Output:
//	- name: vit_Divide_220_0	type: float	shape: Shape{48, 197}
// Fused functions:
// Broadcast, Broadcast_219
// Divide, Divide_220
extern "C" __launch_bounds__(394) __global__ void vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 394 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = fdividef(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Dot_309
// Description:	Dot
// Input:
//	- name: vit_Add_308_0	type: float	shape: Shape{48, 197, 768}
//	- name: vit_Constant_130_0	type: float	shape: Shape{768, 3072}
// Output:
//	- name: vit_Dot_309_0	type: float	shape: Shape{48, 197, 3072}
void Dot_float_float_float_cuda_lib_Dot_309(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 3072, 9456, 768, &alpha, static_cast<const float*>(input1), 3072, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 3072));

}
// Node name:	Reshape_277
// Description:	Reshape
// Input:
//	- name: vit_Reshape_276_0	type: float	shape: Shape{48, 12, 197, 64}
// Output:
//	- name: vit_Reshape_277_0	type: float	shape: Shape{48, 197, 12, 64}
extern "C" __launch_bounds__(64) __global__ void vit_Reshape_float_float_cuda_Reshape_277(float* input0, float* output0)
{
    uint32_t input_strides0 = 151296;
    uint32_t input_strides1 = 12608;
    uint32_t input_strides2 = 64;
    uint32_t input_strides3 = 1;
    uint32_t trans_strides0 = 151296;
    uint32_t trans_strides1 = 64;
    uint32_t trans_strides2 = 768;
    uint32_t trans_strides3 = 1;
    size_t n = 7262208;
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
extern void vit_Reshape_float_float_cuda_Reshape_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_277<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_266
// Description:	Dot
// Input:
//	- name: vit_Add_242_0	type: float	shape: Shape{48, 197, 768}
//	- name: vit_Constant_128_0	type: float	shape: Shape{768, 768}
// Output:
//	- name: vit_Dot_266_0	type: float	shape: Shape{48, 197, 768}
void Dot_float_float_float_cuda_lib_Dot_266(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 768, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 768));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Reshape_1516_0	type: float	shape: Shape{1}
//	- name: vit_Constant_228_0	type: float	shape: Shape{}
//	- name: vit_Sum_227_0	type: float	shape: Shape{48, 197}
// Output:
//	- name: vit_Sqrt_235_0	type: float	shape: Shape{48, 197, 1}
// Fused functions:
// Broadcast, Broadcast_1529
// Broadcast, Broadcast_229
// Divide, Divide_230
// Reshape, /encoder/layer.0/layernorm_before/ReduceMean_1_output_0
// Add, /encoder/layer.0/layernorm_before/Add_output_0
// Sqrt, /encoder/layer.0/layernorm_before/Sqrt_output_0
extern "C" __launch_bounds__(394) __global__ void vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 394 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid % 1];
    float temp2 = fdividef(input2[tid], temp1);
    float temp3 = add(temp2, temp0);
    float temp4 = sqrtf(temp3);
    output0[tid] = temp4;

}
extern void vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_11_0	type: float	shape: Shape{768}
//	- name: vit_Constant_10_0	type: float	shape: Shape{768}
//	- name: vit_Reshape_236_0	type: float	shape: Shape{48, 197}
//	- name: vit_Subtract_224_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Add_242_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_241
// Broadcast, Broadcast_239
// Broadcast, Broadcast_237
// Divide, /encoder/layer.0/layernorm_before/Div_output_0
// Multiply, /encoder/layer.0/layernorm_before/Mul_output_0
// Add, /encoder/layer.0/layernorm_before/Add_1_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3(float* input0, float* input1, float* input2, float* input3, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 768];
    float temp1 = input1[tid % 768];
    float temp2 = input2[tid / 768];
    float temp3 = fdividef(input3[tid], temp2);
    float temp4 = mul(temp3, temp1);
    float temp5 = add(temp4, temp0);
    output0[tid] = temp5;

}
extern void vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_117_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_117(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_117_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_117_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_148
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_148_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_148_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_160
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_160_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_160(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_160_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_160_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_54_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_142
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_142_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_142(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_142_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_142_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_92_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_92_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_56
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_56_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_56_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_753
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_753_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_753(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_753_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_753_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_131
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_131_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_131(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_131_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_131_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_2_0	type: float	shape: Shape{768, 3, 16, 16}
void vit_Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_183
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_183_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_183(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_183_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_183_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_49
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_49_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_49_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_48_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_48_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_100_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_100_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_84_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_829
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_829_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_829(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_829_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_829_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_99
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_99_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_99_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_157
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_157_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_157(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_157_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_157_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_870
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_870_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_870(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_870_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_870_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_52_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_156
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_156_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_156_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_284
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_284_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_284(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_284_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_284_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_127
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_127_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_127(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_127_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_127_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_712
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_712_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_712(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_712_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_712_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_102
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_102_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_102(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_102_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_102_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_145
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_145_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_145(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_145_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_145_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_154
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_154_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_154(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_154_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_154_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_155
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_155_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_155(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_155_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_155_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_125
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_125_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_125_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_140
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_140_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_140_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_158
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_158_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_120
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_120_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_150
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_150_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_26_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_26_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_80
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_80_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_80(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_80_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_80_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_860
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_860_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_860(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_860_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_860_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_46_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_187
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_187_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_187(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_187_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_187_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_646
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_646_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_646(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_646_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_646_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_152
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_152_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_152(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_152_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_152_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_176
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_176_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_176_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1074
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1074_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1074(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1074_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1074_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_153
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_153_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_153(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_153_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_153_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_170
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_170_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_86
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_86_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_86_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_111
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_111_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_111_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_192
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_192_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_192(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_192_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_192_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_42_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_15_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_605
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_605_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_605(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_605_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_605_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_90
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_90_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_72_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_72_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_206
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_206_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_206(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_206_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_206_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_149
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_149_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_149_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_164
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_164_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_205
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_205_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_205(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_205_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_205_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_61_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_173
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_173_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_173(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_173_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_173_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_144_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_144_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_88_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_88(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_88_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_66
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_66_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_66_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_41
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_41_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_41(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_41_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_41_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_53_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_97
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_97_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_97_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_40_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_83
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_83_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_83(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_83_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_83_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_549
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_549_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_549(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_549_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_549_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_29_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_101
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_101_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_101_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_181
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_181_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_181_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_47_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_33_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_33_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_138
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_138_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_138(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_138_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_138_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_27_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_6_0	type: float	shape: Shape{768}
//	- name: vit_Dot_266_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Add_268_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_267
// Add, /encoder/layer.0/attention/attention/value/Add_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 768];
    float temp1 = add(temp0, input1[tid]);
    output0[tid] = temp1;

}
extern void vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Constant_977
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_977_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_977(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_977_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_977_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_143
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_143_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_143_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_39_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_6_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_3_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_62
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_62_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_62_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_65
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_65_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_65_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_204
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_204_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_204(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_204_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_204_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_24_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_51_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_12_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_203
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_203_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_203(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_203_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_203_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_151
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_151_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_151(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_151_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_151_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_137
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_137_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_38_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_147
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_147_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_147(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_147_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_147_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_4_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_130
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_130_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_130(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_130_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_130_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_126_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_172
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_172_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_172(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_172_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_172_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_146
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_146_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_82
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_82_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_82(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_82_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_82_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_135
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_135_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_135(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_135_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_135_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_34
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_34_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_34(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_34_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_74
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_74_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_74(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_74_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_74_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_508
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_508_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_508(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_508_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_508_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1140
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1140_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1140_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_819
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_819_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_819(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_819_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_819_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_391
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_391_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_391(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_391_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_391_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_180
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_180_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_180(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_180_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_180_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_188
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_188_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_188_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Sum_217
// Description:	Sum
// Input:
//	- name: vit_Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: vit_Sum_217_0	type: float	shape: Shape{48, 197}
extern "C" __launch_bounds__(512) __global__ void vit_Sum_float_float_cuda_Sum_217(float* input0, float* output0)
{

    int width = 768;
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
extern void vit_Sum_float_float_cuda_Sum_217_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Sum_float_float_cuda_Sum_217<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_325
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_325_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_325(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_325_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_325_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_23_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_57_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_73
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_73_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_73_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_201
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_201_0	type: float	shape: Shape{1}
void vit_Constant_float_cuda_Constant_201(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_201_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_201_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_178
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_178_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_178(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_178_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_178_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_43_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_22
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_22_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_22_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_161
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_161_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_161_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_191
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_191_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_936
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_936_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_936(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_936_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_936_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_194
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_194_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_194_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_5_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_159
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_159_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_159(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_159_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_159_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_129
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_129_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_129(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_129_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_129_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Broadcast_209
// Description:	Broadcast
// Input:
//	- name: vit_Constant_3_0	type: float	shape: Shape{768}
// Output:
//	- name: vit_Broadcast_209_0	type: float	shape: Shape{48, 768, 14, 14}
extern "C" __launch_bounds__(64) __global__ void vit_Broadcast_float_float_cuda_Broadcast_209(float* input0, float* output0)
{
    size_t nthreads = 7225344;
    uint32_t strides0 = 150528;
    uint32_t strides1 = 196;
    uint32_t strides2 = 14;
    uint32_t strides3 = 1;
    int stride_magic0 = 1869917735;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 16;
    int stride_shift1 = 6;
    int stride_shift2 = 3;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
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
        coordinate_product -= (coordinate3 * strides3);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        reduced_idx += coordinate3 * reduced_strides3;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void vit_Broadcast_float_float_cuda_Broadcast_209_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Broadcast_float_float_cuda_Broadcast_209<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_722
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_722_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_722(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_722_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_722_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_128
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_128_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_128_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_442
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_442_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_442(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_442_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_442_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_76_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_195
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_195_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_195(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_195_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_195_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_218
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_218_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_218_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_11_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_68_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_68_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1_0	type: float	shape: Shape{1, 197, 768}
void vit_Constant_float_cuda_Constant_1(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[605184];
    bin_file.read(tmp_mem, 605184);
    cudaMemcpyAsync(output0, tmp_mem, 605184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_202
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_202_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_202(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_202_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_202_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_58_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_8_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_200
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_200_0	type: float	shape: Shape{48, 1, 768}
void vit_Constant_float_cuda_Constant_200(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_200_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_200_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_139
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_139_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_139(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_139_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_139_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_134
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_134_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_134(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_134_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_134_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_335
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_335_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_335(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_335_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_335_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_67_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_67_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_228
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_228_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_228(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_228_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_228_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_118
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_118_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_118(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_118_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_118_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_30_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_9_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1043
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1043_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1043(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1043_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1043_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_96_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_96_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_28_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_13
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_13_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_13_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_10
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_10_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_89
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_89_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_89(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_89_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_89_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_36_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_36_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_186
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_186_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_186(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_186_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_186_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_69
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_69_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_69_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_169_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_169_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_17
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_17_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_75
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_75_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_75(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_75_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_75_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1033
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1033_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1033(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1033_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1033_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_79
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_79_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_79_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_174
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_174_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_174_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_108
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_108_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_108_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1084
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1084_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1084(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1084_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1084_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_196
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_196_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_196(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_196_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_196_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_35_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_105
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_105_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_105(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_105_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_105_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_175
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_175_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_175(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_175_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_175_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_112_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_93_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_50
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_50_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_50_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_94
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_94_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Reshape_247
// Description:	Reshape
// Input:
//	- name: vit_Reshape_246_0	type: float	shape: Shape{48, 197, 12, 64}
// Output:
//	- name: vit_Reshape_247_0	type: float	shape: Shape{48, 12, 64, 197}
extern "C" __launch_bounds__(256) __global__ void vit_Reshape_float_float_cuda_Reshape_247(float* input0, float* output0)
{
    uint32_t input_strides0 = 151296;
    uint32_t input_strides1 = 768;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 151296;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 197;
    size_t nx = 768;
    size_t ny = 197;
    size_t nz = 48;
    __shared__ float tile[1][16][17];
    uint32_t base2 = blockIdx.x * blockDim.x;
    uint32_t base1 = blockIdx.y * blockDim.y;
    uint32_t base0 = blockIdx.z * blockDim.z;
    uint32_t tid2 = threadIdx.x;
    uint32_t tid1 = threadIdx.y;
    uint32_t tid0 = threadIdx.z;
    uint32_t otid2 = tid2;
    uint32_t otid1 = tid1;
    uint32_t otid0 = tid0;
    uint32_t idx2 = base2 + tid2;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        input_idx += input_strides2* idx2;
        tile[tid0][tid1][tid2] = input0[input_idx];
    }
    otid2 = tid1;
    otid1 = tid2;
    idx2 = base2 + otid2;
    idx1 = base1 + otid1;
    idx0 = base0 + otid0;
    __syncthreads();
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output_idx += trans_strides2* idx2;
        output0[output_idx] = tile[otid0][otid1][otid2];
    }

}
extern void vit_Reshape_float_float_cuda_Reshape_247_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_247<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_193_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_193(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_193_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_193_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1181
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1181_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1181_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_185
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_185_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_185_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1257
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1257_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1257(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1257_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1257_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_124
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_124_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_124(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_124_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_124_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_182
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_182_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_182_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1471
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1471_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1471(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1471_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1471_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1461
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1461_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1461(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1461_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1461_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_184
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_184_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_184(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_184_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_184_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1502
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1502_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1502(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1502_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1502_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_81
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_81_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_81(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_81_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_81_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_31_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1150
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1150_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_121
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_121_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_121(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_121_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_121_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_498
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_498_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_498(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_498_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_498_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vit_Constant_204_0	type: float	shape: Shape{}
//	- name: vit_Constant_203_0	type: float	shape: Shape{}
//	- name: vit_Constant_205_0	type: float	shape: Shape{}
//	- name: vit_Constant_8_0	type: float	shape: Shape{3072}
//	- name: vit_Dot_309_0	type: float	shape: Shape{48, 197, 3072}
// Output:
//	- name: vit_Multiply_319_0	type: float	shape: Shape{48, 197, 3072}
// Fused functions:
// Broadcast, Broadcast_1536
// Broadcast, Broadcast_1535
// Broadcast, Broadcast_1534
// Broadcast, Broadcast_310
// Add, /encoder/layer.0/intermediate/dense/Add_output_0
// Divide, /encoder/layer.0/intermediate/intermediate_act_fn/Div_output_0
// Erf, /encoder/layer.0/intermediate/intermediate_act_fn/Erf_output_0
// Add, /encoder/layer.0/intermediate/intermediate_act_fn/Add_output_0
// Multiply, /encoder/layer.0/intermediate/intermediate_act_fn/Mul_output_0
// Multiply, /encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0
extern "C" __launch_bounds__(512) __global__ void vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid % 1];
    float temp2 = input2[tid % 1];
    float temp3 = input3[tid % 3072];
    float temp4 = add(temp3, input4[tid]);
    float temp5 = fdividef(temp4, temp2);
    float temp6 = erff(temp5);
    float temp7 = add(temp6, temp1);
    float temp8 = mul(temp4, temp7);
    float temp9 = mul(temp8, temp0);
    output0[tid] = temp9;

}
extern void vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Constant_163
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_163_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_163(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_163_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_163_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_432
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_432_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_432(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_432_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_432_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_197
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_197_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_197_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_16
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_16_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_16_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_113
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_113_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_113(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_113_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_113_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_136
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_136_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_136(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_136_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_136_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_615
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_615_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_615(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_615_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_615_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1298
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1298_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1298(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1298_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1298_0 failed.\n");
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
//	- name: vit_Constant_95_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_95_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1288
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1288_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1288(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1288_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1288_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_189
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_189_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_189(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_189_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_189_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1512
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1512_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1512(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1512_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1512_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1364
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1364_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1364(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1364_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1364_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1405
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1405_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1405(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1405_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1405_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_171
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_171_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_171_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_123
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_123_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_123(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_123_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_123_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_190
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_190_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_190_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_116
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_116_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_165
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_165_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_165(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_165_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_165_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_107
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_107_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_107_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_77
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_77_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_77(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_77_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_77_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_85
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_85_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_85_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_104_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_763
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_763_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_763(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_763_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_763_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_87_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_87_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_114
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_114_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_114_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_106
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_106_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_106(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_106_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_106_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_926
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_926_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_926(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_926_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_926_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_45_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_167_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_167_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_179
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_179_0	type: float	shape: Shape{3072, 768}
void vit_Constant_float_cuda_Constant_179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_179_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_119
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_119_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_119_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_64
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_64_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_64(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_64_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_64_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_103
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_103_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_103(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_103_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_103_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_71
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_71_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_71_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_59_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_59_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_63
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_63_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_63(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_63_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_63_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_60
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_60_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_60_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_539
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_539_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_539(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_539_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_539_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_21_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_21_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1247
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1247_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1247(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1247_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1247_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_166
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_166_0	type: float	shape: Shape{768, 3072}
void vit_Constant_float_cuda_Constant_166(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_166_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_166_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1354
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1354_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1354(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1354_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1354_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1191
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_1191_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_1191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_1191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_1191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_133
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_133_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_133(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_133_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_133_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_141
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_141_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_141(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_141_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_141_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_19_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_294
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_294_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_294(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_294_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_294_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_70
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_70_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_70_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_18_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_656
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_656_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_656(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_656_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_656_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_401
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_401_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_401(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_401_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_401_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_110
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_110_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_110_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_177
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_177_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_177_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_20
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_20_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_20_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_132
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_132_0	type: float	shape: Shape{768, 768}
void vit_Constant_float_cuda_Constant_132(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_132_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_132_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_37_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_44_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_122
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_122_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_122(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_122_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_122_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_78_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_78_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_967
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_967_0	type: float	shape: Shape{}
void vit_Constant_float_cuda_Constant_967(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_967_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_967_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_25_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_14_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_32_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_7_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN
int get_device_type()
{
    return 0;
}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_55_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_55_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_98
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_98_0	type: float	shape: Shape{3072}
void vit_Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_98_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {48, 3, 224, 224}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {48, 197, 768}
#endif

// Node name:	Constant_115
// Description:	Constant
// Input:
// Output:
//	- name: vit_Constant_115_0	type: float	shape: Shape{768}
void vit_Constant_float_cuda_Constant_115(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vit/Constant/Constant_115_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vit_Constant_115_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}

extern "C" void vit_cuda_init()
{
// // total memory:725516288

CUDA_SAFE_CALL(cudaMalloc((void**)&vit_group_0_CUDA_GPU0_allocator_memory_pool,353124864));
CUDA_SAFE_CALL(cudaMemset((void*)vit_group_0_CUDA_GPU0_allocator_memory_pool, 0, 353124864));
vit_Broadcast_215_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_209_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Convolution_208_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+57950208);
vit_Add_210_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+57950208);
vit_Reshape_211_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+57950208);
vit_Reshape_212_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Concat_213_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+57950208);
vit_Add_216_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+57950208);
vit_Sum_217_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_220_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_221_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_222_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_226_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_224_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Sum_227_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_235_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_236_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_242_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_266_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Add_268_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Reshape_269_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Reshape_270_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Reshape_272_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Broadcast_274_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Dot_243_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Add_245_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145096704);
vit_Reshape_246_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145096704);
vit_Reshape_247_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Multiply_250_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145096704);
vit_Reshape_260_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145096704);
vit_Broadcast_262_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145096704);
vit_Dot_251_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Add_253_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_254_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_255_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Multiply_258_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_259_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_261_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_263_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174145536);
vit_Reshape_264_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174145536);
vit_Softmax_265_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263561472);
vit_Reshape_271_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263561472);
vit_Broadcast_273_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263561472);
vit_BatchMatMul_275_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_276_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_277_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Reshape_278_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Dot_279_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_282_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+86999040);
vit_Sum_283_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_286_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_287_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_288_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_292_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_290_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29124480);
vit_Sum_293_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_301_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_302_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_308_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_309_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116047872);
vit_Multiply_319_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+232243200);
vit_Dot_320_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_323_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_324_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_327_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_328_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_329_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_333_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_331_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_334_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_342_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_343_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_349_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_373_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_375_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_376_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_377_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_379_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_381_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Dot_350_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_352_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_353_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_354_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_357_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_367_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Broadcast_369_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Dot_358_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_360_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Reshape_361_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Reshape_362_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_365_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Reshape_366_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Broadcast_368_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_BatchMatMul_370_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_371_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Softmax_372_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+234660096);
vit_Reshape_378_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+234660096);
vit_Broadcast_380_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+234660096);
vit_BatchMatMul_382_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Reshape_383_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Reshape_384_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_385_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Dot_386_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Add_389_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sum_390_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Divide_393_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29086656);
vit_Reshape_394_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29086656);
vit_Reshape_395_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29086656);
vit_Power_399_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29124480);
vit_Subtract_397_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58173312);
vit_Sum_400_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sqrt_408_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29086656);
vit_Reshape_409_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29086656);
vit_Add_415_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29124480);
vit_Dot_416_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58173312);
vit_Multiply_426_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174368640);
vit_Dot_427_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Add_430_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_431_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_434_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_435_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_436_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_440_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_438_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_441_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_449_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_450_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_456_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_480_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_482_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_483_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_484_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_486_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_488_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_457_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_459_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_460_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_461_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_464_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_474_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_476_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_465_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_467_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_468_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_469_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_472_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_473_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_475_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_477_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_478_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_479_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_485_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_487_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_489_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_490_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_491_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_492_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_493_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_496_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_497_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_500_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_501_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_502_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_506_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_504_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_507_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_515_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_516_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_522_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_523_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_533_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_534_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_537_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_538_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_541_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_542_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_543_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_547_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_545_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_548_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_556_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_557_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_563_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_587_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_589_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_590_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_591_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_593_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_595_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_564_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_566_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_567_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_568_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_571_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_581_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_583_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_572_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_574_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_575_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_576_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_579_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_580_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_582_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_584_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_585_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_586_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_592_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_594_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_596_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_597_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_598_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_599_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_600_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_603_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_604_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_607_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_608_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_609_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_613_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_611_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_614_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_622_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_623_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_629_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_630_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_640_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_641_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_644_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_645_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_648_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_649_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_650_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_654_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_652_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_655_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_663_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_664_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_670_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_694_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_696_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_697_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_698_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_700_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_702_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_671_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_673_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_674_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_675_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_678_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_688_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_690_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_679_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_681_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_682_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_683_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_686_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_687_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_689_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_691_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_692_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_693_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_699_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_701_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_703_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_704_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_705_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_706_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_707_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_710_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_711_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_714_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_715_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_716_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_720_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_718_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_721_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_729_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_730_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_736_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_737_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_747_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_748_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_751_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_752_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_755_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_756_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_757_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_761_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_759_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_762_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_770_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_771_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_777_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_801_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_803_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_804_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_805_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_807_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_809_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_778_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_780_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_781_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_782_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_785_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_795_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_797_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_786_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_788_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_789_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_790_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_793_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_794_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_796_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_798_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_799_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_800_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_806_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_808_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_810_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_811_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_812_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_813_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_814_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_817_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_818_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_821_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_822_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_823_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_827_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_825_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_828_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_836_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_837_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_843_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_844_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_854_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_855_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_858_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_859_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_862_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_863_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_864_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_868_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_866_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_869_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_877_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_878_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_884_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_908_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_910_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_911_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_912_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_914_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_916_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_885_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_887_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_888_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_889_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_892_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_902_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_904_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_893_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_895_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_896_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_897_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_900_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_901_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_903_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_905_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_906_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_907_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_913_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_915_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_917_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_918_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_919_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_920_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_921_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_924_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_925_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_928_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_929_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_930_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_934_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_932_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_935_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_943_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_944_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_950_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_951_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_961_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_962_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_965_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_966_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_969_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_970_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_971_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_975_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_973_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_976_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_984_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_985_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_991_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_1015_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_1017_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1018_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1019_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_1021_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_1023_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_992_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_994_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_995_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_996_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_999_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1009_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_1011_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_1000_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1002_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1003_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1004_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_1007_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1008_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_1010_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_1012_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_1013_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_1014_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_1020_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_1022_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_1024_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1025_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1026_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_1027_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_1028_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1031_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_1032_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1035_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1036_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1037_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1041_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_1039_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1042_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1050_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1051_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1057_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_1058_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_1068_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_1069_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1072_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_1073_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1076_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1077_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1078_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1082_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_1080_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1083_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1091_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1092_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1098_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_1122_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_1124_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1125_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1126_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_1128_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_1130_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_1099_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1101_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1102_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1103_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_1106_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1116_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_1118_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_1107_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1109_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1110_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1111_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_1114_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1115_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_1117_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_1119_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_1120_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_1121_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_1127_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_1129_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_1131_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1132_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1133_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_1134_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_1135_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1138_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_1139_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1142_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1143_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1144_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1148_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_1146_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1149_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1157_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1158_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1164_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_1165_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_1175_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_1176_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1179_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_1180_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1183_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1184_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1185_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1189_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_1187_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1190_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1198_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1199_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1205_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_1229_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_1231_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1232_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1233_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_1235_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_1237_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_1206_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1208_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1209_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1210_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_1213_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1223_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_1225_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_1214_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1216_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1217_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1218_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_1221_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1222_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_1224_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_1226_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_1227_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_1228_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_1234_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_1236_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_1238_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1239_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1240_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_1241_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_1242_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1245_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_1246_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1249_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1250_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1251_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1255_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_1253_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1256_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1264_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1265_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1271_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_1272_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_1282_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_1283_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1286_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_1287_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1290_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1291_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1292_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1296_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_1294_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1297_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1305_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1306_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1312_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_1336_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_1338_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1339_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1340_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_1342_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_1344_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_1313_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1315_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1316_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1317_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_1320_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1330_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_1332_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_1321_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1323_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1324_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1325_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_1328_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1329_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_1331_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_1333_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_1334_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_1335_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_1341_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_1343_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_1345_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1346_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1347_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_1348_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_1349_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1352_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_1353_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1356_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1357_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1358_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1362_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_1360_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1363_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1371_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1372_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1378_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_1379_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_1389_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_1390_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1393_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_1394_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1397_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1398_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1399_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1403_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_1401_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1404_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1412_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1413_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1419_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Dot_1443_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Add_1445_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1446_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Reshape_1447_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Reshape_1449_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Broadcast_1451_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Dot_1420_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1422_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1423_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1424_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Multiply_1427_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Reshape_1437_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Broadcast_1439_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+145244160);
vit_Dot_1428_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+116195328);
vit_Add_1430_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1431_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1432_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Multiply_1435_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1436_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Broadcast_1438_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_BatchMatMul_1440_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Reshape_1441_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+174292992);
vit_Softmax_1442_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Reshape_1448_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_Broadcast_1450_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+263708928);
vit_BatchMatMul_1452_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1453_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_1454_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Reshape_1455_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Dot_1456_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1459_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+29048832);
vit_Sum_1460_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1463_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1464_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1465_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1469_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Subtract_1467_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1470_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1478_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1479_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Add_1485_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Dot_1486_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Multiply_1496_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+203341824);
vit_Dot_1497_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Add_1500_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+58097664);
vit_Sum_1501_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Divide_1504_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1505_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1506_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Power_1510_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+75648);
vit_Subtract_1508_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+87146496);
vit_Sum_1511_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+0);
vit_Sqrt_1519_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);
vit_Reshape_1520_0 = (float*)(vit_group_0_CUDA_GPU0_allocator_memory_pool+37824);

CUDA_SAFE_CALL(cudaMalloc((void**)&vit_group_persist_CUDA_GPU0_allocator_memory_pool,372391424));
CUDA_SAFE_CALL(cudaMemset((void*)vit_group_persist_CUDA_GPU0_allocator_memory_pool, 0, 372391424));
vit_Constant_1_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+0);
vit_Reshape_214_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+0);
vit_Constant_3_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+605184);
vit_Constant_2_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+608256);
vit_Constant_200_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+2967552);
vit_Constant_129_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+3115008);
vit_Constant_128_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+5474304);
vit_Constant_218_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833600);
vit_Constant_206_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833664);
vit_Constant_228_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833728);
vit_Constant_202_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833792);
vit_Reshape_1516_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833792);
vit_Constant_10_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7833856);
vit_Constant_11_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7836928);
vit_Constant_6_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7840000);
vit_Constant_127_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+7843072);
vit_Constant_5_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+10202368);
vit_Constant_201_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+10205440);
vit_Reshape_248_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+10205440);
vit_Constant_126_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+10205504);
vit_Constant_4_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+12564800);
vit_Constant_7_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+12567872);
vit_Constant_131_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+12570944);
vit_Constant_130_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+22008128);
vit_Constant_284_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31445312);
vit_Constant_294_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31445376);
vit_Constant_12_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31445440);
vit_Constant_13_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31448512);
vit_Constant_8_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31451584);
vit_Constant_205_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31463872);
vit_Constant_203_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31463936);
vit_Constant_204_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31464000);
vit_Constant_9_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31464064);
vit_Constant_135_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+31467136);
vit_Constant_134_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+33826432);
vit_Constant_325_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36185728);
vit_Constant_335_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36185792);
vit_Constant_20_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36185856);
vit_Constant_21_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36188928);
vit_Constant_16_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36192000);
vit_Constant_133_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+36195072);
vit_Constant_15_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+38554368);
vit_Constant_132_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+38557440);
vit_Constant_14_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+40916736);
vit_Constant_17_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+40919808);
vit_Constant_137_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+40922880);
vit_Constant_136_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+50360064);
vit_Constant_391_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59797248);
vit_Constant_401_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59797312);
vit_Constant_22_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59797376);
vit_Constant_23_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59800448);
vit_Constant_18_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59803520);
vit_Constant_19_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59815808);
vit_Constant_141_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+59818880);
vit_Constant_140_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+62178176);
vit_Constant_432_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64537472);
vit_Constant_442_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64537536);
vit_Constant_30_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64537600);
vit_Constant_31_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64540672);
vit_Constant_26_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64543744);
vit_Constant_139_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+64546816);
vit_Constant_25_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+66906112);
vit_Constant_138_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+66909184);
vit_Constant_24_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+69268480);
vit_Constant_27_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+69271552);
vit_Constant_143_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+69274624);
vit_Constant_142_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+78711808);
vit_Constant_498_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88148992);
vit_Constant_508_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88149056);
vit_Constant_32_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88149120);
vit_Constant_33_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88152192);
vit_Constant_28_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88155264);
vit_Constant_29_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88167552);
vit_Constant_147_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+88170624);
vit_Constant_146_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+90529920);
vit_Constant_539_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92889216);
vit_Constant_549_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92889280);
vit_Constant_40_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92889344);
vit_Constant_41_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92892416);
vit_Constant_36_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92895488);
vit_Constant_145_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+92898560);
vit_Constant_35_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+95257856);
vit_Constant_144_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+95260928);
vit_Constant_34_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+97620224);
vit_Constant_37_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+97623296);
vit_Constant_149_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+97626368);
vit_Constant_148_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+107063552);
vit_Constant_605_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116500736);
vit_Constant_615_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116500800);
vit_Constant_42_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116500864);
vit_Constant_43_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116503936);
vit_Constant_38_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116507008);
vit_Constant_39_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116519296);
vit_Constant_153_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+116522368);
vit_Constant_152_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+118881664);
vit_Constant_646_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121240960);
vit_Constant_656_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121241024);
vit_Constant_50_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121241088);
vit_Constant_51_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121244160);
vit_Constant_46_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121247232);
vit_Constant_151_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+121250304);
vit_Constant_45_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+123609600);
vit_Constant_150_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+123612672);
vit_Constant_44_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+125971968);
vit_Constant_47_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+125975040);
vit_Constant_155_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+125978112);
vit_Constant_154_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+135415296);
vit_Constant_712_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144852480);
vit_Constant_722_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144852544);
vit_Constant_52_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144852608);
vit_Constant_53_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144855680);
vit_Constant_48_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144858752);
vit_Constant_49_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144871040);
vit_Constant_159_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+144874112);
vit_Constant_158_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+147233408);
vit_Constant_753_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149592704);
vit_Constant_763_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149592768);
vit_Constant_60_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149592832);
vit_Constant_61_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149595904);
vit_Constant_56_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149598976);
vit_Constant_157_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+149602048);
vit_Constant_55_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+151961344);
vit_Constant_156_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+151964416);
vit_Constant_54_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+154323712);
vit_Constant_57_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+154326784);
vit_Constant_161_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+154329856);
vit_Constant_160_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+163767040);
vit_Constant_819_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173204224);
vit_Constant_829_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173204288);
vit_Constant_62_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173204352);
vit_Constant_63_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173207424);
vit_Constant_58_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173210496);
vit_Constant_59_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173222784);
vit_Constant_165_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+173225856);
vit_Constant_164_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+175585152);
vit_Constant_860_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177944448);
vit_Constant_870_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177944512);
vit_Constant_70_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177944576);
vit_Constant_71_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177947648);
vit_Constant_66_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177950720);
vit_Constant_163_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+177953792);
vit_Constant_65_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+180313088);
vit_Constant_162_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+180316160);
vit_Constant_64_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+182675456);
vit_Constant_67_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+182678528);
vit_Constant_167_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+182681600);
vit_Constant_166_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+192118784);
vit_Constant_926_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201555968);
vit_Constant_936_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201556032);
vit_Constant_72_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201556096);
vit_Constant_73_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201559168);
vit_Constant_68_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201562240);
vit_Constant_69_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201574528);
vit_Constant_171_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+201577600);
vit_Constant_170_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+203936896);
vit_Constant_967_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206296192);
vit_Constant_977_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206296256);
vit_Constant_80_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206296320);
vit_Constant_81_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206299392);
vit_Constant_76_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206302464);
vit_Constant_169_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+206305536);
vit_Constant_75_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+208664832);
vit_Constant_168_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+208667904);
vit_Constant_74_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+211027200);
vit_Constant_77_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+211030272);
vit_Constant_173_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+211033344);
vit_Constant_172_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+220470528);
vit_Constant_1033_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229907712);
vit_Constant_1043_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229907776);
vit_Constant_82_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229907840);
vit_Constant_83_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229910912);
vit_Constant_78_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229913984);
vit_Constant_79_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229926272);
vit_Constant_177_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+229929344);
vit_Constant_176_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+232288640);
vit_Constant_1074_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234647936);
vit_Constant_1084_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234648000);
vit_Constant_90_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234648064);
vit_Constant_91_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234651136);
vit_Constant_86_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234654208);
vit_Constant_175_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+234657280);
vit_Constant_85_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+237016576);
vit_Constant_174_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+237019648);
vit_Constant_84_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+239378944);
vit_Constant_87_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+239382016);
vit_Constant_179_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+239385088);
vit_Constant_178_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+248822272);
vit_Constant_1140_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258259456);
vit_Constant_1150_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258259520);
vit_Constant_92_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258259584);
vit_Constant_93_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258262656);
vit_Constant_88_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258265728);
vit_Constant_89_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258278016);
vit_Constant_183_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+258281088);
vit_Constant_182_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+260640384);
vit_Constant_1181_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+262999680);
vit_Constant_1191_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+262999744);
vit_Constant_100_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+262999808);
vit_Constant_101_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+263002880);
vit_Constant_96_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+263005952);
vit_Constant_181_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+263009024);
vit_Constant_95_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+265368320);
vit_Constant_180_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+265371392);
vit_Constant_94_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+267730688);
vit_Constant_97_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+267733760);
vit_Constant_185_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+267736832);
vit_Constant_184_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+277174016);
vit_Constant_1247_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286611200);
vit_Constant_1257_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286611264);
vit_Constant_102_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286611328);
vit_Constant_103_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286614400);
vit_Constant_98_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286617472);
vit_Constant_99_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286629760);
vit_Constant_189_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+286632832);
vit_Constant_188_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+288992128);
vit_Constant_1288_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291351424);
vit_Constant_1298_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291351488);
vit_Constant_110_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291351552);
vit_Constant_111_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291354624);
vit_Constant_106_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291357696);
vit_Constant_187_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+291360768);
vit_Constant_105_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+293720064);
vit_Constant_186_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+293723136);
vit_Constant_104_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+296082432);
vit_Constant_107_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+296085504);
vit_Constant_191_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+296088576);
vit_Constant_190_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+305525760);
vit_Constant_1354_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314962944);
vit_Constant_1364_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314963008);
vit_Constant_112_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314963072);
vit_Constant_113_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314966144);
vit_Constant_108_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314969216);
vit_Constant_109_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314981504);
vit_Constant_195_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+314984576);
vit_Constant_194_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+317343872);
vit_Constant_1395_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319703168);
vit_Constant_1405_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319703232);
vit_Constant_120_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319703296);
vit_Constant_121_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319706368);
vit_Constant_116_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319709440);
vit_Constant_193_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+319712512);
vit_Constant_115_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+322071808);
vit_Constant_192_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+322074880);
vit_Constant_114_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+324434176);
vit_Constant_117_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+324437248);
vit_Constant_197_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+324440320);
vit_Constant_196_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+333877504);
vit_Constant_1461_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343314688);
vit_Constant_1471_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343314752);
vit_Constant_122_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343314816);
vit_Constant_123_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343317888);
vit_Constant_118_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343320960);
vit_Constant_119_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343333248);
vit_Constant_1502_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343336320);
vit_Constant_1512_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343336384);
vit_Constant_124_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343336448);
vit_Constant_125_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343339520);
vit_last_hidden_state = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343342592);
vit_Result_1527_0 = (float*)(vit_group_persist_CUDA_GPU0_allocator_memory_pool+343342592);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&vit_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&vit_cudnn_handle_0));
 // name=embeddings.position_embeddings
vit_Constant_float_cuda_Constant_1(0, vit_Constant_1_0);
 // name=embeddings.patch_embeddings.projection.bias
vit_Constant_float_cuda_Constant_3(0, vit_Constant_3_0);
 // name=embeddings.patch_embeddings.projection.weight
vit_Constant_float_cuda_Constant_2(0, vit_Constant_2_0);
 // name=/embeddings/Expand_output_0
vit_Constant_float_cuda_Constant_200(0, vit_Constant_200_0);
 // name=onnx::MatMul_1843
vit_Constant_float_cuda_Constant_129(0, vit_Constant_129_0);
 // name=onnx::MatMul_1837
vit_Constant_float_cuda_Constant_128(0, vit_Constant_128_0);
 // name=Constant_218
vit_Constant_float_cuda_Constant_218(0, vit_Constant_218_0);
 // name=ortshared_1_0_1_3_token_103
vit_Constant_float_cuda_Constant_206(0, vit_Constant_206_0);
 // name=Constant_228
vit_Constant_float_cuda_Constant_228(0, vit_Constant_228_0);
 // name=ortshared_1_0_1_4_token_104
vit_Constant_float_cuda_Constant_202(0, vit_Constant_202_0);
 // name=encoder.layer.0.layernorm_before.weight
vit_Constant_float_cuda_Constant_10(0, vit_Constant_10_0);
 // name=encoder.layer.0.layernorm_before.bias
vit_Constant_float_cuda_Constant_11(0, vit_Constant_11_0);
 // name=encoder.layer.0.attention.attention.value.bias
vit_Constant_float_cuda_Constant_6(0, vit_Constant_6_0);
 // name=onnx::MatMul_1834
vit_Constant_float_cuda_Constant_127(0, vit_Constant_127_0);
 // name=encoder.layer.0.attention.attention.key.bias
vit_Constant_float_cuda_Constant_5(0, vit_Constant_5_0);
 // name=ortshared_1_1_1_0_token_101
vit_Constant_float_cuda_Constant_201(0, vit_Constant_201_0);
 // name=onnx::MatMul_1833
vit_Constant_float_cuda_Constant_126(0, vit_Constant_126_0);
 // name=encoder.layer.0.attention.attention.query.bias
vit_Constant_float_cuda_Constant_4(0, vit_Constant_4_0);
 // name=encoder.layer.0.attention.output.dense.bias
vit_Constant_float_cuda_Constant_7(0, vit_Constant_7_0);
 // name=onnx::MatMul_1845
vit_Constant_float_cuda_Constant_131(0, vit_Constant_131_0);
 // name=onnx::MatMul_1844
vit_Constant_float_cuda_Constant_130(0, vit_Constant_130_0);
 // name=Constant_284
vit_Constant_float_cuda_Constant_284(0, vit_Constant_284_0);
 // name=Constant_294
vit_Constant_float_cuda_Constant_294(0, vit_Constant_294_0);
 // name=encoder.layer.0.layernorm_after.weight
vit_Constant_float_cuda_Constant_12(0, vit_Constant_12_0);
 // name=encoder.layer.0.layernorm_after.bias
vit_Constant_float_cuda_Constant_13(0, vit_Constant_13_0);
 // name=encoder.layer.0.intermediate.dense.bias
vit_Constant_float_cuda_Constant_8(0, vit_Constant_8_0);
 // name=ortshared_1_0_1_2_token_100
vit_Constant_float_cuda_Constant_205(0, vit_Constant_205_0);
 // name=ortshared_1_0_1_0_token_97
vit_Constant_float_cuda_Constant_203(0, vit_Constant_203_0);
 // name=ortshared_1_0_1_1_token_99
vit_Constant_float_cuda_Constant_204(0, vit_Constant_204_0);
 // name=encoder.layer.0.output.dense.bias
vit_Constant_float_cuda_Constant_9(0, vit_Constant_9_0);
 // name=onnx::MatMul_1856
vit_Constant_float_cuda_Constant_135(0, vit_Constant_135_0);
 // name=onnx::MatMul_1850
vit_Constant_float_cuda_Constant_134(0, vit_Constant_134_0);
 // name=Constant_325
vit_Constant_float_cuda_Constant_325(0, vit_Constant_325_0);
 // name=Constant_335
vit_Constant_float_cuda_Constant_335(0, vit_Constant_335_0);
 // name=encoder.layer.1.layernorm_before.weight
vit_Constant_float_cuda_Constant_20(0, vit_Constant_20_0);
 // name=encoder.layer.1.layernorm_before.bias
vit_Constant_float_cuda_Constant_21(0, vit_Constant_21_0);
 // name=encoder.layer.1.attention.attention.value.bias
vit_Constant_float_cuda_Constant_16(0, vit_Constant_16_0);
 // name=onnx::MatMul_1847
vit_Constant_float_cuda_Constant_133(0, vit_Constant_133_0);
 // name=encoder.layer.1.attention.attention.key.bias
vit_Constant_float_cuda_Constant_15(0, vit_Constant_15_0);
 // name=onnx::MatMul_1846
vit_Constant_float_cuda_Constant_132(0, vit_Constant_132_0);
 // name=encoder.layer.1.attention.attention.query.bias
vit_Constant_float_cuda_Constant_14(0, vit_Constant_14_0);
 // name=encoder.layer.1.attention.output.dense.bias
vit_Constant_float_cuda_Constant_17(0, vit_Constant_17_0);
 // name=onnx::MatMul_1858
vit_Constant_float_cuda_Constant_137(0, vit_Constant_137_0);
 // name=onnx::MatMul_1857
vit_Constant_float_cuda_Constant_136(0, vit_Constant_136_0);
 // name=Constant_391
vit_Constant_float_cuda_Constant_391(0, vit_Constant_391_0);
 // name=Constant_401
vit_Constant_float_cuda_Constant_401(0, vit_Constant_401_0);
 // name=encoder.layer.1.layernorm_after.weight
vit_Constant_float_cuda_Constant_22(0, vit_Constant_22_0);
 // name=encoder.layer.1.layernorm_after.bias
vit_Constant_float_cuda_Constant_23(0, vit_Constant_23_0);
 // name=encoder.layer.1.intermediate.dense.bias
vit_Constant_float_cuda_Constant_18(0, vit_Constant_18_0);
 // name=encoder.layer.1.output.dense.bias
vit_Constant_float_cuda_Constant_19(0, vit_Constant_19_0);
 // name=onnx::MatMul_1869
vit_Constant_float_cuda_Constant_141(0, vit_Constant_141_0);
 // name=onnx::MatMul_1863
vit_Constant_float_cuda_Constant_140(0, vit_Constant_140_0);
 // name=Constant_432
vit_Constant_float_cuda_Constant_432(0, vit_Constant_432_0);
 // name=Constant_442
vit_Constant_float_cuda_Constant_442(0, vit_Constant_442_0);
 // name=encoder.layer.2.layernorm_before.weight
vit_Constant_float_cuda_Constant_30(0, vit_Constant_30_0);
 // name=encoder.layer.2.layernorm_before.bias
vit_Constant_float_cuda_Constant_31(0, vit_Constant_31_0);
 // name=encoder.layer.2.attention.attention.value.bias
vit_Constant_float_cuda_Constant_26(0, vit_Constant_26_0);
 // name=onnx::MatMul_1860
vit_Constant_float_cuda_Constant_139(0, vit_Constant_139_0);
 // name=encoder.layer.2.attention.attention.key.bias
vit_Constant_float_cuda_Constant_25(0, vit_Constant_25_0);
 // name=onnx::MatMul_1859
vit_Constant_float_cuda_Constant_138(0, vit_Constant_138_0);
 // name=encoder.layer.2.attention.attention.query.bias
vit_Constant_float_cuda_Constant_24(0, vit_Constant_24_0);
 // name=encoder.layer.2.attention.output.dense.bias
vit_Constant_float_cuda_Constant_27(0, vit_Constant_27_0);
 // name=onnx::MatMul_1871
vit_Constant_float_cuda_Constant_143(0, vit_Constant_143_0);
 // name=onnx::MatMul_1870
vit_Constant_float_cuda_Constant_142(0, vit_Constant_142_0);
 // name=Constant_498
vit_Constant_float_cuda_Constant_498(0, vit_Constant_498_0);
 // name=Constant_508
vit_Constant_float_cuda_Constant_508(0, vit_Constant_508_0);
 // name=encoder.layer.2.layernorm_after.weight
vit_Constant_float_cuda_Constant_32(0, vit_Constant_32_0);
 // name=encoder.layer.2.layernorm_after.bias
vit_Constant_float_cuda_Constant_33(0, vit_Constant_33_0);
 // name=encoder.layer.2.intermediate.dense.bias
vit_Constant_float_cuda_Constant_28(0, vit_Constant_28_0);
 // name=encoder.layer.2.output.dense.bias
vit_Constant_float_cuda_Constant_29(0, vit_Constant_29_0);
 // name=onnx::MatMul_1882
vit_Constant_float_cuda_Constant_147(0, vit_Constant_147_0);
 // name=onnx::MatMul_1876
vit_Constant_float_cuda_Constant_146(0, vit_Constant_146_0);
 // name=Constant_539
vit_Constant_float_cuda_Constant_539(0, vit_Constant_539_0);
 // name=Constant_549
vit_Constant_float_cuda_Constant_549(0, vit_Constant_549_0);
 // name=encoder.layer.3.layernorm_before.weight
vit_Constant_float_cuda_Constant_40(0, vit_Constant_40_0);
 // name=encoder.layer.3.layernorm_before.bias
vit_Constant_float_cuda_Constant_41(0, vit_Constant_41_0);
 // name=encoder.layer.3.attention.attention.value.bias
vit_Constant_float_cuda_Constant_36(0, vit_Constant_36_0);
 // name=onnx::MatMul_1873
vit_Constant_float_cuda_Constant_145(0, vit_Constant_145_0);
 // name=encoder.layer.3.attention.attention.key.bias
vit_Constant_float_cuda_Constant_35(0, vit_Constant_35_0);
 // name=onnx::MatMul_1872
vit_Constant_float_cuda_Constant_144(0, vit_Constant_144_0);
 // name=encoder.layer.3.attention.attention.query.bias
vit_Constant_float_cuda_Constant_34(0, vit_Constant_34_0);
 // name=encoder.layer.3.attention.output.dense.bias
vit_Constant_float_cuda_Constant_37(0, vit_Constant_37_0);
 // name=onnx::MatMul_1884
vit_Constant_float_cuda_Constant_149(0, vit_Constant_149_0);
 // name=onnx::MatMul_1883
vit_Constant_float_cuda_Constant_148(0, vit_Constant_148_0);
 // name=Constant_605
vit_Constant_float_cuda_Constant_605(0, vit_Constant_605_0);
 // name=Constant_615
vit_Constant_float_cuda_Constant_615(0, vit_Constant_615_0);
 // name=encoder.layer.3.layernorm_after.weight
vit_Constant_float_cuda_Constant_42(0, vit_Constant_42_0);
 // name=encoder.layer.3.layernorm_after.bias
vit_Constant_float_cuda_Constant_43(0, vit_Constant_43_0);
 // name=encoder.layer.3.intermediate.dense.bias
vit_Constant_float_cuda_Constant_38(0, vit_Constant_38_0);
 // name=encoder.layer.3.output.dense.bias
vit_Constant_float_cuda_Constant_39(0, vit_Constant_39_0);
 // name=onnx::MatMul_1895
vit_Constant_float_cuda_Constant_153(0, vit_Constant_153_0);
 // name=onnx::MatMul_1889
vit_Constant_float_cuda_Constant_152(0, vit_Constant_152_0);
 // name=Constant_646
vit_Constant_float_cuda_Constant_646(0, vit_Constant_646_0);
 // name=Constant_656
vit_Constant_float_cuda_Constant_656(0, vit_Constant_656_0);
 // name=encoder.layer.4.layernorm_before.weight
vit_Constant_float_cuda_Constant_50(0, vit_Constant_50_0);
 // name=encoder.layer.4.layernorm_before.bias
vit_Constant_float_cuda_Constant_51(0, vit_Constant_51_0);
 // name=encoder.layer.4.attention.attention.value.bias
vit_Constant_float_cuda_Constant_46(0, vit_Constant_46_0);
 // name=onnx::MatMul_1886
vit_Constant_float_cuda_Constant_151(0, vit_Constant_151_0);
 // name=encoder.layer.4.attention.attention.key.bias
vit_Constant_float_cuda_Constant_45(0, vit_Constant_45_0);
 // name=onnx::MatMul_1885
vit_Constant_float_cuda_Constant_150(0, vit_Constant_150_0);
 // name=encoder.layer.4.attention.attention.query.bias
vit_Constant_float_cuda_Constant_44(0, vit_Constant_44_0);
 // name=encoder.layer.4.attention.output.dense.bias
vit_Constant_float_cuda_Constant_47(0, vit_Constant_47_0);
 // name=onnx::MatMul_1897
vit_Constant_float_cuda_Constant_155(0, vit_Constant_155_0);
 // name=onnx::MatMul_1896
vit_Constant_float_cuda_Constant_154(0, vit_Constant_154_0);
 // name=Constant_712
vit_Constant_float_cuda_Constant_712(0, vit_Constant_712_0);
 // name=Constant_722
vit_Constant_float_cuda_Constant_722(0, vit_Constant_722_0);
 // name=encoder.layer.4.layernorm_after.weight
vit_Constant_float_cuda_Constant_52(0, vit_Constant_52_0);
 // name=encoder.layer.4.layernorm_after.bias
vit_Constant_float_cuda_Constant_53(0, vit_Constant_53_0);
 // name=encoder.layer.4.intermediate.dense.bias
vit_Constant_float_cuda_Constant_48(0, vit_Constant_48_0);
 // name=encoder.layer.4.output.dense.bias
vit_Constant_float_cuda_Constant_49(0, vit_Constant_49_0);
 // name=onnx::MatMul_1908
vit_Constant_float_cuda_Constant_159(0, vit_Constant_159_0);
 // name=onnx::MatMul_1902
vit_Constant_float_cuda_Constant_158(0, vit_Constant_158_0);
 // name=Constant_753
vit_Constant_float_cuda_Constant_753(0, vit_Constant_753_0);
 // name=Constant_763
vit_Constant_float_cuda_Constant_763(0, vit_Constant_763_0);
 // name=encoder.layer.5.layernorm_before.weight
vit_Constant_float_cuda_Constant_60(0, vit_Constant_60_0);
 // name=encoder.layer.5.layernorm_before.bias
vit_Constant_float_cuda_Constant_61(0, vit_Constant_61_0);
 // name=encoder.layer.5.attention.attention.value.bias
vit_Constant_float_cuda_Constant_56(0, vit_Constant_56_0);
 // name=onnx::MatMul_1899
vit_Constant_float_cuda_Constant_157(0, vit_Constant_157_0);
 // name=encoder.layer.5.attention.attention.key.bias
vit_Constant_float_cuda_Constant_55(0, vit_Constant_55_0);
 // name=onnx::MatMul_1898
vit_Constant_float_cuda_Constant_156(0, vit_Constant_156_0);
 // name=encoder.layer.5.attention.attention.query.bias
vit_Constant_float_cuda_Constant_54(0, vit_Constant_54_0);
 // name=encoder.layer.5.attention.output.dense.bias
vit_Constant_float_cuda_Constant_57(0, vit_Constant_57_0);
 // name=onnx::MatMul_1910
vit_Constant_float_cuda_Constant_161(0, vit_Constant_161_0);
 // name=onnx::MatMul_1909
vit_Constant_float_cuda_Constant_160(0, vit_Constant_160_0);
 // name=Constant_819
vit_Constant_float_cuda_Constant_819(0, vit_Constant_819_0);
 // name=Constant_829
vit_Constant_float_cuda_Constant_829(0, vit_Constant_829_0);
 // name=encoder.layer.5.layernorm_after.weight
vit_Constant_float_cuda_Constant_62(0, vit_Constant_62_0);
 // name=encoder.layer.5.layernorm_after.bias
vit_Constant_float_cuda_Constant_63(0, vit_Constant_63_0);
 // name=encoder.layer.5.intermediate.dense.bias
vit_Constant_float_cuda_Constant_58(0, vit_Constant_58_0);
 // name=encoder.layer.5.output.dense.bias
vit_Constant_float_cuda_Constant_59(0, vit_Constant_59_0);
 // name=onnx::MatMul_1921
vit_Constant_float_cuda_Constant_165(0, vit_Constant_165_0);
 // name=onnx::MatMul_1915
vit_Constant_float_cuda_Constant_164(0, vit_Constant_164_0);
 // name=Constant_860
vit_Constant_float_cuda_Constant_860(0, vit_Constant_860_0);
 // name=Constant_870
vit_Constant_float_cuda_Constant_870(0, vit_Constant_870_0);
 // name=encoder.layer.6.layernorm_before.weight
vit_Constant_float_cuda_Constant_70(0, vit_Constant_70_0);
 // name=encoder.layer.6.layernorm_before.bias
vit_Constant_float_cuda_Constant_71(0, vit_Constant_71_0);
 // name=encoder.layer.6.attention.attention.value.bias
vit_Constant_float_cuda_Constant_66(0, vit_Constant_66_0);
 // name=onnx::MatMul_1912
vit_Constant_float_cuda_Constant_163(0, vit_Constant_163_0);
 // name=encoder.layer.6.attention.attention.key.bias
vit_Constant_float_cuda_Constant_65(0, vit_Constant_65_0);
 // name=onnx::MatMul_1911
vit_Constant_float_cuda_Constant_162(0, vit_Constant_162_0);
 // name=encoder.layer.6.attention.attention.query.bias
vit_Constant_float_cuda_Constant_64(0, vit_Constant_64_0);
 // name=encoder.layer.6.attention.output.dense.bias
vit_Constant_float_cuda_Constant_67(0, vit_Constant_67_0);
 // name=onnx::MatMul_1923
vit_Constant_float_cuda_Constant_167(0, vit_Constant_167_0);
 // name=onnx::MatMul_1922
vit_Constant_float_cuda_Constant_166(0, vit_Constant_166_0);
 // name=Constant_926
vit_Constant_float_cuda_Constant_926(0, vit_Constant_926_0);
 // name=Constant_936
vit_Constant_float_cuda_Constant_936(0, vit_Constant_936_0);
 // name=encoder.layer.6.layernorm_after.weight
vit_Constant_float_cuda_Constant_72(0, vit_Constant_72_0);
 // name=encoder.layer.6.layernorm_after.bias
vit_Constant_float_cuda_Constant_73(0, vit_Constant_73_0);
 // name=encoder.layer.6.intermediate.dense.bias
vit_Constant_float_cuda_Constant_68(0, vit_Constant_68_0);
 // name=encoder.layer.6.output.dense.bias
vit_Constant_float_cuda_Constant_69(0, vit_Constant_69_0);
 // name=onnx::MatMul_1934
vit_Constant_float_cuda_Constant_171(0, vit_Constant_171_0);
 // name=onnx::MatMul_1928
vit_Constant_float_cuda_Constant_170(0, vit_Constant_170_0);
 // name=Constant_967
vit_Constant_float_cuda_Constant_967(0, vit_Constant_967_0);
 // name=Constant_977
vit_Constant_float_cuda_Constant_977(0, vit_Constant_977_0);
 // name=encoder.layer.7.layernorm_before.weight
vit_Constant_float_cuda_Constant_80(0, vit_Constant_80_0);
 // name=encoder.layer.7.layernorm_before.bias
vit_Constant_float_cuda_Constant_81(0, vit_Constant_81_0);
 // name=encoder.layer.7.attention.attention.value.bias
vit_Constant_float_cuda_Constant_76(0, vit_Constant_76_0);
 // name=onnx::MatMul_1925
vit_Constant_float_cuda_Constant_169(0, vit_Constant_169_0);
 // name=encoder.layer.7.attention.attention.key.bias
vit_Constant_float_cuda_Constant_75(0, vit_Constant_75_0);
 // name=onnx::MatMul_1924
vit_Constant_float_cuda_Constant_168(0, vit_Constant_168_0);
 // name=encoder.layer.7.attention.attention.query.bias
vit_Constant_float_cuda_Constant_74(0, vit_Constant_74_0);
 // name=encoder.layer.7.attention.output.dense.bias
vit_Constant_float_cuda_Constant_77(0, vit_Constant_77_0);
 // name=onnx::MatMul_1936
vit_Constant_float_cuda_Constant_173(0, vit_Constant_173_0);
 // name=onnx::MatMul_1935
vit_Constant_float_cuda_Constant_172(0, vit_Constant_172_0);
 // name=Constant_1033
vit_Constant_float_cuda_Constant_1033(0, vit_Constant_1033_0);
 // name=Constant_1043
vit_Constant_float_cuda_Constant_1043(0, vit_Constant_1043_0);
 // name=encoder.layer.7.layernorm_after.weight
vit_Constant_float_cuda_Constant_82(0, vit_Constant_82_0);
 // name=encoder.layer.7.layernorm_after.bias
vit_Constant_float_cuda_Constant_83(0, vit_Constant_83_0);
 // name=encoder.layer.7.intermediate.dense.bias
vit_Constant_float_cuda_Constant_78(0, vit_Constant_78_0);
 // name=encoder.layer.7.output.dense.bias
vit_Constant_float_cuda_Constant_79(0, vit_Constant_79_0);
 // name=onnx::MatMul_1947
vit_Constant_float_cuda_Constant_177(0, vit_Constant_177_0);
 // name=onnx::MatMul_1941
vit_Constant_float_cuda_Constant_176(0, vit_Constant_176_0);
 // name=Constant_1074
vit_Constant_float_cuda_Constant_1074(0, vit_Constant_1074_0);
 // name=Constant_1084
vit_Constant_float_cuda_Constant_1084(0, vit_Constant_1084_0);
 // name=encoder.layer.8.layernorm_before.weight
vit_Constant_float_cuda_Constant_90(0, vit_Constant_90_0);
 // name=encoder.layer.8.layernorm_before.bias
vit_Constant_float_cuda_Constant_91(0, vit_Constant_91_0);
 // name=encoder.layer.8.attention.attention.value.bias
vit_Constant_float_cuda_Constant_86(0, vit_Constant_86_0);
 // name=onnx::MatMul_1938
vit_Constant_float_cuda_Constant_175(0, vit_Constant_175_0);
 // name=encoder.layer.8.attention.attention.key.bias
vit_Constant_float_cuda_Constant_85(0, vit_Constant_85_0);
 // name=onnx::MatMul_1937
vit_Constant_float_cuda_Constant_174(0, vit_Constant_174_0);
 // name=encoder.layer.8.attention.attention.query.bias
vit_Constant_float_cuda_Constant_84(0, vit_Constant_84_0);
 // name=encoder.layer.8.attention.output.dense.bias
vit_Constant_float_cuda_Constant_87(0, vit_Constant_87_0);
 // name=onnx::MatMul_1949
vit_Constant_float_cuda_Constant_179(0, vit_Constant_179_0);
 // name=onnx::MatMul_1948
vit_Constant_float_cuda_Constant_178(0, vit_Constant_178_0);
 // name=Constant_1140
vit_Constant_float_cuda_Constant_1140(0, vit_Constant_1140_0);
 // name=Constant_1150
vit_Constant_float_cuda_Constant_1150(0, vit_Constant_1150_0);
 // name=encoder.layer.8.layernorm_after.weight
vit_Constant_float_cuda_Constant_92(0, vit_Constant_92_0);
 // name=encoder.layer.8.layernorm_after.bias
vit_Constant_float_cuda_Constant_93(0, vit_Constant_93_0);
 // name=encoder.layer.8.intermediate.dense.bias
vit_Constant_float_cuda_Constant_88(0, vit_Constant_88_0);
 // name=encoder.layer.8.output.dense.bias
vit_Constant_float_cuda_Constant_89(0, vit_Constant_89_0);
 // name=onnx::MatMul_1960
vit_Constant_float_cuda_Constant_183(0, vit_Constant_183_0);
 // name=onnx::MatMul_1954
vit_Constant_float_cuda_Constant_182(0, vit_Constant_182_0);
 // name=Constant_1181
vit_Constant_float_cuda_Constant_1181(0, vit_Constant_1181_0);
 // name=Constant_1191
vit_Constant_float_cuda_Constant_1191(0, vit_Constant_1191_0);
 // name=encoder.layer.9.layernorm_before.weight
vit_Constant_float_cuda_Constant_100(0, vit_Constant_100_0);
 // name=encoder.layer.9.layernorm_before.bias
vit_Constant_float_cuda_Constant_101(0, vit_Constant_101_0);
 // name=encoder.layer.9.attention.attention.value.bias
vit_Constant_float_cuda_Constant_96(0, vit_Constant_96_0);
 // name=onnx::MatMul_1951
vit_Constant_float_cuda_Constant_181(0, vit_Constant_181_0);
 // name=encoder.layer.9.attention.attention.key.bias
vit_Constant_float_cuda_Constant_95(0, vit_Constant_95_0);
 // name=onnx::MatMul_1950
vit_Constant_float_cuda_Constant_180(0, vit_Constant_180_0);
 // name=encoder.layer.9.attention.attention.query.bias
vit_Constant_float_cuda_Constant_94(0, vit_Constant_94_0);
 // name=encoder.layer.9.attention.output.dense.bias
vit_Constant_float_cuda_Constant_97(0, vit_Constant_97_0);
 // name=onnx::MatMul_1962
vit_Constant_float_cuda_Constant_185(0, vit_Constant_185_0);
 // name=onnx::MatMul_1961
vit_Constant_float_cuda_Constant_184(0, vit_Constant_184_0);
 // name=Constant_1247
vit_Constant_float_cuda_Constant_1247(0, vit_Constant_1247_0);
 // name=Constant_1257
vit_Constant_float_cuda_Constant_1257(0, vit_Constant_1257_0);
 // name=encoder.layer.9.layernorm_after.weight
vit_Constant_float_cuda_Constant_102(0, vit_Constant_102_0);
 // name=encoder.layer.9.layernorm_after.bias
vit_Constant_float_cuda_Constant_103(0, vit_Constant_103_0);
 // name=encoder.layer.9.intermediate.dense.bias
vit_Constant_float_cuda_Constant_98(0, vit_Constant_98_0);
 // name=encoder.layer.9.output.dense.bias
vit_Constant_float_cuda_Constant_99(0, vit_Constant_99_0);
 // name=onnx::MatMul_1973
vit_Constant_float_cuda_Constant_189(0, vit_Constant_189_0);
 // name=onnx::MatMul_1967
vit_Constant_float_cuda_Constant_188(0, vit_Constant_188_0);
 // name=Constant_1288
vit_Constant_float_cuda_Constant_1288(0, vit_Constant_1288_0);
 // name=Constant_1298
vit_Constant_float_cuda_Constant_1298(0, vit_Constant_1298_0);
 // name=encoder.layer.10.layernorm_before.weight
vit_Constant_float_cuda_Constant_110(0, vit_Constant_110_0);
 // name=encoder.layer.10.layernorm_before.bias
vit_Constant_float_cuda_Constant_111(0, vit_Constant_111_0);
 // name=encoder.layer.10.attention.attention.value.bias
vit_Constant_float_cuda_Constant_106(0, vit_Constant_106_0);
 // name=onnx::MatMul_1964
vit_Constant_float_cuda_Constant_187(0, vit_Constant_187_0);
 // name=encoder.layer.10.attention.attention.key.bias
vit_Constant_float_cuda_Constant_105(0, vit_Constant_105_0);
 // name=onnx::MatMul_1963
vit_Constant_float_cuda_Constant_186(0, vit_Constant_186_0);
 // name=encoder.layer.10.attention.attention.query.bias
vit_Constant_float_cuda_Constant_104(0, vit_Constant_104_0);
 // name=encoder.layer.10.attention.output.dense.bias
vit_Constant_float_cuda_Constant_107(0, vit_Constant_107_0);
 // name=onnx::MatMul_1975
vit_Constant_float_cuda_Constant_191(0, vit_Constant_191_0);
 // name=onnx::MatMul_1974
vit_Constant_float_cuda_Constant_190(0, vit_Constant_190_0);
 // name=Constant_1354
vit_Constant_float_cuda_Constant_1354(0, vit_Constant_1354_0);
 // name=Constant_1364
vit_Constant_float_cuda_Constant_1364(0, vit_Constant_1364_0);
 // name=encoder.layer.10.layernorm_after.weight
vit_Constant_float_cuda_Constant_112(0, vit_Constant_112_0);
 // name=encoder.layer.10.layernorm_after.bias
vit_Constant_float_cuda_Constant_113(0, vit_Constant_113_0);
 // name=encoder.layer.10.intermediate.dense.bias
vit_Constant_float_cuda_Constant_108(0, vit_Constant_108_0);
 // name=encoder.layer.10.output.dense.bias
vit_Constant_float_cuda_Constant_109(0, vit_Constant_109_0);
 // name=onnx::MatMul_1986
vit_Constant_float_cuda_Constant_195(0, vit_Constant_195_0);
 // name=onnx::MatMul_1980
vit_Constant_float_cuda_Constant_194(0, vit_Constant_194_0);
 // name=Constant_1395
vit_Constant_float_cuda_Constant_1395(0, vit_Constant_1395_0);
 // name=Constant_1405
vit_Constant_float_cuda_Constant_1405(0, vit_Constant_1405_0);
 // name=encoder.layer.11.layernorm_before.weight
vit_Constant_float_cuda_Constant_120(0, vit_Constant_120_0);
 // name=encoder.layer.11.layernorm_before.bias
vit_Constant_float_cuda_Constant_121(0, vit_Constant_121_0);
 // name=encoder.layer.11.attention.attention.value.bias
vit_Constant_float_cuda_Constant_116(0, vit_Constant_116_0);
 // name=onnx::MatMul_1977
vit_Constant_float_cuda_Constant_193(0, vit_Constant_193_0);
 // name=encoder.layer.11.attention.attention.key.bias
vit_Constant_float_cuda_Constant_115(0, vit_Constant_115_0);
 // name=onnx::MatMul_1976
vit_Constant_float_cuda_Constant_192(0, vit_Constant_192_0);
 // name=encoder.layer.11.attention.attention.query.bias
vit_Constant_float_cuda_Constant_114(0, vit_Constant_114_0);
 // name=encoder.layer.11.attention.output.dense.bias
vit_Constant_float_cuda_Constant_117(0, vit_Constant_117_0);
 // name=onnx::MatMul_1988
vit_Constant_float_cuda_Constant_197(0, vit_Constant_197_0);
 // name=onnx::MatMul_1987
vit_Constant_float_cuda_Constant_196(0, vit_Constant_196_0);
 // name=Constant_1461
vit_Constant_float_cuda_Constant_1461(0, vit_Constant_1461_0);
 // name=Constant_1471
vit_Constant_float_cuda_Constant_1471(0, vit_Constant_1471_0);
 // name=encoder.layer.11.layernorm_after.weight
vit_Constant_float_cuda_Constant_122(0, vit_Constant_122_0);
 // name=encoder.layer.11.layernorm_after.bias
vit_Constant_float_cuda_Constant_123(0, vit_Constant_123_0);
 // name=encoder.layer.11.intermediate.dense.bias
vit_Constant_float_cuda_Constant_118(0, vit_Constant_118_0);
 // name=encoder.layer.11.output.dense.bias
vit_Constant_float_cuda_Constant_119(0, vit_Constant_119_0);
 // name=Constant_1502
vit_Constant_float_cuda_Constant_1502(0, vit_Constant_1502_0);
 // name=Constant_1512
vit_Constant_float_cuda_Constant_1512(0, vit_Constant_1512_0);
 // name=layernorm.weight
vit_Constant_float_cuda_Constant_124(0, vit_Constant_124_0);
 // name=layernorm.bias
vit_Constant_float_cuda_Constant_125(0, vit_Constant_125_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&vit_num_SMs, cudaDevAttrMultiProcessorCount, 0));
}




extern "C" void vit_cuda_free()
{
CUBLAS_SAFE_CALL(cublasDestroy(vit_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(vit_cudnn_handle_0));

CUDA_SAFE_CALL(cudaFree(vit_group_0_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(vit_group_persist_CUDA_GPU0_allocator_memory_pool));
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "./include/dnn.h"

class vit_Broadcast_float_float_cuda_Broadcast_215_CallKernel : public Kernel {
public:
    vit_Broadcast_float_float_cuda_Broadcast_215_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Broadcast_float_float_cuda_Broadcast_215_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_215_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Broadcast_float_float_cuda_Broadcast_215<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_215_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_Broadcast_float_float_cuda_Broadcast_209_CallKernel : public Kernel {
public:
    vit_Broadcast_float_float_cuda_Broadcast_209_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Broadcast_float_float_cuda_Broadcast_209_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_209_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Broadcast_float_float_cuda_Broadcast_209<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_209_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_Convolution_float_float_float_cuda_lib_Convolution_208Kernel : public Kernel {
public:
    vit_Convolution_float_float_float_cuda_lib_Convolution_208Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Convolution_float_float_float_cuda_lib_Convolution_208";
        this->Id = 0;
        this->mixable = 2;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cudnnHandle_t  cudnn_handle; float*  input0; float*  input1; float*  output0;
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({48, 3, 224, 224, 768, 3, 16, 16, 0, 0, 16, 16, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_208(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 48, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 48, 768, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 768, 3, 16, 16));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 16, 16, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    static bool selected_algo = true;
    static cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    if (!selected_algo) {
        int num_algos;
        int max_algos = 0;
        // cudnnGetConvolutionForwardAlgorithm_v7;
        CUDNN_SAFE_CALL(
            cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
        std::vector<cudnnConvolutionFwdAlgoPerf_t> results(max_algos);
        CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                                tensor_desc_0,
                                                filter_desc,
                                                conv_desc,
                                                tensor_desc_1,
                                                static_cast<int>(results.size()),
                                                &num_algos,
                                                results.data()));
        results.resize(num_algos);
        for (size_t i = 0; i != results.size(); ++i) {
            cudnnConvolutionFwdAlgoPerf_t const& result = results[i];
            if (result.status == CUDNN_STATUS_SUCCESS) {
                conv_fwd_algo = result.algo;
                break;
            }
        }
        selected_algo = true;
    }
    const float alpha = 1.0;
    const float beta = 0.0;
    static void *workspace_ptr_0 = NULL;
    static size_t workspace_size_in_bytes = 0;
    if (!workspace_ptr_0)
    {
        CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, tensor_desc_0, filter_desc, conv_desc, tensor_desc_1, conv_fwd_algo, &workspace_size_in_bytes));
        CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr_0, workspace_size_in_bytes));
    }
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        this->Convolution_float_float_float_cuda_lib_Convolution_208(cudnn_handle, input0, input1, output0);
    }
};


class vit_Add_float_float_float_cuda_Add_210_CallKernel : public Kernel {
public:
    vit_Add_float_float_float_cuda_Add_210_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Add_float_float_float_cuda_Add_210_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_210_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_Add_float_float_float_cuda_Add_210<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Add_float_float_float_cuda_Add_210_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vit_Reshape_float_float_cuda_Reshape_212_CallKernel : public Kernel {
public:
    vit_Reshape_float_float_cuda_Reshape_212_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Reshape_float_float_cuda_Reshape_212_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_212_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_212<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_212_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_Concat_float_float_float_cuda_Concat_213_CallKernel : public Kernel {
public:
    vit_Concat_float_float_float_cuda_Concat_213_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Concat_float_float_float_cuda_Concat_213_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Concat_float_float_float_cuda_Concat_213_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_Concat_float_float_float_cuda_Concat_213<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Concat_float_float_float_cuda_Concat_213_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vit_Sum_float_float_cuda_Sum_217_CallKernel : public Kernel {
public:
    vit_Sum_float_float_cuda_Sum_217_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Sum_float_float_cuda_Sum_217_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Sum_float_float_cuda_Sum_217_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Sum_float_float_cuda_Sum_217<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Sum_float_float_cuda_Sum_217_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, float*  output1, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->output1 = output1, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  output0; float*  output1;
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(grids, blocks, mem, stream, input0, input1, input2, output0, output1);
    }
};


class vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  input2; float*  input3; float*  output0;
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(grids, blocks, mem, stream, input0, input1, input2, input3, output0);
    }
};


class vit_Dot_float_float_float_cuda_lib_Dot_266Kernel : public Kernel {
public:
    vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Dot_float_float_float_cuda_lib_Dot_266";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 768;
    ret[1] = 9456;
    ret[2] = 768;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_266(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 768, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 768));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_266(cublas_handle, input0, input1, output0);
    }
};


class vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vit_Reshape_float_float_cuda_Reshape_270_CallKernel : public Kernel {
public:
    vit_Reshape_float_float_cuda_Reshape_270_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Reshape_float_float_cuda_Reshape_270_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_270_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_270<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_270_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_Reshape_float_float_cuda_Reshape_247_CallKernel : public Kernel {
public:
    vit_Reshape_float_float_cuda_Reshape_247_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Reshape_float_float_cuda_Reshape_247_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_247_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_247<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_247_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel : public Kernel {
public:
    vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 197, 197, 64,
                                    &alpha, input1, 197, 12608, input0, 64, 12608,
                                    &beta, output0, 197, 38809, 576));
                            
    }

}

    void executeImpl(cudaStream_t stream) {
        this->BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle, input0, input1, output0);
    }
};


class vit_Softmax_float_float_cuda_lib_Softmax_265Kernel : public Kernel {
public:
    vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Softmax_float_float_cuda_lib_Softmax_265";
        this->Id = 0;
        this->mixable = 0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }

        cudaStream_t  stream; float*  input0; float*  output0;
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Softmax_float_float_cuda_lib_Softmax_265(cudaStream_t stream, float* input0, float* output0)
{

    dispatch_softmax_forward<float, float, float, false>(stream, output0, input0, 197, 197, 113472);
        

}

    void executeImpl(cudaStream_t stream) {
        this->Softmax_float_float_cuda_lib_Softmax_265(stream, input0, output0);
    }
};


class vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel : public Kernel {
public:
    vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    {

                                static const float alpha = 1.000000000000000000000000e+00F, beta = 0.000000000000000000000000e+00F;
                                // if (!cublas_handle)
                                //     CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
                                CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 197, 197,
                                    &alpha, input1, 64, 12608, input0, 197, 38809,
                                    &beta, output0, 64, 12608, 576));
                            
    }

}

    void executeImpl(cudaStream_t stream) {
        this->BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle, input0, input1, output0);
    }
};


class vit_Reshape_float_float_cuda_Reshape_277_CallKernel : public Kernel {
public:
    vit_Reshape_float_float_cuda_Reshape_277_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Reshape_float_float_cuda_Reshape_277_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vit_Reshape_float_float_cuda_Reshape_277<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_277_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(grids, blocks, mem, stream, input0, input1, input2, output0);
    }
};


class vit_Dot_float_float_float_cuda_lib_Dot_309Kernel : public Kernel {
public:
    vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Dot_float_float_float_cuda_lib_Dot_309";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 3072;
    ret[1] = 9456;
    ret[2] = 768;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_309(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 3072, 9456, 768, &alpha, static_cast<const float*>(input1), 3072, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 3072));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_309(cublas_handle, input0, input1, output0);
    }
};


class vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel : public Kernel {
public:
    vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class vit_Dot_float_float_float_cuda_lib_Dot_320Kernel : public Kernel {
public:
    vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Dot_float_float_float_cuda_lib_Dot_320";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 768;
    ret[1] = 9456;
    ret[2] = 3072;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_320(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 3072, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 3072, &beta, static_cast<float*>(output0), 768));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_320(cublas_handle, input0, input1, output0);
    }
};


class vit_Result_float_float_cuda_lib_Result_1527Kernel : public Kernel {
public:
    vit_Result_float_float_cuda_lib_Result_1527Kernel(float*  input0, float**  output0, float*  Parameter_207_0, float**  vit_Result_1527_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_207_0 = Parameter_207_0, this->vit_Result_1527_0 = vit_Result_1527_0;
        this->kernelName = "vit_Result_float_float_cuda_lib_Result_1527";
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
    float*  Parameter_207_0; float**  vit_Result_1527_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_1527(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        this->Result_float_float_cuda_lib_Result_1527(input0, output0);
    }
};
void ViT::gen_vector(float*  Parameter_207_0, float**  vit_Result_1527_0) {
    kernels.emplace_back(new vit_Broadcast_float_float_cuda_Broadcast_215_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_214_0), std::move(vit_Broadcast_215_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Broadcast_float_float_cuda_Broadcast_209_CallKernel(dim3(112896, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Constant_3_0), std::move(vit_Broadcast_209_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Convolution_float_float_float_cuda_lib_Convolution_208Kernel(std::move(vit_cudnn_handle_0), std::move(Parameter_207_0), std::move(vit_Constant_2_0), std::move(vit_Convolution_208_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Add_float_float_float_cuda_Add_210_CallKernel(dim3(14112, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Convolution_208_0), std::move(vit_Broadcast_209_0), std::move(vit_Add_210_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_212_CallKernel(dim3(13, 48, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_211_0), std::move(vit_Reshape_212_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Concat_float_float_float_cuda_Concat_213_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_200_0), std::move(vit_Reshape_212_0), std::move(vit_Concat_213_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Add_float_float_float_cuda_Add_210_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Concat_213_0), std::move(vit_Broadcast_215_0), std::move(vit_Add_216_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_216_0), std::move(vit_Sum_217_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_218_0), std::move(vit_Sum_217_0), std::move(vit_Divide_220_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_222_0), std::move(vit_Add_216_0), std::move(vit_Subtract_224_0), std::move(vit_Power_226_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_226_0), std::move(vit_Sum_227_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_228_0), std::move(vit_Sum_227_0), std::move(vit_Sqrt_235_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_11_0), std::move(vit_Constant_10_0), std::move(vit_Reshape_236_0), std::move(vit_Subtract_224_0), std::move(vit_Add_242_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_242_0), std::move(vit_Constant_128_0), std::move(vit_Dot_266_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_6_0), std::move(vit_Dot_266_0), std::move(vit_Add_268_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_269_0), std::move(vit_Reshape_270_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_242_0), std::move(vit_Constant_127_0), std::move(vit_Dot_243_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_5_0), std::move(vit_Dot_243_0), std::move(vit_Add_245_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_246_0), std::move(vit_Reshape_247_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_247_0), std::move(vit_Multiply_250_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_242_0), std::move(vit_Constant_126_0), std::move(vit_Dot_251_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_4_0), std::move(vit_Dot_251_0), std::move(vit_Add_253_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_254_0), std::move(vit_Reshape_255_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_255_0), std::move(vit_Multiply_258_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_261_0), std::move(vit_Broadcast_262_0), std::move(vit_BatchMatMul_263_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_264_0), std::move(vit_Softmax_265_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_273_0), std::move(vit_Broadcast_274_0), std::move(vit_BatchMatMul_275_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_276_0), std::move(vit_Reshape_277_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_278_0), std::move(vit_Constant_129_0), std::move(vit_Dot_279_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_7_0), std::move(vit_Dot_279_0), std::move(vit_Add_216_0), std::move(vit_Add_282_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_282_0), std::move(vit_Sum_283_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_284_0), std::move(vit_Sum_283_0), std::move(vit_Divide_286_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_288_0), std::move(vit_Add_282_0), std::move(vit_Subtract_290_0), std::move(vit_Power_292_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_292_0), std::move(vit_Sum_293_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_294_0), std::move(vit_Sum_293_0), std::move(vit_Sqrt_301_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_13_0), std::move(vit_Constant_12_0), std::move(vit_Reshape_302_0), std::move(vit_Subtract_290_0), std::move(vit_Add_308_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_308_0), std::move(vit_Constant_130_0), std::move(vit_Dot_309_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_8_0), std::move(vit_Dot_309_0), std::move(vit_Multiply_319_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_319_0), std::move(vit_Constant_131_0), std::move(vit_Dot_320_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_9_0), std::move(vit_Dot_320_0), std::move(vit_Add_282_0), std::move(vit_Add_323_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_323_0), std::move(vit_Sum_324_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_325_0), std::move(vit_Sum_324_0), std::move(vit_Divide_327_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_329_0), std::move(vit_Add_323_0), std::move(vit_Subtract_331_0), std::move(vit_Power_333_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_333_0), std::move(vit_Sum_334_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_335_0), std::move(vit_Sum_334_0), std::move(vit_Sqrt_342_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_21_0), std::move(vit_Constant_20_0), std::move(vit_Reshape_343_0), std::move(vit_Subtract_331_0), std::move(vit_Add_349_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_349_0), std::move(vit_Constant_134_0), std::move(vit_Dot_373_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_16_0), std::move(vit_Dot_373_0), std::move(vit_Add_375_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_376_0), std::move(vit_Reshape_377_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_349_0), std::move(vit_Constant_133_0), std::move(vit_Dot_350_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_15_0), std::move(vit_Dot_350_0), std::move(vit_Add_352_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_353_0), std::move(vit_Reshape_354_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_354_0), std::move(vit_Multiply_357_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_349_0), std::move(vit_Constant_132_0), std::move(vit_Dot_358_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_14_0), std::move(vit_Dot_358_0), std::move(vit_Add_360_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_361_0), std::move(vit_Reshape_362_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_362_0), std::move(vit_Multiply_365_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_368_0), std::move(vit_Broadcast_369_0), std::move(vit_BatchMatMul_370_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_371_0), std::move(vit_Softmax_372_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_380_0), std::move(vit_Broadcast_381_0), std::move(vit_BatchMatMul_382_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_383_0), std::move(vit_Reshape_384_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_385_0), std::move(vit_Constant_135_0), std::move(vit_Dot_386_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_17_0), std::move(vit_Dot_386_0), std::move(vit_Add_323_0), std::move(vit_Add_389_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_389_0), std::move(vit_Sum_390_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_391_0), std::move(vit_Sum_390_0), std::move(vit_Divide_393_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_395_0), std::move(vit_Add_389_0), std::move(vit_Subtract_397_0), std::move(vit_Power_399_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_399_0), std::move(vit_Sum_400_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_401_0), std::move(vit_Sum_400_0), std::move(vit_Sqrt_408_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_23_0), std::move(vit_Constant_22_0), std::move(vit_Reshape_409_0), std::move(vit_Subtract_397_0), std::move(vit_Add_415_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_415_0), std::move(vit_Constant_136_0), std::move(vit_Dot_416_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_18_0), std::move(vit_Dot_416_0), std::move(vit_Multiply_426_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_426_0), std::move(vit_Constant_137_0), std::move(vit_Dot_427_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_19_0), std::move(vit_Dot_427_0), std::move(vit_Add_389_0), std::move(vit_Add_430_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_430_0), std::move(vit_Sum_431_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_432_0), std::move(vit_Sum_431_0), std::move(vit_Divide_434_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_436_0), std::move(vit_Add_430_0), std::move(vit_Subtract_438_0), std::move(vit_Power_440_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_440_0), std::move(vit_Sum_441_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_442_0), std::move(vit_Sum_441_0), std::move(vit_Sqrt_449_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_31_0), std::move(vit_Constant_30_0), std::move(vit_Reshape_450_0), std::move(vit_Subtract_438_0), std::move(vit_Add_456_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_456_0), std::move(vit_Constant_140_0), std::move(vit_Dot_480_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_26_0), std::move(vit_Dot_480_0), std::move(vit_Add_482_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_483_0), std::move(vit_Reshape_484_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_456_0), std::move(vit_Constant_139_0), std::move(vit_Dot_457_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_25_0), std::move(vit_Dot_457_0), std::move(vit_Add_459_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_460_0), std::move(vit_Reshape_461_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_461_0), std::move(vit_Multiply_464_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_456_0), std::move(vit_Constant_138_0), std::move(vit_Dot_465_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_24_0), std::move(vit_Dot_465_0), std::move(vit_Add_467_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_468_0), std::move(vit_Reshape_469_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_469_0), std::move(vit_Multiply_472_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_475_0), std::move(vit_Broadcast_476_0), std::move(vit_BatchMatMul_477_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_478_0), std::move(vit_Softmax_479_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_487_0), std::move(vit_Broadcast_488_0), std::move(vit_BatchMatMul_489_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_490_0), std::move(vit_Reshape_491_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_492_0), std::move(vit_Constant_141_0), std::move(vit_Dot_493_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_27_0), std::move(vit_Dot_493_0), std::move(vit_Add_430_0), std::move(vit_Add_496_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_496_0), std::move(vit_Sum_497_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_498_0), std::move(vit_Sum_497_0), std::move(vit_Divide_500_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_502_0), std::move(vit_Add_496_0), std::move(vit_Subtract_504_0), std::move(vit_Power_506_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_506_0), std::move(vit_Sum_507_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_508_0), std::move(vit_Sum_507_0), std::move(vit_Sqrt_515_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_33_0), std::move(vit_Constant_32_0), std::move(vit_Reshape_516_0), std::move(vit_Subtract_504_0), std::move(vit_Add_522_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_522_0), std::move(vit_Constant_142_0), std::move(vit_Dot_523_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_28_0), std::move(vit_Dot_523_0), std::move(vit_Multiply_533_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_533_0), std::move(vit_Constant_143_0), std::move(vit_Dot_534_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_29_0), std::move(vit_Dot_534_0), std::move(vit_Add_496_0), std::move(vit_Add_537_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_537_0), std::move(vit_Sum_538_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_539_0), std::move(vit_Sum_538_0), std::move(vit_Divide_541_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_543_0), std::move(vit_Add_537_0), std::move(vit_Subtract_545_0), std::move(vit_Power_547_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_547_0), std::move(vit_Sum_548_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_549_0), std::move(vit_Sum_548_0), std::move(vit_Sqrt_556_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_41_0), std::move(vit_Constant_40_0), std::move(vit_Reshape_557_0), std::move(vit_Subtract_545_0), std::move(vit_Add_563_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_563_0), std::move(vit_Constant_146_0), std::move(vit_Dot_587_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_36_0), std::move(vit_Dot_587_0), std::move(vit_Add_589_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_590_0), std::move(vit_Reshape_591_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_563_0), std::move(vit_Constant_145_0), std::move(vit_Dot_564_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_35_0), std::move(vit_Dot_564_0), std::move(vit_Add_566_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_567_0), std::move(vit_Reshape_568_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_568_0), std::move(vit_Multiply_571_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_563_0), std::move(vit_Constant_144_0), std::move(vit_Dot_572_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_34_0), std::move(vit_Dot_572_0), std::move(vit_Add_574_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_575_0), std::move(vit_Reshape_576_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_576_0), std::move(vit_Multiply_579_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_582_0), std::move(vit_Broadcast_583_0), std::move(vit_BatchMatMul_584_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_585_0), std::move(vit_Softmax_586_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_594_0), std::move(vit_Broadcast_595_0), std::move(vit_BatchMatMul_596_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_597_0), std::move(vit_Reshape_598_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_599_0), std::move(vit_Constant_147_0), std::move(vit_Dot_600_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_37_0), std::move(vit_Dot_600_0), std::move(vit_Add_537_0), std::move(vit_Add_603_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_603_0), std::move(vit_Sum_604_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_605_0), std::move(vit_Sum_604_0), std::move(vit_Divide_607_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_609_0), std::move(vit_Add_603_0), std::move(vit_Subtract_611_0), std::move(vit_Power_613_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_613_0), std::move(vit_Sum_614_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_615_0), std::move(vit_Sum_614_0), std::move(vit_Sqrt_622_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_43_0), std::move(vit_Constant_42_0), std::move(vit_Reshape_623_0), std::move(vit_Subtract_611_0), std::move(vit_Add_629_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_629_0), std::move(vit_Constant_148_0), std::move(vit_Dot_630_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_38_0), std::move(vit_Dot_630_0), std::move(vit_Multiply_640_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_640_0), std::move(vit_Constant_149_0), std::move(vit_Dot_641_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_39_0), std::move(vit_Dot_641_0), std::move(vit_Add_603_0), std::move(vit_Add_644_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_644_0), std::move(vit_Sum_645_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_646_0), std::move(vit_Sum_645_0), std::move(vit_Divide_648_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_650_0), std::move(vit_Add_644_0), std::move(vit_Subtract_652_0), std::move(vit_Power_654_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_654_0), std::move(vit_Sum_655_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_656_0), std::move(vit_Sum_655_0), std::move(vit_Sqrt_663_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_51_0), std::move(vit_Constant_50_0), std::move(vit_Reshape_664_0), std::move(vit_Subtract_652_0), std::move(vit_Add_670_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_670_0), std::move(vit_Constant_152_0), std::move(vit_Dot_694_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_46_0), std::move(vit_Dot_694_0), std::move(vit_Add_696_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_697_0), std::move(vit_Reshape_698_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_670_0), std::move(vit_Constant_151_0), std::move(vit_Dot_671_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_45_0), std::move(vit_Dot_671_0), std::move(vit_Add_673_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_674_0), std::move(vit_Reshape_675_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_675_0), std::move(vit_Multiply_678_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_670_0), std::move(vit_Constant_150_0), std::move(vit_Dot_679_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_44_0), std::move(vit_Dot_679_0), std::move(vit_Add_681_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_682_0), std::move(vit_Reshape_683_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_683_0), std::move(vit_Multiply_686_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_689_0), std::move(vit_Broadcast_690_0), std::move(vit_BatchMatMul_691_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_692_0), std::move(vit_Softmax_693_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_701_0), std::move(vit_Broadcast_702_0), std::move(vit_BatchMatMul_703_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_704_0), std::move(vit_Reshape_705_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_706_0), std::move(vit_Constant_153_0), std::move(vit_Dot_707_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_47_0), std::move(vit_Dot_707_0), std::move(vit_Add_644_0), std::move(vit_Add_710_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_710_0), std::move(vit_Sum_711_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_712_0), std::move(vit_Sum_711_0), std::move(vit_Divide_714_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_716_0), std::move(vit_Add_710_0), std::move(vit_Subtract_718_0), std::move(vit_Power_720_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_720_0), std::move(vit_Sum_721_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_722_0), std::move(vit_Sum_721_0), std::move(vit_Sqrt_729_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_53_0), std::move(vit_Constant_52_0), std::move(vit_Reshape_730_0), std::move(vit_Subtract_718_0), std::move(vit_Add_736_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_736_0), std::move(vit_Constant_154_0), std::move(vit_Dot_737_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_48_0), std::move(vit_Dot_737_0), std::move(vit_Multiply_747_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_747_0), std::move(vit_Constant_155_0), std::move(vit_Dot_748_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_49_0), std::move(vit_Dot_748_0), std::move(vit_Add_710_0), std::move(vit_Add_751_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_751_0), std::move(vit_Sum_752_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_753_0), std::move(vit_Sum_752_0), std::move(vit_Divide_755_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_757_0), std::move(vit_Add_751_0), std::move(vit_Subtract_759_0), std::move(vit_Power_761_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_761_0), std::move(vit_Sum_762_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_763_0), std::move(vit_Sum_762_0), std::move(vit_Sqrt_770_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_61_0), std::move(vit_Constant_60_0), std::move(vit_Reshape_771_0), std::move(vit_Subtract_759_0), std::move(vit_Add_777_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_777_0), std::move(vit_Constant_158_0), std::move(vit_Dot_801_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_56_0), std::move(vit_Dot_801_0), std::move(vit_Add_803_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_804_0), std::move(vit_Reshape_805_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_777_0), std::move(vit_Constant_157_0), std::move(vit_Dot_778_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_55_0), std::move(vit_Dot_778_0), std::move(vit_Add_780_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_781_0), std::move(vit_Reshape_782_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_782_0), std::move(vit_Multiply_785_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_777_0), std::move(vit_Constant_156_0), std::move(vit_Dot_786_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_54_0), std::move(vit_Dot_786_0), std::move(vit_Add_788_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_789_0), std::move(vit_Reshape_790_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_790_0), std::move(vit_Multiply_793_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_796_0), std::move(vit_Broadcast_797_0), std::move(vit_BatchMatMul_798_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_799_0), std::move(vit_Softmax_800_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_808_0), std::move(vit_Broadcast_809_0), std::move(vit_BatchMatMul_810_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_811_0), std::move(vit_Reshape_812_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_813_0), std::move(vit_Constant_159_0), std::move(vit_Dot_814_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_57_0), std::move(vit_Dot_814_0), std::move(vit_Add_751_0), std::move(vit_Add_817_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_817_0), std::move(vit_Sum_818_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_819_0), std::move(vit_Sum_818_0), std::move(vit_Divide_821_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_823_0), std::move(vit_Add_817_0), std::move(vit_Subtract_825_0), std::move(vit_Power_827_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_827_0), std::move(vit_Sum_828_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_829_0), std::move(vit_Sum_828_0), std::move(vit_Sqrt_836_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_63_0), std::move(vit_Constant_62_0), std::move(vit_Reshape_837_0), std::move(vit_Subtract_825_0), std::move(vit_Add_843_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_843_0), std::move(vit_Constant_160_0), std::move(vit_Dot_844_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_58_0), std::move(vit_Dot_844_0), std::move(vit_Multiply_854_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_854_0), std::move(vit_Constant_161_0), std::move(vit_Dot_855_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_59_0), std::move(vit_Dot_855_0), std::move(vit_Add_817_0), std::move(vit_Add_858_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_858_0), std::move(vit_Sum_859_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_860_0), std::move(vit_Sum_859_0), std::move(vit_Divide_862_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_864_0), std::move(vit_Add_858_0), std::move(vit_Subtract_866_0), std::move(vit_Power_868_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_868_0), std::move(vit_Sum_869_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_870_0), std::move(vit_Sum_869_0), std::move(vit_Sqrt_877_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_71_0), std::move(vit_Constant_70_0), std::move(vit_Reshape_878_0), std::move(vit_Subtract_866_0), std::move(vit_Add_884_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_884_0), std::move(vit_Constant_164_0), std::move(vit_Dot_908_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_66_0), std::move(vit_Dot_908_0), std::move(vit_Add_910_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_911_0), std::move(vit_Reshape_912_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_884_0), std::move(vit_Constant_163_0), std::move(vit_Dot_885_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_65_0), std::move(vit_Dot_885_0), std::move(vit_Add_887_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_888_0), std::move(vit_Reshape_889_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_889_0), std::move(vit_Multiply_892_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_884_0), std::move(vit_Constant_162_0), std::move(vit_Dot_893_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_64_0), std::move(vit_Dot_893_0), std::move(vit_Add_895_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_896_0), std::move(vit_Reshape_897_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_897_0), std::move(vit_Multiply_900_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_903_0), std::move(vit_Broadcast_904_0), std::move(vit_BatchMatMul_905_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_906_0), std::move(vit_Softmax_907_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_915_0), std::move(vit_Broadcast_916_0), std::move(vit_BatchMatMul_917_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_918_0), std::move(vit_Reshape_919_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_920_0), std::move(vit_Constant_165_0), std::move(vit_Dot_921_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_67_0), std::move(vit_Dot_921_0), std::move(vit_Add_858_0), std::move(vit_Add_924_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_924_0), std::move(vit_Sum_925_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_926_0), std::move(vit_Sum_925_0), std::move(vit_Divide_928_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_930_0), std::move(vit_Add_924_0), std::move(vit_Subtract_932_0), std::move(vit_Power_934_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_934_0), std::move(vit_Sum_935_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_936_0), std::move(vit_Sum_935_0), std::move(vit_Sqrt_943_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_73_0), std::move(vit_Constant_72_0), std::move(vit_Reshape_944_0), std::move(vit_Subtract_932_0), std::move(vit_Add_950_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_950_0), std::move(vit_Constant_166_0), std::move(vit_Dot_951_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_68_0), std::move(vit_Dot_951_0), std::move(vit_Multiply_961_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_961_0), std::move(vit_Constant_167_0), std::move(vit_Dot_962_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_69_0), std::move(vit_Dot_962_0), std::move(vit_Add_924_0), std::move(vit_Add_965_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_965_0), std::move(vit_Sum_966_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_967_0), std::move(vit_Sum_966_0), std::move(vit_Divide_969_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_971_0), std::move(vit_Add_965_0), std::move(vit_Subtract_973_0), std::move(vit_Power_975_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_975_0), std::move(vit_Sum_976_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_977_0), std::move(vit_Sum_976_0), std::move(vit_Sqrt_984_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_81_0), std::move(vit_Constant_80_0), std::move(vit_Reshape_985_0), std::move(vit_Subtract_973_0), std::move(vit_Add_991_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_991_0), std::move(vit_Constant_170_0), std::move(vit_Dot_1015_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_76_0), std::move(vit_Dot_1015_0), std::move(vit_Add_1017_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1018_0), std::move(vit_Reshape_1019_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_991_0), std::move(vit_Constant_169_0), std::move(vit_Dot_992_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_75_0), std::move(vit_Dot_992_0), std::move(vit_Add_994_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_995_0), std::move(vit_Reshape_996_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_996_0), std::move(vit_Multiply_999_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_991_0), std::move(vit_Constant_168_0), std::move(vit_Dot_1000_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_74_0), std::move(vit_Dot_1000_0), std::move(vit_Add_1002_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1003_0), std::move(vit_Reshape_1004_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1004_0), std::move(vit_Multiply_1007_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1010_0), std::move(vit_Broadcast_1011_0), std::move(vit_BatchMatMul_1012_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_1013_0), std::move(vit_Softmax_1014_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1022_0), std::move(vit_Broadcast_1023_0), std::move(vit_BatchMatMul_1024_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1025_0), std::move(vit_Reshape_1026_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_1027_0), std::move(vit_Constant_171_0), std::move(vit_Dot_1028_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_77_0), std::move(vit_Dot_1028_0), std::move(vit_Add_965_0), std::move(vit_Add_1031_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1031_0), std::move(vit_Sum_1032_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1033_0), std::move(vit_Sum_1032_0), std::move(vit_Divide_1035_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1037_0), std::move(vit_Add_1031_0), std::move(vit_Subtract_1039_0), std::move(vit_Power_1041_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1041_0), std::move(vit_Sum_1042_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1043_0), std::move(vit_Sum_1042_0), std::move(vit_Sqrt_1050_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_83_0), std::move(vit_Constant_82_0), std::move(vit_Reshape_1051_0), std::move(vit_Subtract_1039_0), std::move(vit_Add_1057_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1057_0), std::move(vit_Constant_172_0), std::move(vit_Dot_1058_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_78_0), std::move(vit_Dot_1058_0), std::move(vit_Multiply_1068_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_1068_0), std::move(vit_Constant_173_0), std::move(vit_Dot_1069_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_79_0), std::move(vit_Dot_1069_0), std::move(vit_Add_1031_0), std::move(vit_Add_1072_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1072_0), std::move(vit_Sum_1073_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1074_0), std::move(vit_Sum_1073_0), std::move(vit_Divide_1076_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1078_0), std::move(vit_Add_1072_0), std::move(vit_Subtract_1080_0), std::move(vit_Power_1082_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1082_0), std::move(vit_Sum_1083_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1084_0), std::move(vit_Sum_1083_0), std::move(vit_Sqrt_1091_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_91_0), std::move(vit_Constant_90_0), std::move(vit_Reshape_1092_0), std::move(vit_Subtract_1080_0), std::move(vit_Add_1098_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1098_0), std::move(vit_Constant_176_0), std::move(vit_Dot_1122_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_86_0), std::move(vit_Dot_1122_0), std::move(vit_Add_1124_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1125_0), std::move(vit_Reshape_1126_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1098_0), std::move(vit_Constant_175_0), std::move(vit_Dot_1099_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_85_0), std::move(vit_Dot_1099_0), std::move(vit_Add_1101_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_1102_0), std::move(vit_Reshape_1103_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1103_0), std::move(vit_Multiply_1106_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1098_0), std::move(vit_Constant_174_0), std::move(vit_Dot_1107_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_84_0), std::move(vit_Dot_1107_0), std::move(vit_Add_1109_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1110_0), std::move(vit_Reshape_1111_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1111_0), std::move(vit_Multiply_1114_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1117_0), std::move(vit_Broadcast_1118_0), std::move(vit_BatchMatMul_1119_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_1120_0), std::move(vit_Softmax_1121_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1129_0), std::move(vit_Broadcast_1130_0), std::move(vit_BatchMatMul_1131_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1132_0), std::move(vit_Reshape_1133_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_1134_0), std::move(vit_Constant_177_0), std::move(vit_Dot_1135_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_87_0), std::move(vit_Dot_1135_0), std::move(vit_Add_1072_0), std::move(vit_Add_1138_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1138_0), std::move(vit_Sum_1139_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1140_0), std::move(vit_Sum_1139_0), std::move(vit_Divide_1142_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1144_0), std::move(vit_Add_1138_0), std::move(vit_Subtract_1146_0), std::move(vit_Power_1148_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1148_0), std::move(vit_Sum_1149_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1150_0), std::move(vit_Sum_1149_0), std::move(vit_Sqrt_1157_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_93_0), std::move(vit_Constant_92_0), std::move(vit_Reshape_1158_0), std::move(vit_Subtract_1146_0), std::move(vit_Add_1164_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1164_0), std::move(vit_Constant_178_0), std::move(vit_Dot_1165_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_88_0), std::move(vit_Dot_1165_0), std::move(vit_Multiply_1175_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_1175_0), std::move(vit_Constant_179_0), std::move(vit_Dot_1176_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_89_0), std::move(vit_Dot_1176_0), std::move(vit_Add_1138_0), std::move(vit_Add_1179_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1179_0), std::move(vit_Sum_1180_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1181_0), std::move(vit_Sum_1180_0), std::move(vit_Divide_1183_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1185_0), std::move(vit_Add_1179_0), std::move(vit_Subtract_1187_0), std::move(vit_Power_1189_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1189_0), std::move(vit_Sum_1190_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1191_0), std::move(vit_Sum_1190_0), std::move(vit_Sqrt_1198_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_101_0), std::move(vit_Constant_100_0), std::move(vit_Reshape_1199_0), std::move(vit_Subtract_1187_0), std::move(vit_Add_1205_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1205_0), std::move(vit_Constant_182_0), std::move(vit_Dot_1229_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_96_0), std::move(vit_Dot_1229_0), std::move(vit_Add_1231_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1232_0), std::move(vit_Reshape_1233_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1205_0), std::move(vit_Constant_181_0), std::move(vit_Dot_1206_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_95_0), std::move(vit_Dot_1206_0), std::move(vit_Add_1208_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_1209_0), std::move(vit_Reshape_1210_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1210_0), std::move(vit_Multiply_1213_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1205_0), std::move(vit_Constant_180_0), std::move(vit_Dot_1214_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_94_0), std::move(vit_Dot_1214_0), std::move(vit_Add_1216_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1217_0), std::move(vit_Reshape_1218_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1218_0), std::move(vit_Multiply_1221_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1224_0), std::move(vit_Broadcast_1225_0), std::move(vit_BatchMatMul_1226_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_1227_0), std::move(vit_Softmax_1228_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1236_0), std::move(vit_Broadcast_1237_0), std::move(vit_BatchMatMul_1238_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1239_0), std::move(vit_Reshape_1240_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_1241_0), std::move(vit_Constant_183_0), std::move(vit_Dot_1242_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_97_0), std::move(vit_Dot_1242_0), std::move(vit_Add_1179_0), std::move(vit_Add_1245_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1245_0), std::move(vit_Sum_1246_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1247_0), std::move(vit_Sum_1246_0), std::move(vit_Divide_1249_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1251_0), std::move(vit_Add_1245_0), std::move(vit_Subtract_1253_0), std::move(vit_Power_1255_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1255_0), std::move(vit_Sum_1256_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1257_0), std::move(vit_Sum_1256_0), std::move(vit_Sqrt_1264_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_103_0), std::move(vit_Constant_102_0), std::move(vit_Reshape_1265_0), std::move(vit_Subtract_1253_0), std::move(vit_Add_1271_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1271_0), std::move(vit_Constant_184_0), std::move(vit_Dot_1272_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_98_0), std::move(vit_Dot_1272_0), std::move(vit_Multiply_1282_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_1282_0), std::move(vit_Constant_185_0), std::move(vit_Dot_1283_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_99_0), std::move(vit_Dot_1283_0), std::move(vit_Add_1245_0), std::move(vit_Add_1286_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1286_0), std::move(vit_Sum_1287_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1288_0), std::move(vit_Sum_1287_0), std::move(vit_Divide_1290_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1292_0), std::move(vit_Add_1286_0), std::move(vit_Subtract_1294_0), std::move(vit_Power_1296_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1296_0), std::move(vit_Sum_1297_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1298_0), std::move(vit_Sum_1297_0), std::move(vit_Sqrt_1305_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_111_0), std::move(vit_Constant_110_0), std::move(vit_Reshape_1306_0), std::move(vit_Subtract_1294_0), std::move(vit_Add_1312_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1312_0), std::move(vit_Constant_188_0), std::move(vit_Dot_1336_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_106_0), std::move(vit_Dot_1336_0), std::move(vit_Add_1338_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1339_0), std::move(vit_Reshape_1340_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1312_0), std::move(vit_Constant_187_0), std::move(vit_Dot_1313_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_105_0), std::move(vit_Dot_1313_0), std::move(vit_Add_1315_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_1316_0), std::move(vit_Reshape_1317_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1317_0), std::move(vit_Multiply_1320_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1312_0), std::move(vit_Constant_186_0), std::move(vit_Dot_1321_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_104_0), std::move(vit_Dot_1321_0), std::move(vit_Add_1323_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1324_0), std::move(vit_Reshape_1325_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1325_0), std::move(vit_Multiply_1328_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1331_0), std::move(vit_Broadcast_1332_0), std::move(vit_BatchMatMul_1333_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_1334_0), std::move(vit_Softmax_1335_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1343_0), std::move(vit_Broadcast_1344_0), std::move(vit_BatchMatMul_1345_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1346_0), std::move(vit_Reshape_1347_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_1348_0), std::move(vit_Constant_189_0), std::move(vit_Dot_1349_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_107_0), std::move(vit_Dot_1349_0), std::move(vit_Add_1286_0), std::move(vit_Add_1352_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1352_0), std::move(vit_Sum_1353_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1354_0), std::move(vit_Sum_1353_0), std::move(vit_Divide_1356_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1358_0), std::move(vit_Add_1352_0), std::move(vit_Subtract_1360_0), std::move(vit_Power_1362_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1362_0), std::move(vit_Sum_1363_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1364_0), std::move(vit_Sum_1363_0), std::move(vit_Sqrt_1371_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_113_0), std::move(vit_Constant_112_0), std::move(vit_Reshape_1372_0), std::move(vit_Subtract_1360_0), std::move(vit_Add_1378_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1378_0), std::move(vit_Constant_190_0), std::move(vit_Dot_1379_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_108_0), std::move(vit_Dot_1379_0), std::move(vit_Multiply_1389_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_1389_0), std::move(vit_Constant_191_0), std::move(vit_Dot_1390_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_109_0), std::move(vit_Dot_1390_0), std::move(vit_Add_1352_0), std::move(vit_Add_1393_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1393_0), std::move(vit_Sum_1394_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1395_0), std::move(vit_Sum_1394_0), std::move(vit_Divide_1397_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1399_0), std::move(vit_Add_1393_0), std::move(vit_Subtract_1401_0), std::move(vit_Power_1403_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1403_0), std::move(vit_Sum_1404_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1405_0), std::move(vit_Sum_1404_0), std::move(vit_Sqrt_1412_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_121_0), std::move(vit_Constant_120_0), std::move(vit_Reshape_1413_0), std::move(vit_Subtract_1401_0), std::move(vit_Add_1419_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1419_0), std::move(vit_Constant_194_0), std::move(vit_Dot_1443_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_116_0), std::move(vit_Dot_1443_0), std::move(vit_Add_1445_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1446_0), std::move(vit_Reshape_1447_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1419_0), std::move(vit_Constant_193_0), std::move(vit_Dot_1420_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_115_0), std::move(vit_Dot_1420_0), std::move(vit_Add_1422_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_247_CallKernel(dim3(48, 13, 48), dim3(16, 16, 1), 0, nullptr, std::move(vit_Reshape_1423_0), std::move(vit_Reshape_1424_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1424_0), std::move(vit_Multiply_1427_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1419_0), std::move(vit_Constant_192_0), std::move(vit_Dot_1428_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Add_6_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_114_0), std::move(vit_Dot_1428_0), std::move(vit_Add_1430_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_270_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1431_0), std::move(vit_Reshape_1432_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Reshape_248_0), std::move(vit_Reshape_1432_0), std::move(vit_Multiply_1435_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1438_0), std::move(vit_Broadcast_1439_0), std::move(vit_BatchMatMul_1440_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Softmax_float_float_cuda_lib_Softmax_265Kernel(0, std::move(vit_Reshape_1441_0), std::move(vit_Softmax_1442_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275Kernel(std::move(vit_cublas_handle_0), std::move(vit_Broadcast_1450_0), std::move(vit_Broadcast_1451_0), std::move(vit_BatchMatMul_1452_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Reshape_float_float_cuda_Reshape_277_CallKernel(dim3(113472, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vit_Reshape_1453_0), std::move(vit_Reshape_1454_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_266Kernel(std::move(vit_cublas_handle_0), std::move(vit_Reshape_1455_0), std::move(vit_Constant_195_0), std::move(vit_Dot_1456_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_117_0), std::move(vit_Dot_1456_0), std::move(vit_Add_1393_0), std::move(vit_Add_1459_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1459_0), std::move(vit_Sum_1460_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1461_0), std::move(vit_Sum_1460_0), std::move(vit_Divide_1463_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1465_0), std::move(vit_Add_1459_0), std::move(vit_Subtract_1467_0), std::move(vit_Power_1469_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1469_0), std::move(vit_Sum_1470_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1471_0), std::move(vit_Sum_1470_0), std::move(vit_Sqrt_1478_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_123_0), std::move(vit_Constant_122_0), std::move(vit_Reshape_1479_0), std::move(vit_Subtract_1467_0), std::move(vit_Add_1485_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_309Kernel(std::move(vit_cublas_handle_0), std::move(vit_Add_1485_0), std::move(vit_Constant_196_0), std::move(vit_Dot_1486_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_CallKernel(dim3(56736, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_204_0), std::move(vit_Constant_203_0), std::move(vit_Constant_205_0), std::move(vit_Constant_118_0), std::move(vit_Dot_1486_0), std::move(vit_Multiply_1496_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Dot_float_float_float_cuda_lib_Dot_320Kernel(std::move(vit_cublas_handle_0), std::move(vit_Multiply_1496_0), std::move(vit_Constant_197_0), std::move(vit_Dot_1497_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_119_0), std::move(vit_Dot_1497_0), std::move(vit_Add_1459_0), std::move(vit_Add_1500_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Add_1500_0), std::move(vit_Sum_1501_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_cuda_Broadcast_Divide_0_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Constant_1502_0), std::move(vit_Sum_1501_0), std::move(vit_Divide_1504_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_206_0), std::move(vit_Reshape_1506_0), std::move(vit_Add_1500_0), std::move(vit_Subtract_1508_0), std::move(vit_Power_1510_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Sum_float_float_cuda_Sum_217_CallKernel(dim3(9456, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Power_1510_0), std::move(vit_Sum_1511_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_CallKernel(dim3(24, 1, 1), dim3(394, 1, 1), 0, nullptr, std::move(vit_Reshape_1516_0), std::move(vit_Constant_1512_0), std::move(vit_Sum_1511_0), std::move(vit_Sqrt_1519_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_CallKernel(dim3(14184, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vit_Constant_125_0), std::move(vit_Constant_124_0), std::move(vit_Reshape_1520_0), std::move(vit_Subtract_1508_0), std::move(vit_last_hidden_state), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
    kernels.emplace_back(new vit_Result_float_float_cuda_lib_Result_1527Kernel(std::move(vit_last_hidden_state), std::move(vit_Result_1527_0), std::move(Parameter_207_0), std::move(vit_Result_1527_0)));
}
