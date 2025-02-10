#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <fstream>
#include "nnfusion_rt.h"
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
cublasHandle_t cublas_handle_0;
cudnnHandle_t cudnn_handle_0;
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
int num_SMs;
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* Broadcast_215_0;
float* Broadcast_209_0;
float* Convolution_208_0;
float* Add_210_0;
float* Reshape_211_0;
float* Reshape_212_0;
float* Concat_213_0;
float* Add_216_0;
float* Sum_217_0;
float* Divide_220_0;
float* Reshape_221_0;
float* Reshape_222_0;
float* Power_226_0;
float* Subtract_224_0;
float* Sum_227_0;
float* Sqrt_235_0;
float* Reshape_236_0;
float* Add_242_0;
float* Dot_266_0;
float* Add_268_0;
float* Reshape_269_0;
float* Reshape_270_0;
float* Reshape_272_0;
float* Broadcast_274_0;
float* Dot_243_0;
float* Add_245_0;
float* Reshape_246_0;
float* Reshape_247_0;
float* Multiply_250_0;
float* Reshape_260_0;
float* Broadcast_262_0;
float* Dot_251_0;
float* Add_253_0;
float* Reshape_254_0;
float* Reshape_255_0;
float* Multiply_258_0;
float* Reshape_259_0;
float* Broadcast_261_0;
float* BatchMatMul_263_0;
float* Reshape_264_0;
float* Softmax_265_0;
float* Reshape_271_0;
float* Broadcast_273_0;
float* BatchMatMul_275_0;
float* Reshape_276_0;
float* Reshape_277_0;
float* Reshape_278_0;
float* Dot_279_0;
float* Add_282_0;
float* Sum_283_0;
float* Divide_286_0;
float* Reshape_287_0;
float* Reshape_288_0;
float* Power_292_0;
float* Subtract_290_0;
float* Sum_293_0;
float* Sqrt_301_0;
float* Reshape_302_0;
float* Add_308_0;
float* Dot_309_0;
float* Multiply_319_0;
float* Dot_320_0;
float* Add_323_0;
float* Sum_324_0;
float* Divide_327_0;
float* Reshape_328_0;
float* Reshape_329_0;
float* Power_333_0;
float* Subtract_331_0;
float* Sum_334_0;
float* Sqrt_342_0;
float* Reshape_343_0;
float* Add_349_0;
float* Dot_373_0;
float* Add_375_0;
float* Reshape_376_0;
float* Reshape_377_0;
float* Reshape_379_0;
float* Broadcast_381_0;
float* Dot_350_0;
float* Add_352_0;
float* Reshape_353_0;
float* Reshape_354_0;
float* Multiply_357_0;
float* Reshape_367_0;
float* Broadcast_369_0;
float* Dot_358_0;
float* Add_360_0;
float* Reshape_361_0;
float* Reshape_362_0;
float* Multiply_365_0;
float* Reshape_366_0;
float* Broadcast_368_0;
float* BatchMatMul_370_0;
float* Reshape_371_0;
float* Softmax_372_0;
float* Reshape_378_0;
float* Broadcast_380_0;
float* BatchMatMul_382_0;
float* Reshape_383_0;
float* Reshape_384_0;
float* Reshape_385_0;
float* Dot_386_0;
float* Add_389_0;
float* Sum_390_0;
float* Divide_393_0;
float* Reshape_394_0;
float* Reshape_395_0;
float* Power_399_0;
float* Subtract_397_0;
float* Sum_400_0;
float* Sqrt_408_0;
float* Reshape_409_0;
float* Add_415_0;
float* Dot_416_0;
float* Multiply_426_0;
float* Dot_427_0;
float* Add_430_0;
float* Sum_431_0;
float* Divide_434_0;
float* Reshape_435_0;
float* Reshape_436_0;
float* Power_440_0;
float* Subtract_438_0;
float* Sum_441_0;
float* Sqrt_449_0;
float* Reshape_450_0;
float* Add_456_0;
float* Dot_480_0;
float* Add_482_0;
float* Reshape_483_0;
float* Reshape_484_0;
float* Reshape_486_0;
float* Broadcast_488_0;
float* Dot_457_0;
float* Add_459_0;
float* Reshape_460_0;
float* Reshape_461_0;
float* Multiply_464_0;
float* Reshape_474_0;
float* Broadcast_476_0;
float* Dot_465_0;
float* Add_467_0;
float* Reshape_468_0;
float* Reshape_469_0;
float* Multiply_472_0;
float* Reshape_473_0;
float* Broadcast_475_0;
float* BatchMatMul_477_0;
float* Reshape_478_0;
float* Softmax_479_0;
float* Reshape_485_0;
float* Broadcast_487_0;
float* BatchMatMul_489_0;
float* Reshape_490_0;
float* Reshape_491_0;
float* Reshape_492_0;
float* Dot_493_0;
float* Add_496_0;
float* Sum_497_0;
float* Divide_500_0;
float* Reshape_501_0;
float* Reshape_502_0;
float* Power_506_0;
float* Subtract_504_0;
float* Sum_507_0;
float* Sqrt_515_0;
float* Reshape_516_0;
float* Add_522_0;
float* Dot_523_0;
float* Multiply_533_0;
float* Dot_534_0;
float* Add_537_0;
float* Sum_538_0;
float* Divide_541_0;
float* Reshape_542_0;
float* Reshape_543_0;
float* Power_547_0;
float* Subtract_545_0;
float* Sum_548_0;
float* Sqrt_556_0;
float* Reshape_557_0;
float* Add_563_0;
float* Dot_587_0;
float* Add_589_0;
float* Reshape_590_0;
float* Reshape_591_0;
float* Reshape_593_0;
float* Broadcast_595_0;
float* Dot_564_0;
float* Add_566_0;
float* Reshape_567_0;
float* Reshape_568_0;
float* Multiply_571_0;
float* Reshape_581_0;
float* Broadcast_583_0;
float* Dot_572_0;
float* Add_574_0;
float* Reshape_575_0;
float* Reshape_576_0;
float* Multiply_579_0;
float* Reshape_580_0;
float* Broadcast_582_0;
float* BatchMatMul_584_0;
float* Reshape_585_0;
float* Softmax_586_0;
float* Reshape_592_0;
float* Broadcast_594_0;
float* BatchMatMul_596_0;
float* Reshape_597_0;
float* Reshape_598_0;
float* Reshape_599_0;
float* Dot_600_0;
float* Add_603_0;
float* Sum_604_0;
float* Divide_607_0;
float* Reshape_608_0;
float* Reshape_609_0;
float* Power_613_0;
float* Subtract_611_0;
float* Sum_614_0;
float* Sqrt_622_0;
float* Reshape_623_0;
float* Add_629_0;
float* Dot_630_0;
float* Multiply_640_0;
float* Dot_641_0;
float* Add_644_0;
float* Sum_645_0;
float* Divide_648_0;
float* Reshape_649_0;
float* Reshape_650_0;
float* Power_654_0;
float* Subtract_652_0;
float* Sum_655_0;
float* Sqrt_663_0;
float* Reshape_664_0;
float* Add_670_0;
float* Dot_694_0;
float* Add_696_0;
float* Reshape_697_0;
float* Reshape_698_0;
float* Reshape_700_0;
float* Broadcast_702_0;
float* Dot_671_0;
float* Add_673_0;
float* Reshape_674_0;
float* Reshape_675_0;
float* Multiply_678_0;
float* Reshape_688_0;
float* Broadcast_690_0;
float* Dot_679_0;
float* Add_681_0;
float* Reshape_682_0;
float* Reshape_683_0;
float* Multiply_686_0;
float* Reshape_687_0;
float* Broadcast_689_0;
float* BatchMatMul_691_0;
float* Reshape_692_0;
float* Softmax_693_0;
float* Reshape_699_0;
float* Broadcast_701_0;
float* BatchMatMul_703_0;
float* Reshape_704_0;
float* Reshape_705_0;
float* Reshape_706_0;
float* Dot_707_0;
float* Add_710_0;
float* Sum_711_0;
float* Divide_714_0;
float* Reshape_715_0;
float* Reshape_716_0;
float* Power_720_0;
float* Subtract_718_0;
float* Sum_721_0;
float* Sqrt_729_0;
float* Reshape_730_0;
float* Add_736_0;
float* Dot_737_0;
float* Multiply_747_0;
float* Dot_748_0;
float* Add_751_0;
float* Sum_752_0;
float* Divide_755_0;
float* Reshape_756_0;
float* Reshape_757_0;
float* Power_761_0;
float* Subtract_759_0;
float* Sum_762_0;
float* Sqrt_770_0;
float* Reshape_771_0;
float* Add_777_0;
float* Dot_801_0;
float* Add_803_0;
float* Reshape_804_0;
float* Reshape_805_0;
float* Reshape_807_0;
float* Broadcast_809_0;
float* Dot_778_0;
float* Add_780_0;
float* Reshape_781_0;
float* Reshape_782_0;
float* Multiply_785_0;
float* Reshape_795_0;
float* Broadcast_797_0;
float* Dot_786_0;
float* Add_788_0;
float* Reshape_789_0;
float* Reshape_790_0;
float* Multiply_793_0;
float* Reshape_794_0;
float* Broadcast_796_0;
float* BatchMatMul_798_0;
float* Reshape_799_0;
float* Softmax_800_0;
float* Reshape_806_0;
float* Broadcast_808_0;
float* BatchMatMul_810_0;
float* Reshape_811_0;
float* Reshape_812_0;
float* Reshape_813_0;
float* Dot_814_0;
float* Add_817_0;
float* Sum_818_0;
float* Divide_821_0;
float* Reshape_822_0;
float* Reshape_823_0;
float* Power_827_0;
float* Subtract_825_0;
float* Sum_828_0;
float* Sqrt_836_0;
float* Reshape_837_0;
float* Add_843_0;
float* Dot_844_0;
float* Multiply_854_0;
float* Dot_855_0;
float* Add_858_0;
float* Sum_859_0;
float* Divide_862_0;
float* Reshape_863_0;
float* Reshape_864_0;
float* Power_868_0;
float* Subtract_866_0;
float* Sum_869_0;
float* Sqrt_877_0;
float* Reshape_878_0;
float* Add_884_0;
float* Dot_908_0;
float* Add_910_0;
float* Reshape_911_0;
float* Reshape_912_0;
float* Reshape_914_0;
float* Broadcast_916_0;
float* Dot_885_0;
float* Add_887_0;
float* Reshape_888_0;
float* Reshape_889_0;
float* Multiply_892_0;
float* Reshape_902_0;
float* Broadcast_904_0;
float* Dot_893_0;
float* Add_895_0;
float* Reshape_896_0;
float* Reshape_897_0;
float* Multiply_900_0;
float* Reshape_901_0;
float* Broadcast_903_0;
float* BatchMatMul_905_0;
float* Reshape_906_0;
float* Softmax_907_0;
float* Reshape_913_0;
float* Broadcast_915_0;
float* BatchMatMul_917_0;
float* Reshape_918_0;
float* Reshape_919_0;
float* Reshape_920_0;
float* Dot_921_0;
float* Add_924_0;
float* Sum_925_0;
float* Divide_928_0;
float* Reshape_929_0;
float* Reshape_930_0;
float* Power_934_0;
float* Subtract_932_0;
float* Sum_935_0;
float* Sqrt_943_0;
float* Reshape_944_0;
float* Add_950_0;
float* Dot_951_0;
float* Multiply_961_0;
float* Dot_962_0;
float* Add_965_0;
float* Sum_966_0;
float* Divide_969_0;
float* Reshape_970_0;
float* Reshape_971_0;
float* Power_975_0;
float* Subtract_973_0;
float* Sum_976_0;
float* Sqrt_984_0;
float* Reshape_985_0;
float* Add_991_0;
float* Dot_1015_0;
float* Add_1017_0;
float* Reshape_1018_0;
float* Reshape_1019_0;
float* Reshape_1021_0;
float* Broadcast_1023_0;
float* Dot_992_0;
float* Add_994_0;
float* Reshape_995_0;
float* Reshape_996_0;
float* Multiply_999_0;
float* Reshape_1009_0;
float* Broadcast_1011_0;
float* Dot_1000_0;
float* Add_1002_0;
float* Reshape_1003_0;
float* Reshape_1004_0;
float* Multiply_1007_0;
float* Reshape_1008_0;
float* Broadcast_1010_0;
float* BatchMatMul_1012_0;
float* Reshape_1013_0;
float* Softmax_1014_0;
float* Reshape_1020_0;
float* Broadcast_1022_0;
float* BatchMatMul_1024_0;
float* Reshape_1025_0;
float* Reshape_1026_0;
float* Reshape_1027_0;
float* Dot_1028_0;
float* Add_1031_0;
float* Sum_1032_0;
float* Divide_1035_0;
float* Reshape_1036_0;
float* Reshape_1037_0;
float* Power_1041_0;
float* Subtract_1039_0;
float* Sum_1042_0;
float* Sqrt_1050_0;
float* Reshape_1051_0;
float* Add_1057_0;
float* Dot_1058_0;
float* Multiply_1068_0;
float* Dot_1069_0;
float* Add_1072_0;
float* Sum_1073_0;
float* Divide_1076_0;
float* Reshape_1077_0;
float* Reshape_1078_0;
float* Power_1082_0;
float* Subtract_1080_0;
float* Sum_1083_0;
float* Sqrt_1091_0;
float* Reshape_1092_0;
float* Add_1098_0;
float* Dot_1122_0;
float* Add_1124_0;
float* Reshape_1125_0;
float* Reshape_1126_0;
float* Reshape_1128_0;
float* Broadcast_1130_0;
float* Dot_1099_0;
float* Add_1101_0;
float* Reshape_1102_0;
float* Reshape_1103_0;
float* Multiply_1106_0;
float* Reshape_1116_0;
float* Broadcast_1118_0;
float* Dot_1107_0;
float* Add_1109_0;
float* Reshape_1110_0;
float* Reshape_1111_0;
float* Multiply_1114_0;
float* Reshape_1115_0;
float* Broadcast_1117_0;
float* BatchMatMul_1119_0;
float* Reshape_1120_0;
float* Softmax_1121_0;
float* Reshape_1127_0;
float* Broadcast_1129_0;
float* BatchMatMul_1131_0;
float* Reshape_1132_0;
float* Reshape_1133_0;
float* Reshape_1134_0;
float* Dot_1135_0;
float* Add_1138_0;
float* Sum_1139_0;
float* Divide_1142_0;
float* Reshape_1143_0;
float* Reshape_1144_0;
float* Power_1148_0;
float* Subtract_1146_0;
float* Sum_1149_0;
float* Sqrt_1157_0;
float* Reshape_1158_0;
float* Add_1164_0;
float* Dot_1165_0;
float* Multiply_1175_0;
float* Dot_1176_0;
float* Add_1179_0;
float* Sum_1180_0;
float* Divide_1183_0;
float* Reshape_1184_0;
float* Reshape_1185_0;
float* Power_1189_0;
float* Subtract_1187_0;
float* Sum_1190_0;
float* Sqrt_1198_0;
float* Reshape_1199_0;
float* Add_1205_0;
float* Dot_1229_0;
float* Add_1231_0;
float* Reshape_1232_0;
float* Reshape_1233_0;
float* Reshape_1235_0;
float* Broadcast_1237_0;
float* Dot_1206_0;
float* Add_1208_0;
float* Reshape_1209_0;
float* Reshape_1210_0;
float* Multiply_1213_0;
float* Reshape_1223_0;
float* Broadcast_1225_0;
float* Dot_1214_0;
float* Add_1216_0;
float* Reshape_1217_0;
float* Reshape_1218_0;
float* Multiply_1221_0;
float* Reshape_1222_0;
float* Broadcast_1224_0;
float* BatchMatMul_1226_0;
float* Reshape_1227_0;
float* Softmax_1228_0;
float* Reshape_1234_0;
float* Broadcast_1236_0;
float* BatchMatMul_1238_0;
float* Reshape_1239_0;
float* Reshape_1240_0;
float* Reshape_1241_0;
float* Dot_1242_0;
float* Add_1245_0;
float* Sum_1246_0;
float* Divide_1249_0;
float* Reshape_1250_0;
float* Reshape_1251_0;
float* Power_1255_0;
float* Subtract_1253_0;
float* Sum_1256_0;
float* Sqrt_1264_0;
float* Reshape_1265_0;
float* Add_1271_0;
float* Dot_1272_0;
float* Multiply_1282_0;
float* Dot_1283_0;
float* Add_1286_0;
float* Sum_1287_0;
float* Divide_1290_0;
float* Reshape_1291_0;
float* Reshape_1292_0;
float* Power_1296_0;
float* Subtract_1294_0;
float* Sum_1297_0;
float* Sqrt_1305_0;
float* Reshape_1306_0;
float* Add_1312_0;
float* Dot_1336_0;
float* Add_1338_0;
float* Reshape_1339_0;
float* Reshape_1340_0;
float* Reshape_1342_0;
float* Broadcast_1344_0;
float* Dot_1313_0;
float* Add_1315_0;
float* Reshape_1316_0;
float* Reshape_1317_0;
float* Multiply_1320_0;
float* Reshape_1330_0;
float* Broadcast_1332_0;
float* Dot_1321_0;
float* Add_1323_0;
float* Reshape_1324_0;
float* Reshape_1325_0;
float* Multiply_1328_0;
float* Reshape_1329_0;
float* Broadcast_1331_0;
float* BatchMatMul_1333_0;
float* Reshape_1334_0;
float* Softmax_1335_0;
float* Reshape_1341_0;
float* Broadcast_1343_0;
float* BatchMatMul_1345_0;
float* Reshape_1346_0;
float* Reshape_1347_0;
float* Reshape_1348_0;
float* Dot_1349_0;
float* Add_1352_0;
float* Sum_1353_0;
float* Divide_1356_0;
float* Reshape_1357_0;
float* Reshape_1358_0;
float* Power_1362_0;
float* Subtract_1360_0;
float* Sum_1363_0;
float* Sqrt_1371_0;
float* Reshape_1372_0;
float* Add_1378_0;
float* Dot_1379_0;
float* Multiply_1389_0;
float* Dot_1390_0;
float* Add_1393_0;
float* Sum_1394_0;
float* Divide_1397_0;
float* Reshape_1398_0;
float* Reshape_1399_0;
float* Power_1403_0;
float* Subtract_1401_0;
float* Sum_1404_0;
float* Sqrt_1412_0;
float* Reshape_1413_0;
float* Add_1419_0;
float* Dot_1443_0;
float* Add_1445_0;
float* Reshape_1446_0;
float* Reshape_1447_0;
float* Reshape_1449_0;
float* Broadcast_1451_0;
float* Dot_1420_0;
float* Add_1422_0;
float* Reshape_1423_0;
float* Reshape_1424_0;
float* Multiply_1427_0;
float* Reshape_1437_0;
float* Broadcast_1439_0;
float* Dot_1428_0;
float* Add_1430_0;
float* Reshape_1431_0;
float* Reshape_1432_0;
float* Multiply_1435_0;
float* Reshape_1436_0;
float* Broadcast_1438_0;
float* BatchMatMul_1440_0;
float* Reshape_1441_0;
float* Softmax_1442_0;
float* Reshape_1448_0;
float* Broadcast_1450_0;
float* BatchMatMul_1452_0;
float* Reshape_1453_0;
float* Reshape_1454_0;
float* Reshape_1455_0;
float* Dot_1456_0;
float* Add_1459_0;
float* Sum_1460_0;
float* Divide_1463_0;
float* Reshape_1464_0;
float* Reshape_1465_0;
float* Power_1469_0;
float* Subtract_1467_0;
float* Sum_1470_0;
float* Sqrt_1478_0;
float* Reshape_1479_0;
float* Add_1485_0;
float* Dot_1486_0;
float* Multiply_1496_0;
float* Dot_1497_0;
float* Add_1500_0;
float* Sum_1501_0;
float* Divide_1504_0;
float* Reshape_1505_0;
float* Reshape_1506_0;
float* Power_1510_0;
float* Subtract_1508_0;
float* Sum_1511_0;
float* Sqrt_1519_0;
float* Reshape_1520_0;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_1_0;
float* Reshape_214_0;
float* Constant_3_0;
float* Constant_2_0;
float* Constant_200_0;
float* Constant_129_0;
float* Constant_128_0;
float* Constant_218_0;
float* Constant_206_0;
float* Constant_228_0;
float* Constant_202_0;
float* Reshape_1516_0;
float* Constant_10_0;
float* Constant_11_0;
float* Constant_6_0;
float* Constant_127_0;
float* Constant_5_0;
float* Constant_201_0;
float* Reshape_248_0;
float* Constant_126_0;
float* Constant_4_0;
float* Constant_7_0;
float* Constant_131_0;
float* Constant_130_0;
float* Constant_284_0;
float* Constant_294_0;
float* Constant_12_0;
float* Constant_13_0;
float* Constant_8_0;
float* Constant_205_0;
float* Constant_203_0;
float* Constant_204_0;
float* Constant_9_0;
float* Constant_135_0;
float* Constant_134_0;
float* Constant_325_0;
float* Constant_335_0;
float* Constant_20_0;
float* Constant_21_0;
float* Constant_16_0;
float* Constant_133_0;
float* Constant_15_0;
float* Constant_132_0;
float* Constant_14_0;
float* Constant_17_0;
float* Constant_137_0;
float* Constant_136_0;
float* Constant_391_0;
float* Constant_401_0;
float* Constant_22_0;
float* Constant_23_0;
float* Constant_18_0;
float* Constant_19_0;
float* Constant_141_0;
float* Constant_140_0;
float* Constant_432_0;
float* Constant_442_0;
float* Constant_30_0;
float* Constant_31_0;
float* Constant_26_0;
float* Constant_139_0;
float* Constant_25_0;
float* Constant_138_0;
float* Constant_24_0;
float* Constant_27_0;
float* Constant_143_0;
float* Constant_142_0;
float* Constant_498_0;
float* Constant_508_0;
float* Constant_32_0;
float* Constant_33_0;
float* Constant_28_0;
float* Constant_29_0;
float* Constant_147_0;
float* Constant_146_0;
float* Constant_539_0;
float* Constant_549_0;
float* Constant_40_0;
float* Constant_41_0;
float* Constant_36_0;
float* Constant_145_0;
float* Constant_35_0;
float* Constant_144_0;
float* Constant_34_0;
float* Constant_37_0;
float* Constant_149_0;
float* Constant_148_0;
float* Constant_605_0;
float* Constant_615_0;
float* Constant_42_0;
float* Constant_43_0;
float* Constant_38_0;
float* Constant_39_0;
float* Constant_153_0;
float* Constant_152_0;
float* Constant_646_0;
float* Constant_656_0;
float* Constant_50_0;
float* Constant_51_0;
float* Constant_46_0;
float* Constant_151_0;
float* Constant_45_0;
float* Constant_150_0;
float* Constant_44_0;
float* Constant_47_0;
float* Constant_155_0;
float* Constant_154_0;
float* Constant_712_0;
float* Constant_722_0;
float* Constant_52_0;
float* Constant_53_0;
float* Constant_48_0;
float* Constant_49_0;
float* Constant_159_0;
float* Constant_158_0;
float* Constant_753_0;
float* Constant_763_0;
float* Constant_60_0;
float* Constant_61_0;
float* Constant_56_0;
float* Constant_157_0;
float* Constant_55_0;
float* Constant_156_0;
float* Constant_54_0;
float* Constant_57_0;
float* Constant_161_0;
float* Constant_160_0;
float* Constant_819_0;
float* Constant_829_0;
float* Constant_62_0;
float* Constant_63_0;
float* Constant_58_0;
float* Constant_59_0;
float* Constant_165_0;
float* Constant_164_0;
float* Constant_860_0;
float* Constant_870_0;
float* Constant_70_0;
float* Constant_71_0;
float* Constant_66_0;
float* Constant_163_0;
float* Constant_65_0;
float* Constant_162_0;
float* Constant_64_0;
float* Constant_67_0;
float* Constant_167_0;
float* Constant_166_0;
float* Constant_926_0;
float* Constant_936_0;
float* Constant_72_0;
float* Constant_73_0;
float* Constant_68_0;
float* Constant_69_0;
float* Constant_171_0;
float* Constant_170_0;
float* Constant_967_0;
float* Constant_977_0;
float* Constant_80_0;
float* Constant_81_0;
float* Constant_76_0;
float* Constant_169_0;
float* Constant_75_0;
float* Constant_168_0;
float* Constant_74_0;
float* Constant_77_0;
float* Constant_173_0;
float* Constant_172_0;
float* Constant_1033_0;
float* Constant_1043_0;
float* Constant_82_0;
float* Constant_83_0;
float* Constant_78_0;
float* Constant_79_0;
float* Constant_177_0;
float* Constant_176_0;
float* Constant_1074_0;
float* Constant_1084_0;
float* Constant_90_0;
float* Constant_91_0;
float* Constant_86_0;
float* Constant_175_0;
float* Constant_85_0;
float* Constant_174_0;
float* Constant_84_0;
float* Constant_87_0;
float* Constant_179_0;
float* Constant_178_0;
float* Constant_1140_0;
float* Constant_1150_0;
float* Constant_92_0;
float* Constant_93_0;
float* Constant_88_0;
float* Constant_89_0;
float* Constant_183_0;
float* Constant_182_0;
float* Constant_1181_0;
float* Constant_1191_0;
float* Constant_100_0;
float* Constant_101_0;
float* Constant_96_0;
float* Constant_181_0;
float* Constant_95_0;
float* Constant_180_0;
float* Constant_94_0;
float* Constant_97_0;
float* Constant_185_0;
float* Constant_184_0;
float* Constant_1247_0;
float* Constant_1257_0;
float* Constant_102_0;
float* Constant_103_0;
float* Constant_98_0;
float* Constant_99_0;
float* Constant_189_0;
float* Constant_188_0;
float* Constant_1288_0;
float* Constant_1298_0;
float* Constant_110_0;
float* Constant_111_0;
float* Constant_106_0;
float* Constant_187_0;
float* Constant_105_0;
float* Constant_186_0;
float* Constant_104_0;
float* Constant_107_0;
float* Constant_191_0;
float* Constant_190_0;
float* Constant_1354_0;
float* Constant_1364_0;
float* Constant_112_0;
float* Constant_113_0;
float* Constant_108_0;
float* Constant_109_0;
float* Constant_195_0;
float* Constant_194_0;
float* Constant_1395_0;
float* Constant_1405_0;
float* Constant_120_0;
float* Constant_121_0;
float* Constant_116_0;
float* Constant_193_0;
float* Constant_115_0;
float* Constant_192_0;
float* Constant_114_0;
float* Constant_117_0;
float* Constant_197_0;
float* Constant_196_0;
float* Constant_1461_0;
float* Constant_1471_0;
float* Constant_122_0;
float* Constant_123_0;
float* Constant_118_0;
float* Constant_119_0;
float* Constant_1502_0;
float* Constant_1512_0;
float* Constant_124_0;
float* Constant_125_0;
float* last_hidden_state;
float* Result_1527_0;
int64_t get_workspace_size()
{
    return 725516288;
}
// Node name:	Constant_168
// Description:	Constant
// Input:
// Output:
//	- name: Constant_168_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_168(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_168_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_168_0 failed.\n");
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
//	- name: Constant_162_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_162_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_162_0 failed.\n");
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
//	- name: Constant_1395_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1395(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1395_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1395_0 failed.\n");
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
//	- name: Constant_91_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_91_0 failed.\n");
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
//	- name: Constant_109_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_109_0 failed.\n");
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
//	- name: Convolution_208_0	type: float	shape: Shape{48, 768, 14, 14}
//	- name: Broadcast_209_0	type: float	shape: Shape{48, 768, 14, 14}
// Output:
//	- name: Add_210_0	type: float	shape: Shape{48, 768, 14, 14}
extern "C" __launch_bounds__(512) __global__ void Add_float_float_float_cuda_Add_210(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_210_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_210<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_270
// Description:	Reshape
// Input:
//	- name: Reshape_269_0	type: float	shape: Shape{48, 197, 12, 64}
// Output:
//	- name: Reshape_270_0	type: float	shape: Shape{48, 12, 197, 64}
extern "C" __launch_bounds__(64) __global__ void Reshape_float_float_cuda_Reshape_270(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_270_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_270<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_208
// Description:	Convolution
// Input:
//	- name: Parameter_207_0	type: float	shape: Shape{48, 3, 224, 224}
//	- name: Constant_2_0	type: float	shape: Shape{768, 3, 16, 16}
// Output:
//	- name: Convolution_208_0	type: float	shape: Shape{48, 768, 14, 14}
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

    static bool selected_algo = false;
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
//	- name: Constant_206_0	type: float	shape: Shape{}
//	- name: Reshape_222_0	type: float	shape: Shape{48, 197}
//	- name: Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Subtract_224_0	type: float	shape: Shape{48, 197, 768}
//	- name: Power_226_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_1528
// Broadcast, Broadcast_223
// Subtract, /encoder/layer.0/layernorm_before/Sub_output_0
// Power, /encoder/layer.0/layernorm_before/Pow_output_0
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid / 768];
    float temp2 = subtractf(input2[tid], temp1);
    float temp3 = powf(temp2, temp0);
    output1[tid] = temp3;
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	BatchMatMul_263
// Description:	BatchMatMul
// Input:
//	- name: Broadcast_261_0	type: float	shape: Shape{48, 12, 197, 64}
//	- name: Broadcast_262_0	type: float	shape: Shape{48, 12, 64, 197}
// Output:
//	- name: BatchMatMul_263_0	type: float	shape: Shape{48, 12, 197, 197}
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
//	- name: last_hidden_state	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Result_1527_0	type: float	shape: Shape{48, 197, 768}
void Result_float_float_cuda_lib_Result_1527(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Reshape_248_0	type: float	shape: Shape{}
//	- name: Reshape_247_0	type: float	shape: Shape{48, 12, 64, 197}
// Output:
//	- name: Multiply_250_0	type: float	shape: Shape{48, 12, 64, 197}
// Fused functions:
// Broadcast, Broadcast_1530
// Multiply, /encoder/layer.0/attention/attention/Mul_1_output_0
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Multiply_8(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = mul(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Multiply_8<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Softmax_265
// Description:	Softmax
// Input:
//	- name: Reshape_264_0	type: float	shape: Shape{48, 12, 197, 197}
// Output:
//	- name: Softmax_265_0	type: float	shape: Shape{48, 12, 197, 197}
void Softmax_float_float_cuda_lib_Softmax_265(cudaStream_t stream, float* input0, float* output0)
{

    dispatch_softmax_forward<float, float, float, false>(stream, output0, input0, 197, 197, 113472);
        

}
// Node name:	Dot_320
// Description:	Dot
// Input:
//	- name: Multiply_319_0	type: float	shape: Shape{48, 197, 3072}
//	- name: Constant_131_0	type: float	shape: Shape{3072, 768}
// Output:
//	- name: Dot_320_0	type: float	shape: Shape{48, 197, 768}
void Dot_float_float_float_cuda_lib_Dot_320(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 3072, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 3072, &beta, static_cast<float*>(output0), 768));

}
// Node name:	BatchMatMul_275
// Description:	BatchMatMul
// Input:
//	- name: Broadcast_273_0	type: float	shape: Shape{48, 12, 197, 197}
//	- name: Broadcast_274_0	type: float	shape: Shape{48, 12, 197, 64}
// Output:
//	- name: BatchMatMul_275_0	type: float	shape: Shape{48, 12, 197, 64}
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
//	- name: Reshape_211_0	type: float	shape: Shape{48, 768, 196}
// Output:
//	- name: Reshape_212_0	type: float	shape: Shape{48, 196, 768}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_212(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_212_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_212<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_7_0	type: float	shape: Shape{768}
//	- name: Dot_279_0	type: float	shape: Shape{48, 197, 768}
//	- name: Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Add_282_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_280
// Add, /encoder/layer.0/attention/output/dense/Add_output_0
// Add, /encoder/layer.0/Add_output_0
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 768];
    float temp1 = add(temp0, input1[tid]);
    float temp2 = add(temp1, input2[tid]);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Concat_213
// Description:	Concat
// Input:
//	- name: Constant_200_0	type: float	shape: Shape{48, 1, 768}
//	- name: Reshape_212_0	type: float	shape: Shape{48, 196, 768}
// Output:
//	- name: Concat_213_0	type: float	shape: Shape{48, 197, 768}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_cuda_Concat_213(float* input0, float* input1, float* output0)
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
extern void Concat_float_float_float_cuda_Concat_213_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Concat_float_float_float_cuda_Concat_213<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Broadcast_215
// Description:	Broadcast
// Input:
//	- name: Reshape_214_0	type: float	shape: Shape{197, 768}
// Output:
//	- name: Broadcast_215_0	type: float	shape: Shape{48, 197, 768}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_215(float* input0, float* output0)
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
extern void Broadcast_float_float_cuda_Broadcast_215_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_215<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_218_0	type: float	shape: Shape{}
//	- name: Sum_217_0	type: float	shape: Shape{48, 197}
// Output:
//	- name: Divide_220_0	type: float	shape: Shape{48, 197}
// Fused functions:
// Broadcast, Broadcast_219
// Divide, Divide_220
extern "C" __launch_bounds__(394) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Divide_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 394 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = fdividef(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Divide_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Dot_309
// Description:	Dot
// Input:
//	- name: Add_308_0	type: float	shape: Shape{48, 197, 768}
//	- name: Constant_130_0	type: float	shape: Shape{768, 3072}
// Output:
//	- name: Dot_309_0	type: float	shape: Shape{48, 197, 3072}
void Dot_float_float_float_cuda_lib_Dot_309(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 3072, 9456, 768, &alpha, static_cast<const float*>(input1), 3072, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 3072));

}
// Node name:	Reshape_277
// Description:	Reshape
// Input:
//	- name: Reshape_276_0	type: float	shape: Shape{48, 12, 197, 64}
// Output:
//	- name: Reshape_277_0	type: float	shape: Shape{48, 197, 12, 64}
extern "C" __launch_bounds__(64) __global__ void Reshape_float_float_cuda_Reshape_277(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_277<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_266
// Description:	Dot
// Input:
//	- name: Add_242_0	type: float	shape: Shape{48, 197, 768}
//	- name: Constant_128_0	type: float	shape: Shape{768, 768}
// Output:
//	- name: Dot_266_0	type: float	shape: Shape{48, 197, 768}
void Dot_float_float_float_cuda_lib_Dot_266(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 768, 9456, 768, &alpha, static_cast<const float*>(input1), 768, static_cast<const float*>(input0), 768, &beta, static_cast<float*>(output0), 768));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Reshape_1516_0	type: float	shape: Shape{1}
//	- name: Constant_228_0	type: float	shape: Shape{}
//	- name: Sum_227_0	type: float	shape: Shape{48, 197}
// Output:
//	- name: Sqrt_235_0	type: float	shape: Shape{48, 197, 1}
// Fused functions:
// Broadcast, Broadcast_1529
// Broadcast, Broadcast_229
// Divide, Divide_230
// Reshape, /encoder/layer.0/layernorm_before/ReduceMean_1_output_0
// Add, /encoder/layer.0/layernorm_before/Add_output_0
// Sqrt, /encoder/layer.0/layernorm_before/Sqrt_output_0
extern "C" __launch_bounds__(394) __global__ void FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2(float* input0, float* input1, float* input2, float* output0)
{
    int tid = blockIdx.x * 394 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = input1[tid % 1];
    float temp2 = fdividef(input2[tid], temp1);
    float temp3 = add(temp2, temp0);
    float temp4 = sqrtf(temp3);
    output0[tid] = temp4;

}
extern void FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_11_0	type: float	shape: Shape{768}
//	- name: Constant_10_0	type: float	shape: Shape{768}
//	- name: Reshape_236_0	type: float	shape: Shape{48, 197}
//	- name: Subtract_224_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Add_242_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_241
// Broadcast, Broadcast_239
// Broadcast, Broadcast_237
// Divide, /encoder/layer.0/layernorm_before/Div_output_0
// Multiply, /encoder/layer.0/layernorm_before/Mul_output_0
// Add, /encoder/layer.0/layernorm_before/Add_1_output_0
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3(float* input0, float* input1, float* input2, float* input3, float* output0)
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
extern void FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: Constant_117_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_117(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_117_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_117_0 failed.\n");
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
//	- name: Constant_148_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_148_0 failed.\n");
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
//	- name: Constant_160_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_160(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_160_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_160_0 failed.\n");
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
//	- name: Constant_54_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
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
//	- name: Constant_142_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_142(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_142_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_142_0 failed.\n");
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
//	- name: Constant_92_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_92_0 failed.\n");
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
//	- name: Constant_56_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_56_0 failed.\n");
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
//	- name: Constant_753_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_753(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_753_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_753_0 failed.\n");
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
//	- name: Constant_131_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_131(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_131_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_131_0 failed.\n");
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
//	- name: Constant_2_0	type: float	shape: Shape{768, 3, 16, 16}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
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
//	- name: Constant_183_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_183(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_183_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_183_0 failed.\n");
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
//	- name: Constant_49_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_49_0 failed.\n");
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
//	- name: Constant_48_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_48_0 failed.\n");
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
//	- name: Constant_100_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_100_0 failed.\n");
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
//	- name: Constant_84_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_84_0 failed.\n");
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
//	- name: Constant_829_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_829(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_829_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_829_0 failed.\n");
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
//	- name: Constant_99_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_99_0 failed.\n");
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
//	- name: Constant_157_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_157(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_157_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_157_0 failed.\n");
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
//	- name: Constant_870_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_870(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_870_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_870_0 failed.\n");
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
//	- name: Constant_52_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_52_0 failed.\n");
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
//	- name: Constant_156_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_156_0 failed.\n");
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
//	- name: Constant_284_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_284(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_284_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_284_0 failed.\n");
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
//	- name: Constant_127_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_127(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_127_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_127_0 failed.\n");
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
//	- name: Constant_712_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_712(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_712_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_712_0 failed.\n");
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
//	- name: Constant_102_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_102(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_102_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_102_0 failed.\n");
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
//	- name: Constant_145_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_145(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_145_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_145_0 failed.\n");
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
//	- name: Constant_154_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_154(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_154_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_154_0 failed.\n");
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
//	- name: Constant_155_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_155(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_155_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_155_0 failed.\n");
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
//	- name: Constant_125_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_125_0 failed.\n");
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
//	- name: Constant_140_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_140_0 failed.\n");
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
//	- name: Constant_158_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_158_0 failed.\n");
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
//	- name: Constant_120_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_120_0 failed.\n");
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
//	- name: Constant_150_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_150_0 failed.\n");
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
//	- name: Constant_26_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_26_0 failed.\n");
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
//	- name: Constant_80_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_80(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_80_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_80_0 failed.\n");
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
//	- name: Constant_860_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_860(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_860_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_860_0 failed.\n");
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
//	- name: Constant_46_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
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
//	- name: Constant_187_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_187(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_187_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_187_0 failed.\n");
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
//	- name: Constant_646_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_646(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_646_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_646_0 failed.\n");
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
//	- name: Constant_152_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_152(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_152_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_152_0 failed.\n");
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
//	- name: Constant_176_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_176_0 failed.\n");
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
//	- name: Constant_1074_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1074(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1074_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1074_0 failed.\n");
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
//	- name: Constant_153_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_153(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_153_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_153_0 failed.\n");
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
//	- name: Constant_170_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_170_0 failed.\n");
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
//	- name: Constant_86_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_86_0 failed.\n");
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
//	- name: Constant_111_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_111_0 failed.\n");
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
//	- name: Constant_192_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_192(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_192_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_192_0 failed.\n");
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
//	- name: Constant_42_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_42_0 failed.\n");
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
//	- name: Constant_15_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_15_0 failed.\n");
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
//	- name: Constant_605_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_605(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_605_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_605_0 failed.\n");
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
//	- name: Constant_90_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_90_0 failed.\n");
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
//	- name: Constant_72_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_72_0 failed.\n");
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
//	- name: Constant_206_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_206(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_206_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_206_0 failed.\n");
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
//	- name: Constant_149_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_149_0 failed.\n");
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
//	- name: Constant_164_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_164_0 failed.\n");
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
//	- name: Constant_205_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_205(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_205_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_205_0 failed.\n");
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
//	- name: Constant_61_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_61_0 failed.\n");
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
//	- name: Constant_173_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_173(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_173_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_173_0 failed.\n");
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
//	- name: Constant_144_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_144_0 failed.\n");
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
//	- name: Constant_88_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_88(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_88_0 failed.\n");
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
//	- name: Constant_66_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_66_0 failed.\n");
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
//	- name: Constant_41_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_41(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_41_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_41_0 failed.\n");
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
//	- name: Constant_53_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_53_0 failed.\n");
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
//	- name: Constant_97_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_97_0 failed.\n");
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
//	- name: Constant_40_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_40_0 failed.\n");
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
//	- name: Constant_83_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_83(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_83_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_83_0 failed.\n");
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
//	- name: Constant_549_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_549(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_549_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_549_0 failed.\n");
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
//	- name: Constant_29_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
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
//	- name: Constant_101_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_101_0 failed.\n");
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
//	- name: Constant_181_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_181_0 failed.\n");
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
//	- name: Constant_47_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_47_0 failed.\n");
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
//	- name: Constant_33_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_33_0 failed.\n");
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
//	- name: Constant_138_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_138(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_138_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_138_0 failed.\n");
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
//	- name: Constant_27_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_6_0	type: float	shape: Shape{768}
//	- name: Dot_266_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Add_268_0	type: float	shape: Shape{48, 197, 768}
// Fused functions:
// Broadcast, Broadcast_267
// Add, /encoder/layer.0/attention/attention/value/Add_output_0
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Add_6(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 768];
    float temp1 = add(temp0, input1[tid]);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Add_6<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Constant_977
// Description:	Constant
// Input:
// Output:
//	- name: Constant_977_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_977(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_977_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_977_0 failed.\n");
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
//	- name: Constant_143_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_143_0 failed.\n");
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
//	- name: Constant_39_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
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
//	- name: Constant_6_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
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
//	- name: Constant_3_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
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
//	- name: Constant_62_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_62_0 failed.\n");
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
//	- name: Constant_65_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_65_0 failed.\n");
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
//	- name: Constant_204_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_204(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_204_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_204_0 failed.\n");
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
//	- name: Constant_24_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_24_0 failed.\n");
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
//	- name: Constant_51_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
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
//	- name: Constant_12_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
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
//	- name: Constant_203_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_203(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_203_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_203_0 failed.\n");
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
//	- name: Constant_151_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_151(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_151_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_151_0 failed.\n");
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
//	- name: Constant_137_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_137_0 failed.\n");
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
//	- name: Constant_38_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_38_0 failed.\n");
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
//	- name: Constant_147_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_147(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_147_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_147_0 failed.\n");
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
//	- name: Constant_4_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
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
//	- name: Constant_130_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_130(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_130_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_130_0 failed.\n");
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
//	- name: Constant_126_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_126_0 failed.\n");
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
//	- name: Constant_172_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_172(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_172_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_172_0 failed.\n");
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
//	- name: Constant_146_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_146_0 failed.\n");
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
//	- name: Constant_82_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_82(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_82_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_82_0 failed.\n");
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
//	- name: Constant_135_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_135(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_135_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_135_0 failed.\n");
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
//	- name: Constant_34_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_34(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_34_0 failed.\n");
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
//	- name: Constant_74_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_74(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_74_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_74_0 failed.\n");
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
//	- name: Constant_508_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_508(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_508_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_508_0 failed.\n");
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
//	- name: Constant_1140_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1140_0 failed.\n");
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
//	- name: Constant_819_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_819(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_819_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_819_0 failed.\n");
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
//	- name: Constant_391_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_391(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_391_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_391_0 failed.\n");
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
//	- name: Constant_180_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_180(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_180_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_180_0 failed.\n");
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
//	- name: Constant_188_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_188_0 failed.\n");
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
//	- name: Add_216_0	type: float	shape: Shape{48, 197, 768}
// Output:
//	- name: Sum_217_0	type: float	shape: Shape{48, 197}
extern "C" __launch_bounds__(512) __global__ void Sum_float_float_cuda_Sum_217(float* input0, float* output0)
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
extern void Sum_float_float_cuda_Sum_217_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Sum_float_float_cuda_Sum_217<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_325
// Description:	Constant
// Input:
// Output:
//	- name: Constant_325_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_325(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_325_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_325_0 failed.\n");
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
//	- name: Constant_23_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_23_0 failed.\n");
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
//	- name: Constant_57_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_57_0 failed.\n");
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
//	- name: Constant_73_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_73_0 failed.\n");
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
//	- name: Constant_201_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_201(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_201_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_201_0 failed.\n");
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
//	- name: Constant_178_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_178(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_178_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_178_0 failed.\n");
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
//	- name: Constant_43_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_43_0 failed.\n");
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
//	- name: Constant_22_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_22_0 failed.\n");
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
//	- name: Constant_161_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_161_0 failed.\n");
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
//	- name: Constant_191_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_191_0 failed.\n");
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
//	- name: Constant_936_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_936(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_936_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_936_0 failed.\n");
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
//	- name: Constant_194_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_194_0 failed.\n");
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
//	- name: Constant_5_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_5_0 failed.\n");
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
//	- name: Constant_159_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_159(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_159_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_159_0 failed.\n");
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
//	- name: Constant_129_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_129(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_129_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_129_0 failed.\n");
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
//	- name: Constant_3_0	type: float	shape: Shape{768}
// Output:
//	- name: Broadcast_209_0	type: float	shape: Shape{48, 768, 14, 14}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_209(float* input0, float* output0)
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
extern void Broadcast_float_float_cuda_Broadcast_209_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_209<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_722
// Description:	Constant
// Input:
// Output:
//	- name: Constant_722_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_722(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_722_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_722_0 failed.\n");
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
//	- name: Constant_128_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_128_0 failed.\n");
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
//	- name: Constant_442_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_442(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_442_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_442_0 failed.\n");
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
//	- name: Constant_76_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_76_0 failed.\n");
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
//	- name: Constant_195_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_195(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_195_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_195_0 failed.\n");
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
//	- name: Constant_218_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_218_0 failed.\n");
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
//	- name: Constant_11_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
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
//	- name: Constant_68_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_68_0 failed.\n");
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
//	- name: Constant_1_0	type: float	shape: Shape{1, 197, 768}
void Constant_float_cuda_Constant_1(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1_0 failed.\n");
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
//	- name: Constant_202_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_202(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_202_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_202_0 failed.\n");
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
//	- name: Constant_58_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_58_0 failed.\n");
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
//	- name: Constant_8_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
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
//	- name: Constant_200_0	type: float	shape: Shape{48, 1, 768}
void Constant_float_cuda_Constant_200(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_200_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_200_0 failed.\n");
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
//	- name: Constant_139_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_139(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_139_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_139_0 failed.\n");
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
//	- name: Constant_134_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_134(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_134_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_134_0 failed.\n");
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
//	- name: Constant_335_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_335(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_335_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_335_0 failed.\n");
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
//	- name: Constant_67_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_67_0 failed.\n");
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
//	- name: Constant_228_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_228(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_228_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_228_0 failed.\n");
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
//	- name: Constant_118_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_118(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_118_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_118_0 failed.\n");
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
//	- name: Constant_30_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
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
//	- name: Constant_9_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_9_0 failed.\n");
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
//	- name: Constant_1043_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1043(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1043_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1043_0 failed.\n");
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
//	- name: Constant_96_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_96_0 failed.\n");
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
//	- name: Constant_28_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
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
//	- name: Constant_13_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_13_0 failed.\n");
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
//	- name: Constant_10_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_10_0 failed.\n");
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
//	- name: Constant_89_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_89(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_89_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_89_0 failed.\n");
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
//	- name: Constant_36_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_36_0 failed.\n");
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
//	- name: Constant_186_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_186(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_186_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_186_0 failed.\n");
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
//	- name: Constant_69_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_69_0 failed.\n");
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
//	- name: Constant_169_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_169_0 failed.\n");
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
//	- name: Constant_17_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_17_0 failed.\n");
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
//	- name: Constant_75_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_75(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_75_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_75_0 failed.\n");
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
//	- name: Constant_1033_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1033(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1033_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1033_0 failed.\n");
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
//	- name: Constant_79_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_79_0 failed.\n");
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
//	- name: Constant_174_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_174_0 failed.\n");
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
//	- name: Constant_108_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_108_0 failed.\n");
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
//	- name: Constant_1084_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1084(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1084_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1084_0 failed.\n");
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
//	- name: Constant_196_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_196(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_196_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_196_0 failed.\n");
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
//	- name: Constant_35_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_35_0 failed.\n");
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
//	- name: Constant_105_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_105(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_105_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_105_0 failed.\n");
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
//	- name: Constant_175_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_175(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_175_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_175_0 failed.\n");
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
//	- name: Constant_112_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_112_0 failed.\n");
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
//	- name: Constant_93_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_93_0 failed.\n");
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
//	- name: Constant_50_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_50_0 failed.\n");
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
//	- name: Constant_94_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_94_0 failed.\n");
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
//	- name: Reshape_246_0	type: float	shape: Shape{48, 197, 12, 64}
// Output:
//	- name: Reshape_247_0	type: float	shape: Shape{48, 12, 64, 197}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_247(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_247_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_247<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: Constant_193_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_193(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_193_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_193_0 failed.\n");
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
//	- name: Constant_1181_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1181_0 failed.\n");
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
//	- name: Constant_185_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_185_0 failed.\n");
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
//	- name: Constant_1257_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1257(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1257_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1257_0 failed.\n");
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
//	- name: Constant_124_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_124(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_124_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_124_0 failed.\n");
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
//	- name: Constant_182_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_182_0 failed.\n");
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
//	- name: Constant_1471_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1471(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1471_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1471_0 failed.\n");
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
//	- name: Constant_1461_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1461(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1461_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1461_0 failed.\n");
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
//	- name: Constant_184_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_184(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_184_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_184_0 failed.\n");
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
//	- name: Constant_1502_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1502(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1502_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1502_0 failed.\n");
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
//	- name: Constant_81_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_81(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_81_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_81_0 failed.\n");
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
//	- name: Constant_31_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_31_0 failed.\n");
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
//	- name: Constant_1150_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1150_0 failed.\n");
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
//	- name: Constant_121_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_121(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_121_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_121_0 failed.\n");
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
//	- name: Constant_498_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_498(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_498_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_498_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_204_0	type: float	shape: Shape{}
//	- name: Constant_203_0	type: float	shape: Shape{}
//	- name: Constant_205_0	type: float	shape: Shape{}
//	- name: Constant_8_0	type: float	shape: Shape{3072}
//	- name: Dot_309_0	type: float	shape: Shape{48, 197, 3072}
// Output:
//	- name: Multiply_319_0	type: float	shape: Shape{48, 197, 3072}
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
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
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
extern void FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Constant_163
// Description:	Constant
// Input:
// Output:
//	- name: Constant_163_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_163(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_163_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_163_0 failed.\n");
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
//	- name: Constant_432_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_432(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_432_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_432_0 failed.\n");
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
//	- name: Constant_197_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_197_0 failed.\n");
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
//	- name: Constant_16_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_16_0 failed.\n");
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
//	- name: Constant_113_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_113(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_113_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_113_0 failed.\n");
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
//	- name: Constant_136_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_136(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_136_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_136_0 failed.\n");
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
//	- name: Constant_615_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_615(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_615_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_615_0 failed.\n");
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
//	- name: Constant_1298_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1298(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1298_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1298_0 failed.\n");
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
//	- name: Constant_95_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_95_0 failed.\n");
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
//	- name: Constant_1288_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1288(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1288_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1288_0 failed.\n");
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
//	- name: Constant_189_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_189(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_189_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_189_0 failed.\n");
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
//	- name: Constant_1512_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1512(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1512_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1512_0 failed.\n");
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
//	- name: Constant_1364_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1364(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1364_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1364_0 failed.\n");
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
//	- name: Constant_1405_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1405(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1405_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1405_0 failed.\n");
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
//	- name: Constant_171_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_171_0 failed.\n");
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
//	- name: Constant_123_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_123(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_123_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_123_0 failed.\n");
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
//	- name: Constant_190_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_190_0 failed.\n");
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
//	- name: Constant_116_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_116_0 failed.\n");
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
//	- name: Constant_165_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_165(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_165_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_165_0 failed.\n");
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
//	- name: Constant_107_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_107_0 failed.\n");
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
//	- name: Constant_77_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_77(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_77_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_77_0 failed.\n");
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
//	- name: Constant_85_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_85_0 failed.\n");
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
//	- name: Constant_104_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_104_0 failed.\n");
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
//	- name: Constant_763_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_763(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_763_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_763_0 failed.\n");
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
//	- name: Constant_87_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_87_0 failed.\n");
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
//	- name: Constant_114_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_114_0 failed.\n");
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
//	- name: Constant_106_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_106(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_106_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_106_0 failed.\n");
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
//	- name: Constant_926_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_926(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_926_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_926_0 failed.\n");
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
//	- name: Constant_45_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_45_0 failed.\n");
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
//	- name: Constant_167_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_167_0 failed.\n");
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
//	- name: Constant_179_0	type: float	shape: Shape{3072, 768}
void Constant_float_cuda_Constant_179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_179_0 failed.\n");
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
//	- name: Constant_119_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_119_0 failed.\n");
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
//	- name: Constant_64_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_64(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_64_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_64_0 failed.\n");
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
//	- name: Constant_103_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_103(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_103_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_103_0 failed.\n");
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
//	- name: Constant_71_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_71_0 failed.\n");
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
//	- name: Constant_59_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_59_0 failed.\n");
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
//	- name: Constant_63_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_63(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_63_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_63_0 failed.\n");
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
//	- name: Constant_60_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_60_0 failed.\n");
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
//	- name: Constant_539_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_539(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_539_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_539_0 failed.\n");
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
//	- name: Constant_21_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_21_0 failed.\n");
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
//	- name: Constant_1247_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1247(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1247_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1247_0 failed.\n");
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
//	- name: Constant_166_0	type: float	shape: Shape{768, 3072}
void Constant_float_cuda_Constant_166(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_166_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_166_0 failed.\n");
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
//	- name: Constant_1354_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1354(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1354_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1354_0 failed.\n");
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
//	- name: Constant_1191_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1191_0 failed.\n");
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
//	- name: Constant_133_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_133(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_133_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_133_0 failed.\n");
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
//	- name: Constant_141_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_141(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_141_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_141_0 failed.\n");
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
//	- name: Constant_19_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
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
//	- name: Constant_294_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_294(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_294_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_294_0 failed.\n");
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
//	- name: Constant_70_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_70_0 failed.\n");
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
//	- name: Constant_18_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
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
//	- name: Constant_656_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_656(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_656_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_656_0 failed.\n");
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
//	- name: Constant_401_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_401(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_401_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_401_0 failed.\n");
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
//	- name: Constant_110_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_110_0 failed.\n");
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
//	- name: Constant_177_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_177_0 failed.\n");
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
//	- name: Constant_20_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_20_0 failed.\n");
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
//	- name: Constant_132_0	type: float	shape: Shape{768, 768}
void Constant_float_cuda_Constant_132(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_132_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_132_0 failed.\n");
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
//	- name: Constant_37_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_37_0 failed.\n");
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
//	- name: Constant_44_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_44_0 failed.\n");
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
//	- name: Constant_122_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_122(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_122_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_122_0 failed.\n");
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
//	- name: Constant_78_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_78_0 failed.\n");
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
//	- name: Constant_967_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_967(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_967_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_967_0 failed.\n");
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
//	- name: Constant_25_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_25_0 failed.\n");
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
//	- name: Constant_14_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
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
//	- name: Constant_32_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
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
//	- name: Constant_7_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
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
//	- name: Constant_55_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_55_0 failed.\n");
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
//	- name: Constant_98_0	type: float	shape: Shape{3072}
void Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_98_0 failed.\n");
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
//	- name: Constant_115_0	type: float	shape: Shape{768}
void Constant_float_cuda_Constant_115(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_115_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_115_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3072];
    bin_file.read(tmp_mem, 3072);
    cudaMemcpyAsync(output0, tmp_mem, 3072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}

extern "C" void cuda_init()
{
// CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:725516288
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,353124864));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 353124864));
Broadcast_215_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_209_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Convolution_208_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+57950208);
Add_210_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+57950208);
Reshape_211_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+57950208);
Reshape_212_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Concat_213_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+57950208);
Add_216_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+57950208);
Sum_217_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_220_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_221_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_222_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_226_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_224_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Sum_227_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_235_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_236_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_242_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_266_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Add_268_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Reshape_269_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Reshape_270_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Reshape_272_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Broadcast_274_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Dot_243_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Add_245_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145096704);
Reshape_246_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145096704);
Reshape_247_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Multiply_250_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145096704);
Reshape_260_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145096704);
Broadcast_262_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145096704);
Dot_251_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Add_253_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_254_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_255_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Multiply_258_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_259_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_261_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_263_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174145536);
Reshape_264_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174145536);
Softmax_265_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263561472);
Reshape_271_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263561472);
Broadcast_273_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263561472);
BatchMatMul_275_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_276_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_277_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Reshape_278_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Dot_279_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_282_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+86999040);
Sum_283_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_286_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_287_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_288_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_292_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_290_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29124480);
Sum_293_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_301_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_302_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_308_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_309_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116047872);
Multiply_319_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+232243200);
Dot_320_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_323_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_324_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_327_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_328_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_329_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_333_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_331_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_334_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_342_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_343_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_349_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_373_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_375_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_376_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_377_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_379_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_381_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_350_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_352_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_353_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_354_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_357_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_367_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Broadcast_369_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Dot_358_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_360_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Reshape_361_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Reshape_362_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_365_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Reshape_366_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Broadcast_368_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
BatchMatMul_370_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_371_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Softmax_372_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+234660096);
Reshape_378_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+234660096);
Broadcast_380_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+234660096);
BatchMatMul_382_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Reshape_383_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Reshape_384_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_385_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_386_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Add_389_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sum_390_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Divide_393_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29086656);
Reshape_394_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29086656);
Reshape_395_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29086656);
Power_399_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29124480);
Subtract_397_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58173312);
Sum_400_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sqrt_408_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29086656);
Reshape_409_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29086656);
Add_415_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29124480);
Dot_416_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58173312);
Multiply_426_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174368640);
Dot_427_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Add_430_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_431_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_434_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_435_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_436_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_440_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_438_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_441_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_449_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_450_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_456_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_480_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_482_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_483_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_484_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_486_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_488_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_457_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_459_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_460_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_461_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_464_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_474_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_476_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_465_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_467_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_468_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_469_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_472_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_473_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_475_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_477_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_478_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_479_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_485_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_487_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_489_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_490_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_491_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_492_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_493_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_496_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_497_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_500_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_501_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_502_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_506_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_504_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_507_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_515_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_516_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_522_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_523_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_533_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_534_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_537_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_538_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_541_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_542_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_543_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_547_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_545_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_548_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_556_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_557_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_563_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_587_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_589_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_590_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_591_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_593_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_595_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_564_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_566_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_567_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_568_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_571_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_581_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_583_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_572_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_574_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_575_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_576_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_579_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_580_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_582_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_584_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_585_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_586_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_592_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_594_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_596_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_597_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_598_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_599_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_600_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_603_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_604_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_607_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_608_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_609_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_613_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_611_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_614_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_622_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_623_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_629_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_630_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_640_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_641_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_644_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_645_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_648_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_649_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_650_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_654_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_652_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_655_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_663_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_664_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_670_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_694_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_696_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_697_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_698_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_700_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_702_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_671_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_673_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_674_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_675_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_678_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_688_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_690_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_679_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_681_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_682_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_683_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_686_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_687_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_689_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_691_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_692_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_693_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_699_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_701_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_703_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_704_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_705_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_706_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_707_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_710_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_711_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_714_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_715_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_716_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_720_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_718_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_721_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_729_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_730_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_736_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_737_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_747_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_748_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_751_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_752_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_755_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_756_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_757_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_761_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_759_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_762_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_770_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_771_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_777_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_801_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_803_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_804_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_805_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_807_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_809_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_778_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_780_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_781_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_782_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_785_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_795_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_797_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_786_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_788_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_789_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_790_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_793_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_794_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_796_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_798_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_799_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_800_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_806_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_808_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_810_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_811_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_812_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_813_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_814_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_817_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_818_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_821_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_822_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_823_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_827_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_825_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_828_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_836_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_837_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_843_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_844_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_854_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_855_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_858_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_859_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_862_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_863_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_864_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_868_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_866_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_869_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_877_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_878_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_884_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_908_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_910_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_911_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_912_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_914_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_916_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_885_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_887_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_888_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_889_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_892_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_902_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_904_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_893_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_895_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_896_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_897_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_900_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_901_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_903_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_905_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_906_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_907_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_913_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_915_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_917_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_918_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_919_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_920_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_921_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_924_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_925_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_928_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_929_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_930_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_934_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_932_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_935_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_943_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_944_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_950_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_951_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_961_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_962_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_965_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_966_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_969_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_970_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_971_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_975_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_973_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_976_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_984_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_985_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_991_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_1015_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_1017_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1018_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1019_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_1021_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_1023_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_992_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_994_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_995_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_996_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_999_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1009_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_1011_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_1000_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1002_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1003_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1004_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_1007_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1008_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_1010_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_1012_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_1013_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_1014_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_1020_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_1022_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_1024_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1025_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1026_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_1027_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_1028_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1031_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_1032_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1035_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1036_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1037_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1041_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_1039_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1042_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1050_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1051_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1057_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_1058_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_1068_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_1069_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1072_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_1073_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1076_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1077_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1078_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1082_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_1080_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1083_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1091_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1092_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1098_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_1122_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_1124_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1125_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1126_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_1128_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_1130_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_1099_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1101_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1102_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1103_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_1106_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1116_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_1118_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_1107_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1109_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1110_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1111_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_1114_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1115_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_1117_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_1119_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_1120_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_1121_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_1127_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_1129_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_1131_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1132_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1133_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_1134_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_1135_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1138_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_1139_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1142_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1143_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1144_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1148_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_1146_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1149_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1157_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1158_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1164_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_1165_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_1175_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_1176_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1179_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_1180_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1183_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1184_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1185_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1189_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_1187_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1190_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1198_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1199_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1205_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_1229_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_1231_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1232_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1233_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_1235_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_1237_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_1206_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1208_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1209_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1210_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_1213_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1223_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_1225_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_1214_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1216_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1217_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1218_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_1221_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1222_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_1224_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_1226_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_1227_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_1228_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_1234_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_1236_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_1238_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1239_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1240_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_1241_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_1242_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1245_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_1246_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1249_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1250_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1251_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1255_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_1253_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1256_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1264_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1265_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1271_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_1272_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_1282_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_1283_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1286_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_1287_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1290_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1291_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1292_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1296_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_1294_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1297_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1305_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1306_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1312_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_1336_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_1338_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1339_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1340_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_1342_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_1344_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_1313_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1315_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1316_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1317_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_1320_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1330_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_1332_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_1321_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1323_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1324_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1325_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_1328_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1329_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_1331_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_1333_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_1334_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_1335_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_1341_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_1343_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_1345_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1346_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1347_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_1348_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_1349_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1352_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_1353_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1356_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1357_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1358_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1362_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_1360_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1363_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1371_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1372_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1378_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_1379_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_1389_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_1390_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1393_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_1394_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1397_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1398_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1399_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1403_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_1401_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1404_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1412_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1413_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1419_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Dot_1443_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Add_1445_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1446_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Reshape_1447_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Reshape_1449_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Broadcast_1451_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Dot_1420_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1422_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1423_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1424_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Multiply_1427_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Reshape_1437_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Broadcast_1439_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+145244160);
Dot_1428_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+116195328);
Add_1430_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1431_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1432_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Multiply_1435_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1436_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_1438_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchMatMul_1440_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Reshape_1441_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+174292992);
Softmax_1442_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Reshape_1448_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
Broadcast_1450_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+263708928);
BatchMatMul_1452_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1453_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_1454_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Reshape_1455_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Dot_1456_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1459_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+29048832);
Sum_1460_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1463_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1464_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1465_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1469_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Subtract_1467_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1470_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1478_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1479_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Add_1485_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Dot_1486_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Multiply_1496_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+203341824);
Dot_1497_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_1500_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+58097664);
Sum_1501_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Divide_1504_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1505_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1506_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Power_1510_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75648);
Subtract_1508_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+87146496);
Sum_1511_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sqrt_1519_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
Reshape_1520_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37824);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,372391424));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 372391424));
Constant_1_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Reshape_214_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+605184);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+608256);
Constant_200_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2967552);
Constant_129_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3115008);
Constant_128_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5474304);
Constant_218_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833600);
Constant_206_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833664);
Constant_228_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833728);
Constant_202_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833792);
Reshape_1516_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833792);
Constant_10_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7833856);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7836928);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7840000);
Constant_127_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7843072);
Constant_5_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10202368);
Constant_201_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10205440);
Reshape_248_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10205440);
Constant_126_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10205504);
Constant_4_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12564800);
Constant_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12567872);
Constant_131_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12570944);
Constant_130_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22008128);
Constant_284_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31445312);
Constant_294_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31445376);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31445440);
Constant_13_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31448512);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31451584);
Constant_205_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31463872);
Constant_203_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31463936);
Constant_204_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31464000);
Constant_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31464064);
Constant_135_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31467136);
Constant_134_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33826432);
Constant_325_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36185728);
Constant_335_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36185792);
Constant_20_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36185856);
Constant_21_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36188928);
Constant_16_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36192000);
Constant_133_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36195072);
Constant_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38554368);
Constant_132_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38557440);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40916736);
Constant_17_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40919808);
Constant_137_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40922880);
Constant_136_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50360064);
Constant_391_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59797248);
Constant_401_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59797312);
Constant_22_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59797376);
Constant_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59800448);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59803520);
Constant_19_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59815808);
Constant_141_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+59818880);
Constant_140_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62178176);
Constant_432_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64537472);
Constant_442_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64537536);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64537600);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64540672);
Constant_26_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64543744);
Constant_139_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64546816);
Constant_25_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+66906112);
Constant_138_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+66909184);
Constant_24_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69268480);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69271552);
Constant_143_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69274624);
Constant_142_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+78711808);
Constant_498_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88148992);
Constant_508_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88149056);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88149120);
Constant_33_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88152192);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88155264);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88167552);
Constant_147_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+88170624);
Constant_146_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90529920);
Constant_539_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92889216);
Constant_549_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92889280);
Constant_40_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92889344);
Constant_41_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92892416);
Constant_36_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92895488);
Constant_145_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92898560);
Constant_35_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95257856);
Constant_144_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95260928);
Constant_34_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+97620224);
Constant_37_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+97623296);
Constant_149_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+97626368);
Constant_148_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+107063552);
Constant_605_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116500736);
Constant_615_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116500800);
Constant_42_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116500864);
Constant_43_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116503936);
Constant_38_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116507008);
Constant_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116519296);
Constant_153_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+116522368);
Constant_152_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+118881664);
Constant_646_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121240960);
Constant_656_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121241024);
Constant_50_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121241088);
Constant_51_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121244160);
Constant_46_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121247232);
Constant_151_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+121250304);
Constant_45_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+123609600);
Constant_150_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+123612672);
Constant_44_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+125971968);
Constant_47_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+125975040);
Constant_155_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+125978112);
Constant_154_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+135415296);
Constant_712_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144852480);
Constant_722_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144852544);
Constant_52_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144852608);
Constant_53_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144855680);
Constant_48_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144858752);
Constant_49_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144871040);
Constant_159_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+144874112);
Constant_158_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+147233408);
Constant_753_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149592704);
Constant_763_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149592768);
Constant_60_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149592832);
Constant_61_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149595904);
Constant_56_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149598976);
Constant_157_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+149602048);
Constant_55_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+151961344);
Constant_156_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+151964416);
Constant_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+154323712);
Constant_57_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+154326784);
Constant_161_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+154329856);
Constant_160_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+163767040);
Constant_819_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173204224);
Constant_829_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173204288);
Constant_62_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173204352);
Constant_63_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173207424);
Constant_58_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173210496);
Constant_59_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173222784);
Constant_165_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+173225856);
Constant_164_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+175585152);
Constant_860_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177944448);
Constant_870_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177944512);
Constant_70_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177944576);
Constant_71_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177947648);
Constant_66_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177950720);
Constant_163_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+177953792);
Constant_65_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+180313088);
Constant_162_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+180316160);
Constant_64_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+182675456);
Constant_67_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+182678528);
Constant_167_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+182681600);
Constant_166_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+192118784);
Constant_926_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201555968);
Constant_936_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201556032);
Constant_72_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201556096);
Constant_73_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201559168);
Constant_68_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201562240);
Constant_69_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201574528);
Constant_171_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+201577600);
Constant_170_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+203936896);
Constant_967_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206296192);
Constant_977_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206296256);
Constant_80_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206296320);
Constant_81_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206299392);
Constant_76_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206302464);
Constant_169_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+206305536);
Constant_75_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+208664832);
Constant_168_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+208667904);
Constant_74_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+211027200);
Constant_77_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+211030272);
Constant_173_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+211033344);
Constant_172_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+220470528);
Constant_1033_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229907712);
Constant_1043_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229907776);
Constant_82_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229907840);
Constant_83_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229910912);
Constant_78_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229913984);
Constant_79_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229926272);
Constant_177_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+229929344);
Constant_176_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+232288640);
Constant_1074_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234647936);
Constant_1084_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234648000);
Constant_90_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234648064);
Constant_91_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234651136);
Constant_86_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234654208);
Constant_175_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+234657280);
Constant_85_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+237016576);
Constant_174_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+237019648);
Constant_84_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+239378944);
Constant_87_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+239382016);
Constant_179_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+239385088);
Constant_178_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+248822272);
Constant_1140_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258259456);
Constant_1150_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258259520);
Constant_92_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258259584);
Constant_93_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258262656);
Constant_88_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258265728);
Constant_89_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258278016);
Constant_183_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+258281088);
Constant_182_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+260640384);
Constant_1181_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+262999680);
Constant_1191_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+262999744);
Constant_100_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+262999808);
Constant_101_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+263002880);
Constant_96_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+263005952);
Constant_181_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+263009024);
Constant_95_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+265368320);
Constant_180_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+265371392);
Constant_94_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+267730688);
Constant_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+267733760);
Constant_185_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+267736832);
Constant_184_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+277174016);
Constant_1247_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286611200);
Constant_1257_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286611264);
Constant_102_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286611328);
Constant_103_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286614400);
Constant_98_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286617472);
Constant_99_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286629760);
Constant_189_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+286632832);
Constant_188_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+288992128);
Constant_1288_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291351424);
Constant_1298_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291351488);
Constant_110_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291351552);
Constant_111_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291354624);
Constant_106_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291357696);
Constant_187_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+291360768);
Constant_105_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+293720064);
Constant_186_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+293723136);
Constant_104_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+296082432);
Constant_107_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+296085504);
Constant_191_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+296088576);
Constant_190_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+305525760);
Constant_1354_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314962944);
Constant_1364_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314963008);
Constant_112_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314963072);
Constant_113_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314966144);
Constant_108_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314969216);
Constant_109_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314981504);
Constant_195_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+314984576);
Constant_194_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+317343872);
Constant_1395_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319703168);
Constant_1405_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319703232);
Constant_120_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319703296);
Constant_121_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319706368);
Constant_116_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319709440);
Constant_193_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+319712512);
Constant_115_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+322071808);
Constant_192_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+322074880);
Constant_114_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+324434176);
Constant_117_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+324437248);
Constant_197_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+324440320);
Constant_196_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+333877504);
Constant_1461_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343314688);
Constant_1471_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343314752);
Constant_122_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343314816);
Constant_123_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343317888);
Constant_118_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343320960);
Constant_119_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343333248);
Constant_1502_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343336320);
Constant_1512_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343336384);
Constant_124_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343336448);
Constant_125_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343339520);
last_hidden_state = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343342592);
Result_1527_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343342592);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
 // name=embeddings.position_embeddings
Constant_float_cuda_Constant_1(0, Constant_1_0);
 // name=embeddings.patch_embeddings.projection.bias
Constant_float_cuda_Constant_3(0, Constant_3_0);
 // name=embeddings.patch_embeddings.projection.weight
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=/embeddings/Expand_output_0
Constant_float_cuda_Constant_200(0, Constant_200_0);
 // name=onnx::MatMul_1843
Constant_float_cuda_Constant_129(0, Constant_129_0);
 // name=onnx::MatMul_1837
Constant_float_cuda_Constant_128(0, Constant_128_0);
 // name=Constant_218
Constant_float_cuda_Constant_218(0, Constant_218_0);
 // name=ortshared_1_0_1_3_token_103
Constant_float_cuda_Constant_206(0, Constant_206_0);
 // name=Constant_228
Constant_float_cuda_Constant_228(0, Constant_228_0);
 // name=ortshared_1_0_1_4_token_104
Constant_float_cuda_Constant_202(0, Constant_202_0);
 // name=encoder.layer.0.layernorm_before.weight
Constant_float_cuda_Constant_10(0, Constant_10_0);
 // name=encoder.layer.0.layernorm_before.bias
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=encoder.layer.0.attention.attention.value.bias
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=onnx::MatMul_1834
Constant_float_cuda_Constant_127(0, Constant_127_0);
 // name=encoder.layer.0.attention.attention.key.bias
Constant_float_cuda_Constant_5(0, Constant_5_0);
 // name=ortshared_1_1_1_0_token_101
Constant_float_cuda_Constant_201(0, Constant_201_0);
 // name=onnx::MatMul_1833
Constant_float_cuda_Constant_126(0, Constant_126_0);
 // name=encoder.layer.0.attention.attention.query.bias
Constant_float_cuda_Constant_4(0, Constant_4_0);
 // name=encoder.layer.0.attention.output.dense.bias
Constant_float_cuda_Constant_7(0, Constant_7_0);
 // name=onnx::MatMul_1845
Constant_float_cuda_Constant_131(0, Constant_131_0);
 // name=onnx::MatMul_1844
Constant_float_cuda_Constant_130(0, Constant_130_0);
 // name=Constant_284
Constant_float_cuda_Constant_284(0, Constant_284_0);
 // name=Constant_294
Constant_float_cuda_Constant_294(0, Constant_294_0);
 // name=encoder.layer.0.layernorm_after.weight
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=encoder.layer.0.layernorm_after.bias
Constant_float_cuda_Constant_13(0, Constant_13_0);
 // name=encoder.layer.0.intermediate.dense.bias
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=ortshared_1_0_1_2_token_100
Constant_float_cuda_Constant_205(0, Constant_205_0);
 // name=ortshared_1_0_1_0_token_97
Constant_float_cuda_Constant_203(0, Constant_203_0);
 // name=ortshared_1_0_1_1_token_99
Constant_float_cuda_Constant_204(0, Constant_204_0);
 // name=encoder.layer.0.output.dense.bias
Constant_float_cuda_Constant_9(0, Constant_9_0);
 // name=onnx::MatMul_1856
Constant_float_cuda_Constant_135(0, Constant_135_0);
 // name=onnx::MatMul_1850
Constant_float_cuda_Constant_134(0, Constant_134_0);
 // name=Constant_325
Constant_float_cuda_Constant_325(0, Constant_325_0);
 // name=Constant_335
Constant_float_cuda_Constant_335(0, Constant_335_0);
 // name=encoder.layer.1.layernorm_before.weight
Constant_float_cuda_Constant_20(0, Constant_20_0);
 // name=encoder.layer.1.layernorm_before.bias
Constant_float_cuda_Constant_21(0, Constant_21_0);
 // name=encoder.layer.1.attention.attention.value.bias
Constant_float_cuda_Constant_16(0, Constant_16_0);
 // name=onnx::MatMul_1847
Constant_float_cuda_Constant_133(0, Constant_133_0);
 // name=encoder.layer.1.attention.attention.key.bias
Constant_float_cuda_Constant_15(0, Constant_15_0);
 // name=onnx::MatMul_1846
Constant_float_cuda_Constant_132(0, Constant_132_0);
 // name=encoder.layer.1.attention.attention.query.bias
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=encoder.layer.1.attention.output.dense.bias
Constant_float_cuda_Constant_17(0, Constant_17_0);
 // name=onnx::MatMul_1858
Constant_float_cuda_Constant_137(0, Constant_137_0);
 // name=onnx::MatMul_1857
Constant_float_cuda_Constant_136(0, Constant_136_0);
 // name=Constant_391
Constant_float_cuda_Constant_391(0, Constant_391_0);
 // name=Constant_401
Constant_float_cuda_Constant_401(0, Constant_401_0);
 // name=encoder.layer.1.layernorm_after.weight
Constant_float_cuda_Constant_22(0, Constant_22_0);
 // name=encoder.layer.1.layernorm_after.bias
Constant_float_cuda_Constant_23(0, Constant_23_0);
 // name=encoder.layer.1.intermediate.dense.bias
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=encoder.layer.1.output.dense.bias
Constant_float_cuda_Constant_19(0, Constant_19_0);
 // name=onnx::MatMul_1869
Constant_float_cuda_Constant_141(0, Constant_141_0);
 // name=onnx::MatMul_1863
Constant_float_cuda_Constant_140(0, Constant_140_0);
 // name=Constant_432
Constant_float_cuda_Constant_432(0, Constant_432_0);
 // name=Constant_442
Constant_float_cuda_Constant_442(0, Constant_442_0);
 // name=encoder.layer.2.layernorm_before.weight
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=encoder.layer.2.layernorm_before.bias
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=encoder.layer.2.attention.attention.value.bias
Constant_float_cuda_Constant_26(0, Constant_26_0);
 // name=onnx::MatMul_1860
Constant_float_cuda_Constant_139(0, Constant_139_0);
 // name=encoder.layer.2.attention.attention.key.bias
Constant_float_cuda_Constant_25(0, Constant_25_0);
 // name=onnx::MatMul_1859
Constant_float_cuda_Constant_138(0, Constant_138_0);
 // name=encoder.layer.2.attention.attention.query.bias
Constant_float_cuda_Constant_24(0, Constant_24_0);
 // name=encoder.layer.2.attention.output.dense.bias
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=onnx::MatMul_1871
Constant_float_cuda_Constant_143(0, Constant_143_0);
 // name=onnx::MatMul_1870
Constant_float_cuda_Constant_142(0, Constant_142_0);
 // name=Constant_498
Constant_float_cuda_Constant_498(0, Constant_498_0);
 // name=Constant_508
Constant_float_cuda_Constant_508(0, Constant_508_0);
 // name=encoder.layer.2.layernorm_after.weight
Constant_float_cuda_Constant_32(0, Constant_32_0);
 // name=encoder.layer.2.layernorm_after.bias
Constant_float_cuda_Constant_33(0, Constant_33_0);
 // name=encoder.layer.2.intermediate.dense.bias
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=encoder.layer.2.output.dense.bias
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=onnx::MatMul_1882
Constant_float_cuda_Constant_147(0, Constant_147_0);
 // name=onnx::MatMul_1876
Constant_float_cuda_Constant_146(0, Constant_146_0);
 // name=Constant_539
Constant_float_cuda_Constant_539(0, Constant_539_0);
 // name=Constant_549
Constant_float_cuda_Constant_549(0, Constant_549_0);
 // name=encoder.layer.3.layernorm_before.weight
Constant_float_cuda_Constant_40(0, Constant_40_0);
 // name=encoder.layer.3.layernorm_before.bias
Constant_float_cuda_Constant_41(0, Constant_41_0);
 // name=encoder.layer.3.attention.attention.value.bias
Constant_float_cuda_Constant_36(0, Constant_36_0);
 // name=onnx::MatMul_1873
Constant_float_cuda_Constant_145(0, Constant_145_0);
 // name=encoder.layer.3.attention.attention.key.bias
Constant_float_cuda_Constant_35(0, Constant_35_0);
 // name=onnx::MatMul_1872
Constant_float_cuda_Constant_144(0, Constant_144_0);
 // name=encoder.layer.3.attention.attention.query.bias
Constant_float_cuda_Constant_34(0, Constant_34_0);
 // name=encoder.layer.3.attention.output.dense.bias
Constant_float_cuda_Constant_37(0, Constant_37_0);
 // name=onnx::MatMul_1884
Constant_float_cuda_Constant_149(0, Constant_149_0);
 // name=onnx::MatMul_1883
Constant_float_cuda_Constant_148(0, Constant_148_0);
 // name=Constant_605
Constant_float_cuda_Constant_605(0, Constant_605_0);
 // name=Constant_615
Constant_float_cuda_Constant_615(0, Constant_615_0);
 // name=encoder.layer.3.layernorm_after.weight
Constant_float_cuda_Constant_42(0, Constant_42_0);
 // name=encoder.layer.3.layernorm_after.bias
Constant_float_cuda_Constant_43(0, Constant_43_0);
 // name=encoder.layer.3.intermediate.dense.bias
Constant_float_cuda_Constant_38(0, Constant_38_0);
 // name=encoder.layer.3.output.dense.bias
Constant_float_cuda_Constant_39(0, Constant_39_0);
 // name=onnx::MatMul_1895
Constant_float_cuda_Constant_153(0, Constant_153_0);
 // name=onnx::MatMul_1889
Constant_float_cuda_Constant_152(0, Constant_152_0);
 // name=Constant_646
Constant_float_cuda_Constant_646(0, Constant_646_0);
 // name=Constant_656
Constant_float_cuda_Constant_656(0, Constant_656_0);
 // name=encoder.layer.4.layernorm_before.weight
Constant_float_cuda_Constant_50(0, Constant_50_0);
 // name=encoder.layer.4.layernorm_before.bias
Constant_float_cuda_Constant_51(0, Constant_51_0);
 // name=encoder.layer.4.attention.attention.value.bias
Constant_float_cuda_Constant_46(0, Constant_46_0);
 // name=onnx::MatMul_1886
Constant_float_cuda_Constant_151(0, Constant_151_0);
 // name=encoder.layer.4.attention.attention.key.bias
Constant_float_cuda_Constant_45(0, Constant_45_0);
 // name=onnx::MatMul_1885
Constant_float_cuda_Constant_150(0, Constant_150_0);
 // name=encoder.layer.4.attention.attention.query.bias
Constant_float_cuda_Constant_44(0, Constant_44_0);
 // name=encoder.layer.4.attention.output.dense.bias
Constant_float_cuda_Constant_47(0, Constant_47_0);
 // name=onnx::MatMul_1897
Constant_float_cuda_Constant_155(0, Constant_155_0);
 // name=onnx::MatMul_1896
Constant_float_cuda_Constant_154(0, Constant_154_0);
 // name=Constant_712
Constant_float_cuda_Constant_712(0, Constant_712_0);
 // name=Constant_722
Constant_float_cuda_Constant_722(0, Constant_722_0);
 // name=encoder.layer.4.layernorm_after.weight
Constant_float_cuda_Constant_52(0, Constant_52_0);
 // name=encoder.layer.4.layernorm_after.bias
Constant_float_cuda_Constant_53(0, Constant_53_0);
 // name=encoder.layer.4.intermediate.dense.bias
Constant_float_cuda_Constant_48(0, Constant_48_0);
 // name=encoder.layer.4.output.dense.bias
Constant_float_cuda_Constant_49(0, Constant_49_0);
 // name=onnx::MatMul_1908
Constant_float_cuda_Constant_159(0, Constant_159_0);
 // name=onnx::MatMul_1902
Constant_float_cuda_Constant_158(0, Constant_158_0);
 // name=Constant_753
Constant_float_cuda_Constant_753(0, Constant_753_0);
 // name=Constant_763
Constant_float_cuda_Constant_763(0, Constant_763_0);
 // name=encoder.layer.5.layernorm_before.weight
Constant_float_cuda_Constant_60(0, Constant_60_0);
 // name=encoder.layer.5.layernorm_before.bias
Constant_float_cuda_Constant_61(0, Constant_61_0);
 // name=encoder.layer.5.attention.attention.value.bias
Constant_float_cuda_Constant_56(0, Constant_56_0);
 // name=onnx::MatMul_1899
Constant_float_cuda_Constant_157(0, Constant_157_0);
 // name=encoder.layer.5.attention.attention.key.bias
Constant_float_cuda_Constant_55(0, Constant_55_0);
 // name=onnx::MatMul_1898
Constant_float_cuda_Constant_156(0, Constant_156_0);
 // name=encoder.layer.5.attention.attention.query.bias
Constant_float_cuda_Constant_54(0, Constant_54_0);
 // name=encoder.layer.5.attention.output.dense.bias
Constant_float_cuda_Constant_57(0, Constant_57_0);
 // name=onnx::MatMul_1910
Constant_float_cuda_Constant_161(0, Constant_161_0);
 // name=onnx::MatMul_1909
Constant_float_cuda_Constant_160(0, Constant_160_0);
 // name=Constant_819
Constant_float_cuda_Constant_819(0, Constant_819_0);
 // name=Constant_829
Constant_float_cuda_Constant_829(0, Constant_829_0);
 // name=encoder.layer.5.layernorm_after.weight
Constant_float_cuda_Constant_62(0, Constant_62_0);
 // name=encoder.layer.5.layernorm_after.bias
Constant_float_cuda_Constant_63(0, Constant_63_0);
 // name=encoder.layer.5.intermediate.dense.bias
Constant_float_cuda_Constant_58(0, Constant_58_0);
 // name=encoder.layer.5.output.dense.bias
Constant_float_cuda_Constant_59(0, Constant_59_0);
 // name=onnx::MatMul_1921
Constant_float_cuda_Constant_165(0, Constant_165_0);
 // name=onnx::MatMul_1915
Constant_float_cuda_Constant_164(0, Constant_164_0);
 // name=Constant_860
Constant_float_cuda_Constant_860(0, Constant_860_0);
 // name=Constant_870
Constant_float_cuda_Constant_870(0, Constant_870_0);
 // name=encoder.layer.6.layernorm_before.weight
Constant_float_cuda_Constant_70(0, Constant_70_0);
 // name=encoder.layer.6.layernorm_before.bias
Constant_float_cuda_Constant_71(0, Constant_71_0);
 // name=encoder.layer.6.attention.attention.value.bias
Constant_float_cuda_Constant_66(0, Constant_66_0);
 // name=onnx::MatMul_1912
Constant_float_cuda_Constant_163(0, Constant_163_0);
 // name=encoder.layer.6.attention.attention.key.bias
Constant_float_cuda_Constant_65(0, Constant_65_0);
 // name=onnx::MatMul_1911
Constant_float_cuda_Constant_162(0, Constant_162_0);
 // name=encoder.layer.6.attention.attention.query.bias
Constant_float_cuda_Constant_64(0, Constant_64_0);
 // name=encoder.layer.6.attention.output.dense.bias
Constant_float_cuda_Constant_67(0, Constant_67_0);
 // name=onnx::MatMul_1923
Constant_float_cuda_Constant_167(0, Constant_167_0);
 // name=onnx::MatMul_1922
Constant_float_cuda_Constant_166(0, Constant_166_0);
 // name=Constant_926
Constant_float_cuda_Constant_926(0, Constant_926_0);
 // name=Constant_936
Constant_float_cuda_Constant_936(0, Constant_936_0);
 // name=encoder.layer.6.layernorm_after.weight
Constant_float_cuda_Constant_72(0, Constant_72_0);
 // name=encoder.layer.6.layernorm_after.bias
Constant_float_cuda_Constant_73(0, Constant_73_0);
 // name=encoder.layer.6.intermediate.dense.bias
Constant_float_cuda_Constant_68(0, Constant_68_0);
 // name=encoder.layer.6.output.dense.bias
Constant_float_cuda_Constant_69(0, Constant_69_0);
 // name=onnx::MatMul_1934
Constant_float_cuda_Constant_171(0, Constant_171_0);
 // name=onnx::MatMul_1928
Constant_float_cuda_Constant_170(0, Constant_170_0);
 // name=Constant_967
Constant_float_cuda_Constant_967(0, Constant_967_0);
 // name=Constant_977
Constant_float_cuda_Constant_977(0, Constant_977_0);
 // name=encoder.layer.7.layernorm_before.weight
Constant_float_cuda_Constant_80(0, Constant_80_0);
 // name=encoder.layer.7.layernorm_before.bias
Constant_float_cuda_Constant_81(0, Constant_81_0);
 // name=encoder.layer.7.attention.attention.value.bias
Constant_float_cuda_Constant_76(0, Constant_76_0);
 // name=onnx::MatMul_1925
Constant_float_cuda_Constant_169(0, Constant_169_0);
 // name=encoder.layer.7.attention.attention.key.bias
Constant_float_cuda_Constant_75(0, Constant_75_0);
 // name=onnx::MatMul_1924
Constant_float_cuda_Constant_168(0, Constant_168_0);
 // name=encoder.layer.7.attention.attention.query.bias
Constant_float_cuda_Constant_74(0, Constant_74_0);
 // name=encoder.layer.7.attention.output.dense.bias
Constant_float_cuda_Constant_77(0, Constant_77_0);
 // name=onnx::MatMul_1936
Constant_float_cuda_Constant_173(0, Constant_173_0);
 // name=onnx::MatMul_1935
Constant_float_cuda_Constant_172(0, Constant_172_0);
 // name=Constant_1033
Constant_float_cuda_Constant_1033(0, Constant_1033_0);
 // name=Constant_1043
Constant_float_cuda_Constant_1043(0, Constant_1043_0);
 // name=encoder.layer.7.layernorm_after.weight
Constant_float_cuda_Constant_82(0, Constant_82_0);
 // name=encoder.layer.7.layernorm_after.bias
Constant_float_cuda_Constant_83(0, Constant_83_0);
 // name=encoder.layer.7.intermediate.dense.bias
Constant_float_cuda_Constant_78(0, Constant_78_0);
 // name=encoder.layer.7.output.dense.bias
Constant_float_cuda_Constant_79(0, Constant_79_0);
 // name=onnx::MatMul_1947
Constant_float_cuda_Constant_177(0, Constant_177_0);
 // name=onnx::MatMul_1941
Constant_float_cuda_Constant_176(0, Constant_176_0);
 // name=Constant_1074
Constant_float_cuda_Constant_1074(0, Constant_1074_0);
 // name=Constant_1084
Constant_float_cuda_Constant_1084(0, Constant_1084_0);
 // name=encoder.layer.8.layernorm_before.weight
Constant_float_cuda_Constant_90(0, Constant_90_0);
 // name=encoder.layer.8.layernorm_before.bias
Constant_float_cuda_Constant_91(0, Constant_91_0);
 // name=encoder.layer.8.attention.attention.value.bias
Constant_float_cuda_Constant_86(0, Constant_86_0);
 // name=onnx::MatMul_1938
Constant_float_cuda_Constant_175(0, Constant_175_0);
 // name=encoder.layer.8.attention.attention.key.bias
Constant_float_cuda_Constant_85(0, Constant_85_0);
 // name=onnx::MatMul_1937
Constant_float_cuda_Constant_174(0, Constant_174_0);
 // name=encoder.layer.8.attention.attention.query.bias
Constant_float_cuda_Constant_84(0, Constant_84_0);
 // name=encoder.layer.8.attention.output.dense.bias
Constant_float_cuda_Constant_87(0, Constant_87_0);
 // name=onnx::MatMul_1949
Constant_float_cuda_Constant_179(0, Constant_179_0);
 // name=onnx::MatMul_1948
Constant_float_cuda_Constant_178(0, Constant_178_0);
 // name=Constant_1140
Constant_float_cuda_Constant_1140(0, Constant_1140_0);
 // name=Constant_1150
Constant_float_cuda_Constant_1150(0, Constant_1150_0);
 // name=encoder.layer.8.layernorm_after.weight
Constant_float_cuda_Constant_92(0, Constant_92_0);
 // name=encoder.layer.8.layernorm_after.bias
Constant_float_cuda_Constant_93(0, Constant_93_0);
 // name=encoder.layer.8.intermediate.dense.bias
Constant_float_cuda_Constant_88(0, Constant_88_0);
 // name=encoder.layer.8.output.dense.bias
Constant_float_cuda_Constant_89(0, Constant_89_0);
 // name=onnx::MatMul_1960
Constant_float_cuda_Constant_183(0, Constant_183_0);
 // name=onnx::MatMul_1954
Constant_float_cuda_Constant_182(0, Constant_182_0);
 // name=Constant_1181
Constant_float_cuda_Constant_1181(0, Constant_1181_0);
 // name=Constant_1191
Constant_float_cuda_Constant_1191(0, Constant_1191_0);
 // name=encoder.layer.9.layernorm_before.weight
Constant_float_cuda_Constant_100(0, Constant_100_0);
 // name=encoder.layer.9.layernorm_before.bias
Constant_float_cuda_Constant_101(0, Constant_101_0);
 // name=encoder.layer.9.attention.attention.value.bias
Constant_float_cuda_Constant_96(0, Constant_96_0);
 // name=onnx::MatMul_1951
Constant_float_cuda_Constant_181(0, Constant_181_0);
 // name=encoder.layer.9.attention.attention.key.bias
Constant_float_cuda_Constant_95(0, Constant_95_0);
 // name=onnx::MatMul_1950
Constant_float_cuda_Constant_180(0, Constant_180_0);
 // name=encoder.layer.9.attention.attention.query.bias
Constant_float_cuda_Constant_94(0, Constant_94_0);
 // name=encoder.layer.9.attention.output.dense.bias
Constant_float_cuda_Constant_97(0, Constant_97_0);
 // name=onnx::MatMul_1962
Constant_float_cuda_Constant_185(0, Constant_185_0);
 // name=onnx::MatMul_1961
Constant_float_cuda_Constant_184(0, Constant_184_0);
 // name=Constant_1247
Constant_float_cuda_Constant_1247(0, Constant_1247_0);
 // name=Constant_1257
Constant_float_cuda_Constant_1257(0, Constant_1257_0);
 // name=encoder.layer.9.layernorm_after.weight
Constant_float_cuda_Constant_102(0, Constant_102_0);
 // name=encoder.layer.9.layernorm_after.bias
Constant_float_cuda_Constant_103(0, Constant_103_0);
 // name=encoder.layer.9.intermediate.dense.bias
Constant_float_cuda_Constant_98(0, Constant_98_0);
 // name=encoder.layer.9.output.dense.bias
Constant_float_cuda_Constant_99(0, Constant_99_0);
 // name=onnx::MatMul_1973
Constant_float_cuda_Constant_189(0, Constant_189_0);
 // name=onnx::MatMul_1967
Constant_float_cuda_Constant_188(0, Constant_188_0);
 // name=Constant_1288
Constant_float_cuda_Constant_1288(0, Constant_1288_0);
 // name=Constant_1298
Constant_float_cuda_Constant_1298(0, Constant_1298_0);
 // name=encoder.layer.10.layernorm_before.weight
Constant_float_cuda_Constant_110(0, Constant_110_0);
 // name=encoder.layer.10.layernorm_before.bias
Constant_float_cuda_Constant_111(0, Constant_111_0);
 // name=encoder.layer.10.attention.attention.value.bias
Constant_float_cuda_Constant_106(0, Constant_106_0);
 // name=onnx::MatMul_1964
Constant_float_cuda_Constant_187(0, Constant_187_0);
 // name=encoder.layer.10.attention.attention.key.bias
Constant_float_cuda_Constant_105(0, Constant_105_0);
 // name=onnx::MatMul_1963
Constant_float_cuda_Constant_186(0, Constant_186_0);
 // name=encoder.layer.10.attention.attention.query.bias
Constant_float_cuda_Constant_104(0, Constant_104_0);
 // name=encoder.layer.10.attention.output.dense.bias
Constant_float_cuda_Constant_107(0, Constant_107_0);
 // name=onnx::MatMul_1975
Constant_float_cuda_Constant_191(0, Constant_191_0);
 // name=onnx::MatMul_1974
Constant_float_cuda_Constant_190(0, Constant_190_0);
 // name=Constant_1354
Constant_float_cuda_Constant_1354(0, Constant_1354_0);
 // name=Constant_1364
Constant_float_cuda_Constant_1364(0, Constant_1364_0);
 // name=encoder.layer.10.layernorm_after.weight
Constant_float_cuda_Constant_112(0, Constant_112_0);
 // name=encoder.layer.10.layernorm_after.bias
Constant_float_cuda_Constant_113(0, Constant_113_0);
 // name=encoder.layer.10.intermediate.dense.bias
Constant_float_cuda_Constant_108(0, Constant_108_0);
 // name=encoder.layer.10.output.dense.bias
Constant_float_cuda_Constant_109(0, Constant_109_0);
 // name=onnx::MatMul_1986
Constant_float_cuda_Constant_195(0, Constant_195_0);
 // name=onnx::MatMul_1980
Constant_float_cuda_Constant_194(0, Constant_194_0);
 // name=Constant_1395
Constant_float_cuda_Constant_1395(0, Constant_1395_0);
 // name=Constant_1405
Constant_float_cuda_Constant_1405(0, Constant_1405_0);
 // name=encoder.layer.11.layernorm_before.weight
Constant_float_cuda_Constant_120(0, Constant_120_0);
 // name=encoder.layer.11.layernorm_before.bias
Constant_float_cuda_Constant_121(0, Constant_121_0);
 // name=encoder.layer.11.attention.attention.value.bias
Constant_float_cuda_Constant_116(0, Constant_116_0);
 // name=onnx::MatMul_1977
Constant_float_cuda_Constant_193(0, Constant_193_0);
 // name=encoder.layer.11.attention.attention.key.bias
Constant_float_cuda_Constant_115(0, Constant_115_0);
 // name=onnx::MatMul_1976
Constant_float_cuda_Constant_192(0, Constant_192_0);
 // name=encoder.layer.11.attention.attention.query.bias
Constant_float_cuda_Constant_114(0, Constant_114_0);
 // name=encoder.layer.11.attention.output.dense.bias
Constant_float_cuda_Constant_117(0, Constant_117_0);
 // name=onnx::MatMul_1988
Constant_float_cuda_Constant_197(0, Constant_197_0);
 // name=onnx::MatMul_1987
Constant_float_cuda_Constant_196(0, Constant_196_0);
 // name=Constant_1461
Constant_float_cuda_Constant_1461(0, Constant_1461_0);
 // name=Constant_1471
Constant_float_cuda_Constant_1471(0, Constant_1471_0);
 // name=encoder.layer.11.layernorm_after.weight
Constant_float_cuda_Constant_122(0, Constant_122_0);
 // name=encoder.layer.11.layernorm_after.bias
Constant_float_cuda_Constant_123(0, Constant_123_0);
 // name=encoder.layer.11.intermediate.dense.bias
Constant_float_cuda_Constant_118(0, Constant_118_0);
 // name=encoder.layer.11.output.dense.bias
Constant_float_cuda_Constant_119(0, Constant_119_0);
 // name=Constant_1502
Constant_float_cuda_Constant_1502(0, Constant_1502_0);
 // name=Constant_1512
Constant_float_cuda_Constant_1512(0, Constant_1512_0);
 // name=layernorm.weight
Constant_float_cuda_Constant_124(0, Constant_124_0);
 // name=layernorm.bias
Constant_float_cuda_Constant_125(0, Constant_125_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
}


extern "C" int kernel_entry(float* Parameter_207_0, float** Result_1527_0)
{
// kernel_entry_init
 // name=Reshape_214
// eliminated: Reshape_float_float_cuda_lib_Reshape_214(0, Constant_1_0, Reshape_214_0);
 // name=Broadcast_215
Broadcast_float_float_cuda_Broadcast_215_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_214_0, Broadcast_215_0);
 // name=Broadcast_209
Broadcast_float_float_cuda_Broadcast_209_Call(dim3(112896, 1, 1), dim3(64, 1, 1), 0, 0, Constant_3_0, Broadcast_209_0);
 // name=Convolution_208
Convolution_float_float_float_cuda_lib_Convolution_208(cudnn_handle_0, Parameter_207_0, Constant_2_0, Convolution_208_0);
 // name=Add_210
Add_float_float_float_cuda_Add_210_Call(dim3(14112, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_208_0, Broadcast_209_0, Add_210_0);
 // name=/embeddings/patch_embeddings/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_211(0, Add_210_0, Reshape_211_0);
 // name=/embeddings/patch_embeddings/Transpose_output_0
Reshape_float_float_cuda_Reshape_212_Call(dim3(13, 48, 48), dim3(16, 16, 1), 0, 0, Reshape_211_0, Reshape_212_0);
 // name=/embeddings/Concat_1_output_0
Concat_float_float_float_cuda_Concat_213_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_200_0, Reshape_212_0, Concat_213_0);
 // name=/embeddings/Add_output_0
Add_float_float_float_cuda_Add_210_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Concat_213_0, Broadcast_215_0, Add_216_0);
 // name=Sum_217
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_216_0, Sum_217_0);
 // name=ElementWiseFused_1631
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_218_0, Sum_217_0, Divide_220_0);
 // name=/encoder/layer.0/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_221(0, Divide_220_0, Reshape_221_0);
 // name=Reshape_222
// eliminated: Reshape_float_float_cuda_lib_Reshape_222(0, Reshape_221_0, Reshape_222_0);
 // name=ElementWiseFused_1632
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_222_0, Add_216_0, Subtract_224_0, Power_226_0);
 // name=Sum_227
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_226_0, Sum_227_0);
 // name=Reshape_1516
// eliminated: Reshape_float_float_cuda_lib_Reshape_1516(0, Constant_202_0, Reshape_1516_0);
 // name=ElementWiseFused_1633
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_228_0, Sum_227_0, Sqrt_235_0);
 // name=Reshape_236
// eliminated: Reshape_float_float_cuda_lib_Reshape_236(0, Sqrt_235_0, Reshape_236_0);
 // name=ElementWiseFused_1634
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_11_0, Constant_10_0, Reshape_236_0, Subtract_224_0, Add_242_0);
 // name=/encoder/layer.0/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_242_0, Constant_128_0, Dot_266_0);
 // name=ElementWiseFused_1637
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_6_0, Dot_266_0, Add_268_0);
 // name=/encoder/layer.0/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_269(0, Add_268_0, Reshape_269_0);
 // name=/encoder/layer.0/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_269_0, Reshape_270_0);
 // name=Reshape_272
// eliminated: Reshape_float_float_cuda_lib_Reshape_272(0, Reshape_270_0, Reshape_272_0);
 // name=Broadcast_274
// eliminated: Broadcast_float_float_cuda_Broadcast_274_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_272_0, Broadcast_274_0);
 // name=/encoder/layer.0/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_242_0, Constant_127_0, Dot_243_0);
 // name=ElementWiseFused_1635
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_5_0, Dot_243_0, Add_245_0);
 // name=/encoder/layer.0/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_246(0, Add_245_0, Reshape_246_0);
 // name=/encoder/layer.0/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_246_0, Reshape_247_0);
 // name=Reshape_248
// eliminated: Reshape_float_float_cuda_lib_Reshape_248(0, Constant_201_0, Reshape_248_0);
 // name=ElementWiseFused_1639
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_247_0, Multiply_250_0);
 // name=Reshape_260
// eliminated: Reshape_float_float_cuda_lib_Reshape_260(0, Multiply_250_0, Reshape_260_0);
 // name=Broadcast_262
// eliminated: Broadcast_float_float_cuda_Broadcast_262_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_260_0, Broadcast_262_0);
 // name=/encoder/layer.0/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_242_0, Constant_126_0, Dot_251_0);
 // name=ElementWiseFused_1636
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_4_0, Dot_251_0, Add_253_0);
 // name=/encoder/layer.0/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_254(0, Add_253_0, Reshape_254_0);
 // name=/encoder/layer.0/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_254_0, Reshape_255_0);
 // name=ElementWiseFused_1638
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_255_0, Multiply_258_0);
 // name=Reshape_259
// eliminated: Reshape_float_float_cuda_lib_Reshape_259(0, Multiply_258_0, Reshape_259_0);
 // name=Broadcast_261
// eliminated: Broadcast_float_float_cuda_Broadcast_261_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_259_0, Broadcast_261_0);
 // name=/encoder/layer.0/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_261_0, Broadcast_262_0, BatchMatMul_263_0);
 // name=Reshape_264
// eliminated: Reshape_float_float_cuda_lib_Reshape_264(0, BatchMatMul_263_0, Reshape_264_0);
 // name=/encoder/layer.0/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_264_0, Softmax_265_0);
 // name=Reshape_271
// eliminated: Reshape_float_float_cuda_lib_Reshape_271(0, Softmax_265_0, Reshape_271_0);
 // name=Broadcast_273
// eliminated: Broadcast_float_float_cuda_Broadcast_273_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_271_0, Broadcast_273_0);
 // name=/encoder/layer.0/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_273_0, Broadcast_274_0, BatchMatMul_275_0);
 // name=Reshape_276
// eliminated: Reshape_float_float_cuda_lib_Reshape_276(0, BatchMatMul_275_0, Reshape_276_0);
 // name=/encoder/layer.0/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_276_0, Reshape_277_0);
 // name=/encoder/layer.0/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_278(0, Reshape_277_0, Reshape_278_0);
 // name=/encoder/layer.0/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_278_0, Constant_129_0, Dot_279_0);
 // name=ElementWiseFused_1640
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_7_0, Dot_279_0, Add_216_0, Add_282_0);
 // name=Sum_283
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_282_0, Sum_283_0);
 // name=ElementWiseFused_1641
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_284_0, Sum_283_0, Divide_286_0);
 // name=/encoder/layer.0/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_287(0, Divide_286_0, Reshape_287_0);
 // name=Reshape_288
// eliminated: Reshape_float_float_cuda_lib_Reshape_288(0, Reshape_287_0, Reshape_288_0);
 // name=ElementWiseFused_1642
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_288_0, Add_282_0, Subtract_290_0, Power_292_0);
 // name=Sum_293
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_292_0, Sum_293_0);
 // name=ElementWiseFused_1643
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_294_0, Sum_293_0, Sqrt_301_0);
 // name=Reshape_302
// eliminated: Reshape_float_float_cuda_lib_Reshape_302(0, Sqrt_301_0, Reshape_302_0);
 // name=ElementWiseFused_1644
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_13_0, Constant_12_0, Reshape_302_0, Subtract_290_0, Add_308_0);
 // name=/encoder/layer.0/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_308_0, Constant_130_0, Dot_309_0);
 // name=ElementWiseFused_1645
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_8_0, Dot_309_0, Multiply_319_0);
 // name=/encoder/layer.0/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_319_0, Constant_131_0, Dot_320_0);
 // name=ElementWiseFused_1646
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_9_0, Dot_320_0, Add_282_0, Add_323_0);
 // name=Sum_324
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_323_0, Sum_324_0);
 // name=ElementWiseFused_1647
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_325_0, Sum_324_0, Divide_327_0);
 // name=/encoder/layer.1/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_328(0, Divide_327_0, Reshape_328_0);
 // name=Reshape_329
// eliminated: Reshape_float_float_cuda_lib_Reshape_329(0, Reshape_328_0, Reshape_329_0);
 // name=ElementWiseFused_1648
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_329_0, Add_323_0, Subtract_331_0, Power_333_0);
 // name=Sum_334
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_333_0, Sum_334_0);
 // name=ElementWiseFused_1649
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_335_0, Sum_334_0, Sqrt_342_0);
 // name=Reshape_343
// eliminated: Reshape_float_float_cuda_lib_Reshape_343(0, Sqrt_342_0, Reshape_343_0);
 // name=ElementWiseFused_1650
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_21_0, Constant_20_0, Reshape_343_0, Subtract_331_0, Add_349_0);
 // name=/encoder/layer.1/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_349_0, Constant_134_0, Dot_373_0);
 // name=ElementWiseFused_1653
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_16_0, Dot_373_0, Add_375_0);
 // name=/encoder/layer.1/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_376(0, Add_375_0, Reshape_376_0);
 // name=/encoder/layer.1/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_376_0, Reshape_377_0);
 // name=Reshape_379
// eliminated: Reshape_float_float_cuda_lib_Reshape_379(0, Reshape_377_0, Reshape_379_0);
 // name=Broadcast_381
// eliminated: Broadcast_float_float_cuda_Broadcast_381_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_379_0, Broadcast_381_0);
 // name=/encoder/layer.1/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_349_0, Constant_133_0, Dot_350_0);
 // name=ElementWiseFused_1651
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_15_0, Dot_350_0, Add_352_0);
 // name=/encoder/layer.1/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_353(0, Add_352_0, Reshape_353_0);
 // name=/encoder/layer.1/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_353_0, Reshape_354_0);
 // name=ElementWiseFused_1655
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_354_0, Multiply_357_0);
 // name=Reshape_367
// eliminated: Reshape_float_float_cuda_lib_Reshape_367(0, Multiply_357_0, Reshape_367_0);
 // name=Broadcast_369
// eliminated: Broadcast_float_float_cuda_Broadcast_369_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_367_0, Broadcast_369_0);
 // name=/encoder/layer.1/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_349_0, Constant_132_0, Dot_358_0);
 // name=ElementWiseFused_1652
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_14_0, Dot_358_0, Add_360_0);
 // name=/encoder/layer.1/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_361(0, Add_360_0, Reshape_361_0);
 // name=/encoder/layer.1/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_361_0, Reshape_362_0);
 // name=ElementWiseFused_1654
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_362_0, Multiply_365_0);
 // name=Reshape_366
// eliminated: Reshape_float_float_cuda_lib_Reshape_366(0, Multiply_365_0, Reshape_366_0);
 // name=Broadcast_368
// eliminated: Broadcast_float_float_cuda_Broadcast_368_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_366_0, Broadcast_368_0);
 // name=/encoder/layer.1/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_368_0, Broadcast_369_0, BatchMatMul_370_0);
 // name=Reshape_371
// eliminated: Reshape_float_float_cuda_lib_Reshape_371(0, BatchMatMul_370_0, Reshape_371_0);
 // name=/encoder/layer.1/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_371_0, Softmax_372_0);
 // name=Reshape_378
// eliminated: Reshape_float_float_cuda_lib_Reshape_378(0, Softmax_372_0, Reshape_378_0);
 // name=Broadcast_380
// eliminated: Broadcast_float_float_cuda_Broadcast_380_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_378_0, Broadcast_380_0);
 // name=/encoder/layer.1/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_380_0, Broadcast_381_0, BatchMatMul_382_0);
 // name=Reshape_383
// eliminated: Reshape_float_float_cuda_lib_Reshape_383(0, BatchMatMul_382_0, Reshape_383_0);
 // name=/encoder/layer.1/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_383_0, Reshape_384_0);
 // name=/encoder/layer.1/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_385(0, Reshape_384_0, Reshape_385_0);
 // name=/encoder/layer.1/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_385_0, Constant_135_0, Dot_386_0);
 // name=ElementWiseFused_1656
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_17_0, Dot_386_0, Add_323_0, Add_389_0);
 // name=Sum_390
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_389_0, Sum_390_0);
 // name=ElementWiseFused_1657
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_391_0, Sum_390_0, Divide_393_0);
 // name=/encoder/layer.1/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_394(0, Divide_393_0, Reshape_394_0);
 // name=Reshape_395
// eliminated: Reshape_float_float_cuda_lib_Reshape_395(0, Reshape_394_0, Reshape_395_0);
 // name=ElementWiseFused_1658
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_395_0, Add_389_0, Subtract_397_0, Power_399_0);
 // name=Sum_400
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_399_0, Sum_400_0);
 // name=ElementWiseFused_1659
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_401_0, Sum_400_0, Sqrt_408_0);
 // name=Reshape_409
// eliminated: Reshape_float_float_cuda_lib_Reshape_409(0, Sqrt_408_0, Reshape_409_0);
 // name=ElementWiseFused_1660
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_23_0, Constant_22_0, Reshape_409_0, Subtract_397_0, Add_415_0);
 // name=/encoder/layer.1/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_415_0, Constant_136_0, Dot_416_0);
 // name=ElementWiseFused_1661
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_18_0, Dot_416_0, Multiply_426_0);
 // name=/encoder/layer.1/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_426_0, Constant_137_0, Dot_427_0);
 // name=ElementWiseFused_1662
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_19_0, Dot_427_0, Add_389_0, Add_430_0);
 // name=Sum_431
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_430_0, Sum_431_0);
 // name=ElementWiseFused_1663
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_432_0, Sum_431_0, Divide_434_0);
 // name=/encoder/layer.2/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_435(0, Divide_434_0, Reshape_435_0);
 // name=Reshape_436
// eliminated: Reshape_float_float_cuda_lib_Reshape_436(0, Reshape_435_0, Reshape_436_0);
 // name=ElementWiseFused_1664
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_436_0, Add_430_0, Subtract_438_0, Power_440_0);
 // name=Sum_441
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_440_0, Sum_441_0);
 // name=ElementWiseFused_1665
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_442_0, Sum_441_0, Sqrt_449_0);
 // name=Reshape_450
// eliminated: Reshape_float_float_cuda_lib_Reshape_450(0, Sqrt_449_0, Reshape_450_0);
 // name=ElementWiseFused_1666
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_31_0, Constant_30_0, Reshape_450_0, Subtract_438_0, Add_456_0);
 // name=/encoder/layer.2/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_456_0, Constant_140_0, Dot_480_0);
 // name=ElementWiseFused_1669
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_26_0, Dot_480_0, Add_482_0);
 // name=/encoder/layer.2/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_483(0, Add_482_0, Reshape_483_0);
 // name=/encoder/layer.2/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_483_0, Reshape_484_0);
 // name=Reshape_486
// eliminated: Reshape_float_float_cuda_lib_Reshape_486(0, Reshape_484_0, Reshape_486_0);
 // name=Broadcast_488
// eliminated: Broadcast_float_float_cuda_Broadcast_488_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_486_0, Broadcast_488_0);
 // name=/encoder/layer.2/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_456_0, Constant_139_0, Dot_457_0);
 // name=ElementWiseFused_1667
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_25_0, Dot_457_0, Add_459_0);
 // name=/encoder/layer.2/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_460(0, Add_459_0, Reshape_460_0);
 // name=/encoder/layer.2/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_460_0, Reshape_461_0);
 // name=ElementWiseFused_1671
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_461_0, Multiply_464_0);
 // name=Reshape_474
// eliminated: Reshape_float_float_cuda_lib_Reshape_474(0, Multiply_464_0, Reshape_474_0);
 // name=Broadcast_476
// eliminated: Broadcast_float_float_cuda_Broadcast_476_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_474_0, Broadcast_476_0);
 // name=/encoder/layer.2/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_456_0, Constant_138_0, Dot_465_0);
 // name=ElementWiseFused_1668
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_24_0, Dot_465_0, Add_467_0);
 // name=/encoder/layer.2/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_468(0, Add_467_0, Reshape_468_0);
 // name=/encoder/layer.2/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_468_0, Reshape_469_0);
 // name=ElementWiseFused_1670
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_469_0, Multiply_472_0);
 // name=Reshape_473
// eliminated: Reshape_float_float_cuda_lib_Reshape_473(0, Multiply_472_0, Reshape_473_0);
 // name=Broadcast_475
// eliminated: Broadcast_float_float_cuda_Broadcast_475_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_473_0, Broadcast_475_0);
 // name=/encoder/layer.2/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_475_0, Broadcast_476_0, BatchMatMul_477_0);
 // name=Reshape_478
// eliminated: Reshape_float_float_cuda_lib_Reshape_478(0, BatchMatMul_477_0, Reshape_478_0);
 // name=/encoder/layer.2/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_478_0, Softmax_479_0);
 // name=Reshape_485
// eliminated: Reshape_float_float_cuda_lib_Reshape_485(0, Softmax_479_0, Reshape_485_0);
 // name=Broadcast_487
// eliminated: Broadcast_float_float_cuda_Broadcast_487_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_485_0, Broadcast_487_0);
 // name=/encoder/layer.2/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_487_0, Broadcast_488_0, BatchMatMul_489_0);
 // name=Reshape_490
// eliminated: Reshape_float_float_cuda_lib_Reshape_490(0, BatchMatMul_489_0, Reshape_490_0);
 // name=/encoder/layer.2/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_490_0, Reshape_491_0);
 // name=/encoder/layer.2/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_492(0, Reshape_491_0, Reshape_492_0);
 // name=/encoder/layer.2/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_492_0, Constant_141_0, Dot_493_0);
 // name=ElementWiseFused_1672
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_27_0, Dot_493_0, Add_430_0, Add_496_0);
 // name=Sum_497
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_496_0, Sum_497_0);
 // name=ElementWiseFused_1673
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_498_0, Sum_497_0, Divide_500_0);
 // name=/encoder/layer.2/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_501(0, Divide_500_0, Reshape_501_0);
 // name=Reshape_502
// eliminated: Reshape_float_float_cuda_lib_Reshape_502(0, Reshape_501_0, Reshape_502_0);
 // name=ElementWiseFused_1674
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_502_0, Add_496_0, Subtract_504_0, Power_506_0);
 // name=Sum_507
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_506_0, Sum_507_0);
 // name=ElementWiseFused_1675
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_508_0, Sum_507_0, Sqrt_515_0);
 // name=Reshape_516
// eliminated: Reshape_float_float_cuda_lib_Reshape_516(0, Sqrt_515_0, Reshape_516_0);
 // name=ElementWiseFused_1676
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_33_0, Constant_32_0, Reshape_516_0, Subtract_504_0, Add_522_0);
 // name=/encoder/layer.2/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_522_0, Constant_142_0, Dot_523_0);
 // name=ElementWiseFused_1677
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_28_0, Dot_523_0, Multiply_533_0);
 // name=/encoder/layer.2/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_533_0, Constant_143_0, Dot_534_0);
 // name=ElementWiseFused_1678
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_29_0, Dot_534_0, Add_496_0, Add_537_0);
 // name=Sum_538
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_537_0, Sum_538_0);
 // name=ElementWiseFused_1679
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_539_0, Sum_538_0, Divide_541_0);
 // name=/encoder/layer.3/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_542(0, Divide_541_0, Reshape_542_0);
 // name=Reshape_543
// eliminated: Reshape_float_float_cuda_lib_Reshape_543(0, Reshape_542_0, Reshape_543_0);
 // name=ElementWiseFused_1680
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_543_0, Add_537_0, Subtract_545_0, Power_547_0);
 // name=Sum_548
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_547_0, Sum_548_0);
 // name=ElementWiseFused_1681
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_549_0, Sum_548_0, Sqrt_556_0);
 // name=Reshape_557
// eliminated: Reshape_float_float_cuda_lib_Reshape_557(0, Sqrt_556_0, Reshape_557_0);
 // name=ElementWiseFused_1682
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_41_0, Constant_40_0, Reshape_557_0, Subtract_545_0, Add_563_0);
 // name=/encoder/layer.3/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_563_0, Constant_146_0, Dot_587_0);
 // name=ElementWiseFused_1685
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_36_0, Dot_587_0, Add_589_0);
 // name=/encoder/layer.3/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_590(0, Add_589_0, Reshape_590_0);
 // name=/encoder/layer.3/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_590_0, Reshape_591_0);
 // name=Reshape_593
// eliminated: Reshape_float_float_cuda_lib_Reshape_593(0, Reshape_591_0, Reshape_593_0);
 // name=Broadcast_595
// eliminated: Broadcast_float_float_cuda_Broadcast_595_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_593_0, Broadcast_595_0);
 // name=/encoder/layer.3/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_563_0, Constant_145_0, Dot_564_0);
 // name=ElementWiseFused_1683
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_35_0, Dot_564_0, Add_566_0);
 // name=/encoder/layer.3/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_567(0, Add_566_0, Reshape_567_0);
 // name=/encoder/layer.3/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_567_0, Reshape_568_0);
 // name=ElementWiseFused_1687
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_568_0, Multiply_571_0);
 // name=Reshape_581
// eliminated: Reshape_float_float_cuda_lib_Reshape_581(0, Multiply_571_0, Reshape_581_0);
 // name=Broadcast_583
// eliminated: Broadcast_float_float_cuda_Broadcast_583_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_581_0, Broadcast_583_0);
 // name=/encoder/layer.3/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_563_0, Constant_144_0, Dot_572_0);
 // name=ElementWiseFused_1684
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_34_0, Dot_572_0, Add_574_0);
 // name=/encoder/layer.3/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_575(0, Add_574_0, Reshape_575_0);
 // name=/encoder/layer.3/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_575_0, Reshape_576_0);
 // name=ElementWiseFused_1686
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_576_0, Multiply_579_0);
 // name=Reshape_580
// eliminated: Reshape_float_float_cuda_lib_Reshape_580(0, Multiply_579_0, Reshape_580_0);
 // name=Broadcast_582
// eliminated: Broadcast_float_float_cuda_Broadcast_582_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_580_0, Broadcast_582_0);
 // name=/encoder/layer.3/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_582_0, Broadcast_583_0, BatchMatMul_584_0);
 // name=Reshape_585
// eliminated: Reshape_float_float_cuda_lib_Reshape_585(0, BatchMatMul_584_0, Reshape_585_0);
 // name=/encoder/layer.3/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_585_0, Softmax_586_0);
 // name=Reshape_592
// eliminated: Reshape_float_float_cuda_lib_Reshape_592(0, Softmax_586_0, Reshape_592_0);
 // name=Broadcast_594
// eliminated: Broadcast_float_float_cuda_Broadcast_594_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_592_0, Broadcast_594_0);
 // name=/encoder/layer.3/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_594_0, Broadcast_595_0, BatchMatMul_596_0);
 // name=Reshape_597
// eliminated: Reshape_float_float_cuda_lib_Reshape_597(0, BatchMatMul_596_0, Reshape_597_0);
 // name=/encoder/layer.3/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_597_0, Reshape_598_0);
 // name=/encoder/layer.3/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_599(0, Reshape_598_0, Reshape_599_0);
 // name=/encoder/layer.3/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_599_0, Constant_147_0, Dot_600_0);
 // name=ElementWiseFused_1688
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_37_0, Dot_600_0, Add_537_0, Add_603_0);
 // name=Sum_604
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_603_0, Sum_604_0);
 // name=ElementWiseFused_1689
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_605_0, Sum_604_0, Divide_607_0);
 // name=/encoder/layer.3/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_608(0, Divide_607_0, Reshape_608_0);
 // name=Reshape_609
// eliminated: Reshape_float_float_cuda_lib_Reshape_609(0, Reshape_608_0, Reshape_609_0);
 // name=ElementWiseFused_1690
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_609_0, Add_603_0, Subtract_611_0, Power_613_0);
 // name=Sum_614
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_613_0, Sum_614_0);
 // name=ElementWiseFused_1691
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_615_0, Sum_614_0, Sqrt_622_0);
 // name=Reshape_623
// eliminated: Reshape_float_float_cuda_lib_Reshape_623(0, Sqrt_622_0, Reshape_623_0);
 // name=ElementWiseFused_1692
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_43_0, Constant_42_0, Reshape_623_0, Subtract_611_0, Add_629_0);
 // name=/encoder/layer.3/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_629_0, Constant_148_0, Dot_630_0);
 // name=ElementWiseFused_1693
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_38_0, Dot_630_0, Multiply_640_0);
 // name=/encoder/layer.3/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_640_0, Constant_149_0, Dot_641_0);
 // name=ElementWiseFused_1694
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_39_0, Dot_641_0, Add_603_0, Add_644_0);
 // name=Sum_645
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_644_0, Sum_645_0);
 // name=ElementWiseFused_1695
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_646_0, Sum_645_0, Divide_648_0);
 // name=/encoder/layer.4/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_649(0, Divide_648_0, Reshape_649_0);
 // name=Reshape_650
// eliminated: Reshape_float_float_cuda_lib_Reshape_650(0, Reshape_649_0, Reshape_650_0);
 // name=ElementWiseFused_1696
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_650_0, Add_644_0, Subtract_652_0, Power_654_0);
 // name=Sum_655
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_654_0, Sum_655_0);
 // name=ElementWiseFused_1697
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_656_0, Sum_655_0, Sqrt_663_0);
 // name=Reshape_664
// eliminated: Reshape_float_float_cuda_lib_Reshape_664(0, Sqrt_663_0, Reshape_664_0);
 // name=ElementWiseFused_1698
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_51_0, Constant_50_0, Reshape_664_0, Subtract_652_0, Add_670_0);
 // name=/encoder/layer.4/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_670_0, Constant_152_0, Dot_694_0);
 // name=ElementWiseFused_1701
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_46_0, Dot_694_0, Add_696_0);
 // name=/encoder/layer.4/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_697(0, Add_696_0, Reshape_697_0);
 // name=/encoder/layer.4/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_697_0, Reshape_698_0);
 // name=Reshape_700
// eliminated: Reshape_float_float_cuda_lib_Reshape_700(0, Reshape_698_0, Reshape_700_0);
 // name=Broadcast_702
// eliminated: Broadcast_float_float_cuda_Broadcast_702_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_700_0, Broadcast_702_0);
 // name=/encoder/layer.4/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_670_0, Constant_151_0, Dot_671_0);
 // name=ElementWiseFused_1699
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_45_0, Dot_671_0, Add_673_0);
 // name=/encoder/layer.4/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_674(0, Add_673_0, Reshape_674_0);
 // name=/encoder/layer.4/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_674_0, Reshape_675_0);
 // name=ElementWiseFused_1703
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_675_0, Multiply_678_0);
 // name=Reshape_688
// eliminated: Reshape_float_float_cuda_lib_Reshape_688(0, Multiply_678_0, Reshape_688_0);
 // name=Broadcast_690
// eliminated: Broadcast_float_float_cuda_Broadcast_690_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_688_0, Broadcast_690_0);
 // name=/encoder/layer.4/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_670_0, Constant_150_0, Dot_679_0);
 // name=ElementWiseFused_1700
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_44_0, Dot_679_0, Add_681_0);
 // name=/encoder/layer.4/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_682(0, Add_681_0, Reshape_682_0);
 // name=/encoder/layer.4/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_682_0, Reshape_683_0);
 // name=ElementWiseFused_1702
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_683_0, Multiply_686_0);
 // name=Reshape_687
// eliminated: Reshape_float_float_cuda_lib_Reshape_687(0, Multiply_686_0, Reshape_687_0);
 // name=Broadcast_689
// eliminated: Broadcast_float_float_cuda_Broadcast_689_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_687_0, Broadcast_689_0);
 // name=/encoder/layer.4/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_689_0, Broadcast_690_0, BatchMatMul_691_0);
 // name=Reshape_692
// eliminated: Reshape_float_float_cuda_lib_Reshape_692(0, BatchMatMul_691_0, Reshape_692_0);
 // name=/encoder/layer.4/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_692_0, Softmax_693_0);
 // name=Reshape_699
// eliminated: Reshape_float_float_cuda_lib_Reshape_699(0, Softmax_693_0, Reshape_699_0);
 // name=Broadcast_701
// eliminated: Broadcast_float_float_cuda_Broadcast_701_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_699_0, Broadcast_701_0);
 // name=/encoder/layer.4/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_701_0, Broadcast_702_0, BatchMatMul_703_0);
 // name=Reshape_704
// eliminated: Reshape_float_float_cuda_lib_Reshape_704(0, BatchMatMul_703_0, Reshape_704_0);
 // name=/encoder/layer.4/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_704_0, Reshape_705_0);
 // name=/encoder/layer.4/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_706(0, Reshape_705_0, Reshape_706_0);
 // name=/encoder/layer.4/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_706_0, Constant_153_0, Dot_707_0);
 // name=ElementWiseFused_1704
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_47_0, Dot_707_0, Add_644_0, Add_710_0);
 // name=Sum_711
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_710_0, Sum_711_0);
 // name=ElementWiseFused_1705
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_712_0, Sum_711_0, Divide_714_0);
 // name=/encoder/layer.4/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_715(0, Divide_714_0, Reshape_715_0);
 // name=Reshape_716
// eliminated: Reshape_float_float_cuda_lib_Reshape_716(0, Reshape_715_0, Reshape_716_0);
 // name=ElementWiseFused_1706
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_716_0, Add_710_0, Subtract_718_0, Power_720_0);
 // name=Sum_721
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_720_0, Sum_721_0);
 // name=ElementWiseFused_1707
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_722_0, Sum_721_0, Sqrt_729_0);
 // name=Reshape_730
// eliminated: Reshape_float_float_cuda_lib_Reshape_730(0, Sqrt_729_0, Reshape_730_0);
 // name=ElementWiseFused_1708
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_53_0, Constant_52_0, Reshape_730_0, Subtract_718_0, Add_736_0);
 // name=/encoder/layer.4/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_736_0, Constant_154_0, Dot_737_0);
 // name=ElementWiseFused_1709
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_48_0, Dot_737_0, Multiply_747_0);
 // name=/encoder/layer.4/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_747_0, Constant_155_0, Dot_748_0);
 // name=ElementWiseFused_1710
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_49_0, Dot_748_0, Add_710_0, Add_751_0);
 // name=Sum_752
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_751_0, Sum_752_0);
 // name=ElementWiseFused_1711
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_753_0, Sum_752_0, Divide_755_0);
 // name=/encoder/layer.5/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_756(0, Divide_755_0, Reshape_756_0);
 // name=Reshape_757
// eliminated: Reshape_float_float_cuda_lib_Reshape_757(0, Reshape_756_0, Reshape_757_0);
 // name=ElementWiseFused_1712
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_757_0, Add_751_0, Subtract_759_0, Power_761_0);
 // name=Sum_762
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_761_0, Sum_762_0);
 // name=ElementWiseFused_1713
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_763_0, Sum_762_0, Sqrt_770_0);
 // name=Reshape_771
// eliminated: Reshape_float_float_cuda_lib_Reshape_771(0, Sqrt_770_0, Reshape_771_0);
 // name=ElementWiseFused_1714
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_61_0, Constant_60_0, Reshape_771_0, Subtract_759_0, Add_777_0);
 // name=/encoder/layer.5/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_777_0, Constant_158_0, Dot_801_0);
 // name=ElementWiseFused_1717
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_56_0, Dot_801_0, Add_803_0);
 // name=/encoder/layer.5/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_804(0, Add_803_0, Reshape_804_0);
 // name=/encoder/layer.5/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_804_0, Reshape_805_0);
 // name=Reshape_807
// eliminated: Reshape_float_float_cuda_lib_Reshape_807(0, Reshape_805_0, Reshape_807_0);
 // name=Broadcast_809
// eliminated: Broadcast_float_float_cuda_Broadcast_809_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_807_0, Broadcast_809_0);
 // name=/encoder/layer.5/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_777_0, Constant_157_0, Dot_778_0);
 // name=ElementWiseFused_1715
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_55_0, Dot_778_0, Add_780_0);
 // name=/encoder/layer.5/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_781(0, Add_780_0, Reshape_781_0);
 // name=/encoder/layer.5/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_781_0, Reshape_782_0);
 // name=ElementWiseFused_1719
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_782_0, Multiply_785_0);
 // name=Reshape_795
// eliminated: Reshape_float_float_cuda_lib_Reshape_795(0, Multiply_785_0, Reshape_795_0);
 // name=Broadcast_797
// eliminated: Broadcast_float_float_cuda_Broadcast_797_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_795_0, Broadcast_797_0);
 // name=/encoder/layer.5/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_777_0, Constant_156_0, Dot_786_0);
 // name=ElementWiseFused_1716
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_54_0, Dot_786_0, Add_788_0);
 // name=/encoder/layer.5/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_789(0, Add_788_0, Reshape_789_0);
 // name=/encoder/layer.5/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_789_0, Reshape_790_0);
 // name=ElementWiseFused_1718
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_790_0, Multiply_793_0);
 // name=Reshape_794
// eliminated: Reshape_float_float_cuda_lib_Reshape_794(0, Multiply_793_0, Reshape_794_0);
 // name=Broadcast_796
// eliminated: Broadcast_float_float_cuda_Broadcast_796_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_794_0, Broadcast_796_0);
 // name=/encoder/layer.5/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_796_0, Broadcast_797_0, BatchMatMul_798_0);
 // name=Reshape_799
// eliminated: Reshape_float_float_cuda_lib_Reshape_799(0, BatchMatMul_798_0, Reshape_799_0);
 // name=/encoder/layer.5/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_799_0, Softmax_800_0);
 // name=Reshape_806
// eliminated: Reshape_float_float_cuda_lib_Reshape_806(0, Softmax_800_0, Reshape_806_0);
 // name=Broadcast_808
// eliminated: Broadcast_float_float_cuda_Broadcast_808_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_806_0, Broadcast_808_0);
 // name=/encoder/layer.5/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_808_0, Broadcast_809_0, BatchMatMul_810_0);
 // name=Reshape_811
// eliminated: Reshape_float_float_cuda_lib_Reshape_811(0, BatchMatMul_810_0, Reshape_811_0);
 // name=/encoder/layer.5/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_811_0, Reshape_812_0);
 // name=/encoder/layer.5/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_813(0, Reshape_812_0, Reshape_813_0);
 // name=/encoder/layer.5/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_813_0, Constant_159_0, Dot_814_0);
 // name=ElementWiseFused_1720
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_57_0, Dot_814_0, Add_751_0, Add_817_0);
 // name=Sum_818
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_817_0, Sum_818_0);
 // name=ElementWiseFused_1721
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_819_0, Sum_818_0, Divide_821_0);
 // name=/encoder/layer.5/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_822(0, Divide_821_0, Reshape_822_0);
 // name=Reshape_823
// eliminated: Reshape_float_float_cuda_lib_Reshape_823(0, Reshape_822_0, Reshape_823_0);
 // name=ElementWiseFused_1722
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_823_0, Add_817_0, Subtract_825_0, Power_827_0);
 // name=Sum_828
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_827_0, Sum_828_0);
 // name=ElementWiseFused_1723
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_829_0, Sum_828_0, Sqrt_836_0);
 // name=Reshape_837
// eliminated: Reshape_float_float_cuda_lib_Reshape_837(0, Sqrt_836_0, Reshape_837_0);
 // name=ElementWiseFused_1724
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_63_0, Constant_62_0, Reshape_837_0, Subtract_825_0, Add_843_0);
 // name=/encoder/layer.5/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_843_0, Constant_160_0, Dot_844_0);
 // name=ElementWiseFused_1725
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_58_0, Dot_844_0, Multiply_854_0);
 // name=/encoder/layer.5/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_854_0, Constant_161_0, Dot_855_0);
 // name=ElementWiseFused_1726
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_59_0, Dot_855_0, Add_817_0, Add_858_0);
 // name=Sum_859
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_858_0, Sum_859_0);
 // name=ElementWiseFused_1727
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_860_0, Sum_859_0, Divide_862_0);
 // name=/encoder/layer.6/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_863(0, Divide_862_0, Reshape_863_0);
 // name=Reshape_864
// eliminated: Reshape_float_float_cuda_lib_Reshape_864(0, Reshape_863_0, Reshape_864_0);
 // name=ElementWiseFused_1728
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_864_0, Add_858_0, Subtract_866_0, Power_868_0);
 // name=Sum_869
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_868_0, Sum_869_0);
 // name=ElementWiseFused_1729
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_870_0, Sum_869_0, Sqrt_877_0);
 // name=Reshape_878
// eliminated: Reshape_float_float_cuda_lib_Reshape_878(0, Sqrt_877_0, Reshape_878_0);
 // name=ElementWiseFused_1730
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_71_0, Constant_70_0, Reshape_878_0, Subtract_866_0, Add_884_0);
 // name=/encoder/layer.6/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_884_0, Constant_164_0, Dot_908_0);
 // name=ElementWiseFused_1733
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_66_0, Dot_908_0, Add_910_0);
 // name=/encoder/layer.6/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_911(0, Add_910_0, Reshape_911_0);
 // name=/encoder/layer.6/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_911_0, Reshape_912_0);
 // name=Reshape_914
// eliminated: Reshape_float_float_cuda_lib_Reshape_914(0, Reshape_912_0, Reshape_914_0);
 // name=Broadcast_916
// eliminated: Broadcast_float_float_cuda_Broadcast_916_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_914_0, Broadcast_916_0);
 // name=/encoder/layer.6/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_884_0, Constant_163_0, Dot_885_0);
 // name=ElementWiseFused_1731
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_65_0, Dot_885_0, Add_887_0);
 // name=/encoder/layer.6/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_888(0, Add_887_0, Reshape_888_0);
 // name=/encoder/layer.6/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_888_0, Reshape_889_0);
 // name=ElementWiseFused_1735
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_889_0, Multiply_892_0);
 // name=Reshape_902
// eliminated: Reshape_float_float_cuda_lib_Reshape_902(0, Multiply_892_0, Reshape_902_0);
 // name=Broadcast_904
// eliminated: Broadcast_float_float_cuda_Broadcast_904_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_902_0, Broadcast_904_0);
 // name=/encoder/layer.6/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_884_0, Constant_162_0, Dot_893_0);
 // name=ElementWiseFused_1732
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_64_0, Dot_893_0, Add_895_0);
 // name=/encoder/layer.6/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_896(0, Add_895_0, Reshape_896_0);
 // name=/encoder/layer.6/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_896_0, Reshape_897_0);
 // name=ElementWiseFused_1734
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_897_0, Multiply_900_0);
 // name=Reshape_901
// eliminated: Reshape_float_float_cuda_lib_Reshape_901(0, Multiply_900_0, Reshape_901_0);
 // name=Broadcast_903
// eliminated: Broadcast_float_float_cuda_Broadcast_903_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_901_0, Broadcast_903_0);
 // name=/encoder/layer.6/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_903_0, Broadcast_904_0, BatchMatMul_905_0);
 // name=Reshape_906
// eliminated: Reshape_float_float_cuda_lib_Reshape_906(0, BatchMatMul_905_0, Reshape_906_0);
 // name=/encoder/layer.6/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_906_0, Softmax_907_0);
 // name=Reshape_913
// eliminated: Reshape_float_float_cuda_lib_Reshape_913(0, Softmax_907_0, Reshape_913_0);
 // name=Broadcast_915
// eliminated: Broadcast_float_float_cuda_Broadcast_915_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_913_0, Broadcast_915_0);
 // name=/encoder/layer.6/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_915_0, Broadcast_916_0, BatchMatMul_917_0);
 // name=Reshape_918
// eliminated: Reshape_float_float_cuda_lib_Reshape_918(0, BatchMatMul_917_0, Reshape_918_0);
 // name=/encoder/layer.6/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_918_0, Reshape_919_0);
 // name=/encoder/layer.6/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_920(0, Reshape_919_0, Reshape_920_0);
 // name=/encoder/layer.6/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_920_0, Constant_165_0, Dot_921_0);
 // name=ElementWiseFused_1736
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_67_0, Dot_921_0, Add_858_0, Add_924_0);
 // name=Sum_925
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_924_0, Sum_925_0);
 // name=ElementWiseFused_1737
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_926_0, Sum_925_0, Divide_928_0);
 // name=/encoder/layer.6/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_929(0, Divide_928_0, Reshape_929_0);
 // name=Reshape_930
// eliminated: Reshape_float_float_cuda_lib_Reshape_930(0, Reshape_929_0, Reshape_930_0);
 // name=ElementWiseFused_1738
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_930_0, Add_924_0, Subtract_932_0, Power_934_0);
 // name=Sum_935
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_934_0, Sum_935_0);
 // name=ElementWiseFused_1739
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_936_0, Sum_935_0, Sqrt_943_0);
 // name=Reshape_944
// eliminated: Reshape_float_float_cuda_lib_Reshape_944(0, Sqrt_943_0, Reshape_944_0);
 // name=ElementWiseFused_1740
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_73_0, Constant_72_0, Reshape_944_0, Subtract_932_0, Add_950_0);
 // name=/encoder/layer.6/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_950_0, Constant_166_0, Dot_951_0);
 // name=ElementWiseFused_1741
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_68_0, Dot_951_0, Multiply_961_0);
 // name=/encoder/layer.6/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_961_0, Constant_167_0, Dot_962_0);
 // name=ElementWiseFused_1742
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_69_0, Dot_962_0, Add_924_0, Add_965_0);
 // name=Sum_966
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_965_0, Sum_966_0);
 // name=ElementWiseFused_1743
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_967_0, Sum_966_0, Divide_969_0);
 // name=/encoder/layer.7/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_970(0, Divide_969_0, Reshape_970_0);
 // name=Reshape_971
// eliminated: Reshape_float_float_cuda_lib_Reshape_971(0, Reshape_970_0, Reshape_971_0);
 // name=ElementWiseFused_1744
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_971_0, Add_965_0, Subtract_973_0, Power_975_0);
 // name=Sum_976
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_975_0, Sum_976_0);
 // name=ElementWiseFused_1745
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_977_0, Sum_976_0, Sqrt_984_0);
 // name=Reshape_985
// eliminated: Reshape_float_float_cuda_lib_Reshape_985(0, Sqrt_984_0, Reshape_985_0);
 // name=ElementWiseFused_1746
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_81_0, Constant_80_0, Reshape_985_0, Subtract_973_0, Add_991_0);
 // name=/encoder/layer.7/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_991_0, Constant_170_0, Dot_1015_0);
 // name=ElementWiseFused_1749
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_76_0, Dot_1015_0, Add_1017_0);
 // name=/encoder/layer.7/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1018(0, Add_1017_0, Reshape_1018_0);
 // name=/encoder/layer.7/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1018_0, Reshape_1019_0);
 // name=Reshape_1021
// eliminated: Reshape_float_float_cuda_lib_Reshape_1021(0, Reshape_1019_0, Reshape_1021_0);
 // name=Broadcast_1023
// eliminated: Broadcast_float_float_cuda_Broadcast_1023_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1021_0, Broadcast_1023_0);
 // name=/encoder/layer.7/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_991_0, Constant_169_0, Dot_992_0);
 // name=ElementWiseFused_1747
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_75_0, Dot_992_0, Add_994_0);
 // name=/encoder/layer.7/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_995(0, Add_994_0, Reshape_995_0);
 // name=/encoder/layer.7/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_995_0, Reshape_996_0);
 // name=ElementWiseFused_1751
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_996_0, Multiply_999_0);
 // name=Reshape_1009
// eliminated: Reshape_float_float_cuda_lib_Reshape_1009(0, Multiply_999_0, Reshape_1009_0);
 // name=Broadcast_1011
// eliminated: Broadcast_float_float_cuda_Broadcast_1011_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1009_0, Broadcast_1011_0);
 // name=/encoder/layer.7/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_991_0, Constant_168_0, Dot_1000_0);
 // name=ElementWiseFused_1748
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_74_0, Dot_1000_0, Add_1002_0);
 // name=/encoder/layer.7/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1003(0, Add_1002_0, Reshape_1003_0);
 // name=/encoder/layer.7/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1003_0, Reshape_1004_0);
 // name=ElementWiseFused_1750
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1004_0, Multiply_1007_0);
 // name=Reshape_1008
// eliminated: Reshape_float_float_cuda_lib_Reshape_1008(0, Multiply_1007_0, Reshape_1008_0);
 // name=Broadcast_1010
// eliminated: Broadcast_float_float_cuda_Broadcast_1010_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1008_0, Broadcast_1010_0);
 // name=/encoder/layer.7/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_1010_0, Broadcast_1011_0, BatchMatMul_1012_0);
 // name=Reshape_1013
// eliminated: Reshape_float_float_cuda_lib_Reshape_1013(0, BatchMatMul_1012_0, Reshape_1013_0);
 // name=/encoder/layer.7/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_1013_0, Softmax_1014_0);
 // name=Reshape_1020
// eliminated: Reshape_float_float_cuda_lib_Reshape_1020(0, Softmax_1014_0, Reshape_1020_0);
 // name=Broadcast_1022
// eliminated: Broadcast_float_float_cuda_Broadcast_1022_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1020_0, Broadcast_1022_0);
 // name=/encoder/layer.7/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_1022_0, Broadcast_1023_0, BatchMatMul_1024_0);
 // name=Reshape_1025
// eliminated: Reshape_float_float_cuda_lib_Reshape_1025(0, BatchMatMul_1024_0, Reshape_1025_0);
 // name=/encoder/layer.7/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1025_0, Reshape_1026_0);
 // name=/encoder/layer.7/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1027(0, Reshape_1026_0, Reshape_1027_0);
 // name=/encoder/layer.7/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_1027_0, Constant_171_0, Dot_1028_0);
 // name=ElementWiseFused_1752
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_77_0, Dot_1028_0, Add_965_0, Add_1031_0);
 // name=Sum_1032
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1031_0, Sum_1032_0);
 // name=ElementWiseFused_1753
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1033_0, Sum_1032_0, Divide_1035_0);
 // name=/encoder/layer.7/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1036(0, Divide_1035_0, Reshape_1036_0);
 // name=Reshape_1037
// eliminated: Reshape_float_float_cuda_lib_Reshape_1037(0, Reshape_1036_0, Reshape_1037_0);
 // name=ElementWiseFused_1754
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1037_0, Add_1031_0, Subtract_1039_0, Power_1041_0);
 // name=Sum_1042
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1041_0, Sum_1042_0);
 // name=ElementWiseFused_1755
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1043_0, Sum_1042_0, Sqrt_1050_0);
 // name=Reshape_1051
// eliminated: Reshape_float_float_cuda_lib_Reshape_1051(0, Sqrt_1050_0, Reshape_1051_0);
 // name=ElementWiseFused_1756
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_83_0, Constant_82_0, Reshape_1051_0, Subtract_1039_0, Add_1057_0);
 // name=/encoder/layer.7/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_1057_0, Constant_172_0, Dot_1058_0);
 // name=ElementWiseFused_1757
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_78_0, Dot_1058_0, Multiply_1068_0);
 // name=/encoder/layer.7/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_1068_0, Constant_173_0, Dot_1069_0);
 // name=ElementWiseFused_1758
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_79_0, Dot_1069_0, Add_1031_0, Add_1072_0);
 // name=Sum_1073
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1072_0, Sum_1073_0);
 // name=ElementWiseFused_1759
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1074_0, Sum_1073_0, Divide_1076_0);
 // name=/encoder/layer.8/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1077(0, Divide_1076_0, Reshape_1077_0);
 // name=Reshape_1078
// eliminated: Reshape_float_float_cuda_lib_Reshape_1078(0, Reshape_1077_0, Reshape_1078_0);
 // name=ElementWiseFused_1760
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1078_0, Add_1072_0, Subtract_1080_0, Power_1082_0);
 // name=Sum_1083
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1082_0, Sum_1083_0);
 // name=ElementWiseFused_1761
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1084_0, Sum_1083_0, Sqrt_1091_0);
 // name=Reshape_1092
// eliminated: Reshape_float_float_cuda_lib_Reshape_1092(0, Sqrt_1091_0, Reshape_1092_0);
 // name=ElementWiseFused_1762
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_91_0, Constant_90_0, Reshape_1092_0, Subtract_1080_0, Add_1098_0);
 // name=/encoder/layer.8/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1098_0, Constant_176_0, Dot_1122_0);
 // name=ElementWiseFused_1765
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_86_0, Dot_1122_0, Add_1124_0);
 // name=/encoder/layer.8/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1125(0, Add_1124_0, Reshape_1125_0);
 // name=/encoder/layer.8/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1125_0, Reshape_1126_0);
 // name=Reshape_1128
// eliminated: Reshape_float_float_cuda_lib_Reshape_1128(0, Reshape_1126_0, Reshape_1128_0);
 // name=Broadcast_1130
// eliminated: Broadcast_float_float_cuda_Broadcast_1130_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1128_0, Broadcast_1130_0);
 // name=/encoder/layer.8/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1098_0, Constant_175_0, Dot_1099_0);
 // name=ElementWiseFused_1763
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_85_0, Dot_1099_0, Add_1101_0);
 // name=/encoder/layer.8/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1102(0, Add_1101_0, Reshape_1102_0);
 // name=/encoder/layer.8/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_1102_0, Reshape_1103_0);
 // name=ElementWiseFused_1767
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1103_0, Multiply_1106_0);
 // name=Reshape_1116
// eliminated: Reshape_float_float_cuda_lib_Reshape_1116(0, Multiply_1106_0, Reshape_1116_0);
 // name=Broadcast_1118
// eliminated: Broadcast_float_float_cuda_Broadcast_1118_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1116_0, Broadcast_1118_0);
 // name=/encoder/layer.8/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1098_0, Constant_174_0, Dot_1107_0);
 // name=ElementWiseFused_1764
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_84_0, Dot_1107_0, Add_1109_0);
 // name=/encoder/layer.8/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1110(0, Add_1109_0, Reshape_1110_0);
 // name=/encoder/layer.8/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1110_0, Reshape_1111_0);
 // name=ElementWiseFused_1766
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1111_0, Multiply_1114_0);
 // name=Reshape_1115
// eliminated: Reshape_float_float_cuda_lib_Reshape_1115(0, Multiply_1114_0, Reshape_1115_0);
 // name=Broadcast_1117
// eliminated: Broadcast_float_float_cuda_Broadcast_1117_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1115_0, Broadcast_1117_0);
 // name=/encoder/layer.8/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_1117_0, Broadcast_1118_0, BatchMatMul_1119_0);
 // name=Reshape_1120
// eliminated: Reshape_float_float_cuda_lib_Reshape_1120(0, BatchMatMul_1119_0, Reshape_1120_0);
 // name=/encoder/layer.8/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_1120_0, Softmax_1121_0);
 // name=Reshape_1127
// eliminated: Reshape_float_float_cuda_lib_Reshape_1127(0, Softmax_1121_0, Reshape_1127_0);
 // name=Broadcast_1129
// eliminated: Broadcast_float_float_cuda_Broadcast_1129_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1127_0, Broadcast_1129_0);
 // name=/encoder/layer.8/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_1129_0, Broadcast_1130_0, BatchMatMul_1131_0);
 // name=Reshape_1132
// eliminated: Reshape_float_float_cuda_lib_Reshape_1132(0, BatchMatMul_1131_0, Reshape_1132_0);
 // name=/encoder/layer.8/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1132_0, Reshape_1133_0);
 // name=/encoder/layer.8/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1134(0, Reshape_1133_0, Reshape_1134_0);
 // name=/encoder/layer.8/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_1134_0, Constant_177_0, Dot_1135_0);
 // name=ElementWiseFused_1768
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_87_0, Dot_1135_0, Add_1072_0, Add_1138_0);
 // name=Sum_1139
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1138_0, Sum_1139_0);
 // name=ElementWiseFused_1769
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1140_0, Sum_1139_0, Divide_1142_0);
 // name=/encoder/layer.8/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1143(0, Divide_1142_0, Reshape_1143_0);
 // name=Reshape_1144
// eliminated: Reshape_float_float_cuda_lib_Reshape_1144(0, Reshape_1143_0, Reshape_1144_0);
 // name=ElementWiseFused_1770
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1144_0, Add_1138_0, Subtract_1146_0, Power_1148_0);
 // name=Sum_1149
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1148_0, Sum_1149_0);
 // name=ElementWiseFused_1771
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1150_0, Sum_1149_0, Sqrt_1157_0);
 // name=Reshape_1158
// eliminated: Reshape_float_float_cuda_lib_Reshape_1158(0, Sqrt_1157_0, Reshape_1158_0);
 // name=ElementWiseFused_1772
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_93_0, Constant_92_0, Reshape_1158_0, Subtract_1146_0, Add_1164_0);
 // name=/encoder/layer.8/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_1164_0, Constant_178_0, Dot_1165_0);
 // name=ElementWiseFused_1773
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_88_0, Dot_1165_0, Multiply_1175_0);
 // name=/encoder/layer.8/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_1175_0, Constant_179_0, Dot_1176_0);
 // name=ElementWiseFused_1774
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_89_0, Dot_1176_0, Add_1138_0, Add_1179_0);
 // name=Sum_1180
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1179_0, Sum_1180_0);
 // name=ElementWiseFused_1775
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1181_0, Sum_1180_0, Divide_1183_0);
 // name=/encoder/layer.9/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1184(0, Divide_1183_0, Reshape_1184_0);
 // name=Reshape_1185
// eliminated: Reshape_float_float_cuda_lib_Reshape_1185(0, Reshape_1184_0, Reshape_1185_0);
 // name=ElementWiseFused_1776
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1185_0, Add_1179_0, Subtract_1187_0, Power_1189_0);
 // name=Sum_1190
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1189_0, Sum_1190_0);
 // name=ElementWiseFused_1777
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1191_0, Sum_1190_0, Sqrt_1198_0);
 // name=Reshape_1199
// eliminated: Reshape_float_float_cuda_lib_Reshape_1199(0, Sqrt_1198_0, Reshape_1199_0);
 // name=ElementWiseFused_1778
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_101_0, Constant_100_0, Reshape_1199_0, Subtract_1187_0, Add_1205_0);
 // name=/encoder/layer.9/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1205_0, Constant_182_0, Dot_1229_0);
 // name=ElementWiseFused_1781
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_96_0, Dot_1229_0, Add_1231_0);
 // name=/encoder/layer.9/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1232(0, Add_1231_0, Reshape_1232_0);
 // name=/encoder/layer.9/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1232_0, Reshape_1233_0);
 // name=Reshape_1235
// eliminated: Reshape_float_float_cuda_lib_Reshape_1235(0, Reshape_1233_0, Reshape_1235_0);
 // name=Broadcast_1237
// eliminated: Broadcast_float_float_cuda_Broadcast_1237_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1235_0, Broadcast_1237_0);
 // name=/encoder/layer.9/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1205_0, Constant_181_0, Dot_1206_0);
 // name=ElementWiseFused_1779
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_95_0, Dot_1206_0, Add_1208_0);
 // name=/encoder/layer.9/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1209(0, Add_1208_0, Reshape_1209_0);
 // name=/encoder/layer.9/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_1209_0, Reshape_1210_0);
 // name=ElementWiseFused_1782
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1210_0, Multiply_1213_0);
 // name=Reshape_1223
// eliminated: Reshape_float_float_cuda_lib_Reshape_1223(0, Multiply_1213_0, Reshape_1223_0);
 // name=Broadcast_1225
// eliminated: Broadcast_float_float_cuda_Broadcast_1225_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1223_0, Broadcast_1225_0);
 // name=/encoder/layer.9/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1205_0, Constant_180_0, Dot_1214_0);
 // name=ElementWiseFused_1780
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_94_0, Dot_1214_0, Add_1216_0);
 // name=/encoder/layer.9/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1217(0, Add_1216_0, Reshape_1217_0);
 // name=/encoder/layer.9/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1217_0, Reshape_1218_0);
 // name=ElementWiseFused_1783
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1218_0, Multiply_1221_0);
 // name=Reshape_1222
// eliminated: Reshape_float_float_cuda_lib_Reshape_1222(0, Multiply_1221_0, Reshape_1222_0);
 // name=Broadcast_1224
// eliminated: Broadcast_float_float_cuda_Broadcast_1224_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1222_0, Broadcast_1224_0);
 // name=/encoder/layer.9/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_1224_0, Broadcast_1225_0, BatchMatMul_1226_0);
 // name=Reshape_1227
// eliminated: Reshape_float_float_cuda_lib_Reshape_1227(0, BatchMatMul_1226_0, Reshape_1227_0);
 // name=/encoder/layer.9/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_1227_0, Softmax_1228_0);
 // name=Reshape_1234
// eliminated: Reshape_float_float_cuda_lib_Reshape_1234(0, Softmax_1228_0, Reshape_1234_0);
 // name=Broadcast_1236
// eliminated: Broadcast_float_float_cuda_Broadcast_1236_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1234_0, Broadcast_1236_0);
 // name=/encoder/layer.9/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_1236_0, Broadcast_1237_0, BatchMatMul_1238_0);
 // name=Reshape_1239
// eliminated: Reshape_float_float_cuda_lib_Reshape_1239(0, BatchMatMul_1238_0, Reshape_1239_0);
 // name=/encoder/layer.9/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1239_0, Reshape_1240_0);
 // name=/encoder/layer.9/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1241(0, Reshape_1240_0, Reshape_1241_0);
 // name=/encoder/layer.9/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_1241_0, Constant_183_0, Dot_1242_0);
 // name=ElementWiseFused_1784
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_97_0, Dot_1242_0, Add_1179_0, Add_1245_0);
 // name=Sum_1246
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1245_0, Sum_1246_0);
 // name=ElementWiseFused_1785
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1247_0, Sum_1246_0, Divide_1249_0);
 // name=/encoder/layer.9/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1250(0, Divide_1249_0, Reshape_1250_0);
 // name=Reshape_1251
// eliminated: Reshape_float_float_cuda_lib_Reshape_1251(0, Reshape_1250_0, Reshape_1251_0);
 // name=ElementWiseFused_1786
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1251_0, Add_1245_0, Subtract_1253_0, Power_1255_0);
 // name=Sum_1256
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1255_0, Sum_1256_0);
 // name=ElementWiseFused_1787
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1257_0, Sum_1256_0, Sqrt_1264_0);
 // name=Reshape_1265
// eliminated: Reshape_float_float_cuda_lib_Reshape_1265(0, Sqrt_1264_0, Reshape_1265_0);
 // name=ElementWiseFused_1788
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_103_0, Constant_102_0, Reshape_1265_0, Subtract_1253_0, Add_1271_0);
 // name=/encoder/layer.9/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_1271_0, Constant_184_0, Dot_1272_0);
 // name=ElementWiseFused_1789
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_98_0, Dot_1272_0, Multiply_1282_0);
 // name=/encoder/layer.9/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_1282_0, Constant_185_0, Dot_1283_0);
 // name=ElementWiseFused_1790
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_99_0, Dot_1283_0, Add_1245_0, Add_1286_0);
 // name=Sum_1287
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1286_0, Sum_1287_0);
 // name=ElementWiseFused_1791
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1288_0, Sum_1287_0, Divide_1290_0);
 // name=/encoder/layer.10/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1291(0, Divide_1290_0, Reshape_1291_0);
 // name=Reshape_1292
// eliminated: Reshape_float_float_cuda_lib_Reshape_1292(0, Reshape_1291_0, Reshape_1292_0);
 // name=ElementWiseFused_1792
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1292_0, Add_1286_0, Subtract_1294_0, Power_1296_0);
 // name=Sum_1297
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1296_0, Sum_1297_0);
 // name=ElementWiseFused_1793
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1298_0, Sum_1297_0, Sqrt_1305_0);
 // name=Reshape_1306
// eliminated: Reshape_float_float_cuda_lib_Reshape_1306(0, Sqrt_1305_0, Reshape_1306_0);
 // name=ElementWiseFused_1794
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_111_0, Constant_110_0, Reshape_1306_0, Subtract_1294_0, Add_1312_0);
 // name=/encoder/layer.10/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1312_0, Constant_188_0, Dot_1336_0);
 // name=ElementWiseFused_1797
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_106_0, Dot_1336_0, Add_1338_0);
 // name=/encoder/layer.10/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1339(0, Add_1338_0, Reshape_1339_0);
 // name=/encoder/layer.10/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1339_0, Reshape_1340_0);
 // name=Reshape_1342
// eliminated: Reshape_float_float_cuda_lib_Reshape_1342(0, Reshape_1340_0, Reshape_1342_0);
 // name=Broadcast_1344
// eliminated: Broadcast_float_float_cuda_Broadcast_1344_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1342_0, Broadcast_1344_0);
 // name=/encoder/layer.10/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1312_0, Constant_187_0, Dot_1313_0);
 // name=ElementWiseFused_1795
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_105_0, Dot_1313_0, Add_1315_0);
 // name=/encoder/layer.10/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1316(0, Add_1315_0, Reshape_1316_0);
 // name=/encoder/layer.10/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_1316_0, Reshape_1317_0);
 // name=ElementWiseFused_1799
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1317_0, Multiply_1320_0);
 // name=Reshape_1330
// eliminated: Reshape_float_float_cuda_lib_Reshape_1330(0, Multiply_1320_0, Reshape_1330_0);
 // name=Broadcast_1332
// eliminated: Broadcast_float_float_cuda_Broadcast_1332_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1330_0, Broadcast_1332_0);
 // name=/encoder/layer.10/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1312_0, Constant_186_0, Dot_1321_0);
 // name=ElementWiseFused_1796
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_104_0, Dot_1321_0, Add_1323_0);
 // name=/encoder/layer.10/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1324(0, Add_1323_0, Reshape_1324_0);
 // name=/encoder/layer.10/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1324_0, Reshape_1325_0);
 // name=ElementWiseFused_1798
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1325_0, Multiply_1328_0);
 // name=Reshape_1329
// eliminated: Reshape_float_float_cuda_lib_Reshape_1329(0, Multiply_1328_0, Reshape_1329_0);
 // name=Broadcast_1331
// eliminated: Broadcast_float_float_cuda_Broadcast_1331_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1329_0, Broadcast_1331_0);
 // name=/encoder/layer.10/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_1331_0, Broadcast_1332_0, BatchMatMul_1333_0);
 // name=Reshape_1334
// eliminated: Reshape_float_float_cuda_lib_Reshape_1334(0, BatchMatMul_1333_0, Reshape_1334_0);
 // name=/encoder/layer.10/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_1334_0, Softmax_1335_0);
 // name=Reshape_1341
// eliminated: Reshape_float_float_cuda_lib_Reshape_1341(0, Softmax_1335_0, Reshape_1341_0);
 // name=Broadcast_1343
// eliminated: Broadcast_float_float_cuda_Broadcast_1343_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1341_0, Broadcast_1343_0);
 // name=/encoder/layer.10/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_1343_0, Broadcast_1344_0, BatchMatMul_1345_0);
 // name=Reshape_1346
// eliminated: Reshape_float_float_cuda_lib_Reshape_1346(0, BatchMatMul_1345_0, Reshape_1346_0);
 // name=/encoder/layer.10/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1346_0, Reshape_1347_0);
 // name=/encoder/layer.10/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1348(0, Reshape_1347_0, Reshape_1348_0);
 // name=/encoder/layer.10/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_1348_0, Constant_189_0, Dot_1349_0);
 // name=ElementWiseFused_1800
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_107_0, Dot_1349_0, Add_1286_0, Add_1352_0);
 // name=Sum_1353
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1352_0, Sum_1353_0);
 // name=ElementWiseFused_1801
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1354_0, Sum_1353_0, Divide_1356_0);
 // name=/encoder/layer.10/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1357(0, Divide_1356_0, Reshape_1357_0);
 // name=Reshape_1358
// eliminated: Reshape_float_float_cuda_lib_Reshape_1358(0, Reshape_1357_0, Reshape_1358_0);
 // name=ElementWiseFused_1802
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1358_0, Add_1352_0, Subtract_1360_0, Power_1362_0);
 // name=Sum_1363
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1362_0, Sum_1363_0);
 // name=ElementWiseFused_1803
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1364_0, Sum_1363_0, Sqrt_1371_0);
 // name=Reshape_1372
// eliminated: Reshape_float_float_cuda_lib_Reshape_1372(0, Sqrt_1371_0, Reshape_1372_0);
 // name=ElementWiseFused_1804
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_113_0, Constant_112_0, Reshape_1372_0, Subtract_1360_0, Add_1378_0);
 // name=/encoder/layer.10/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_1378_0, Constant_190_0, Dot_1379_0);
 // name=ElementWiseFused_1805
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_108_0, Dot_1379_0, Multiply_1389_0);
 // name=/encoder/layer.10/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_1389_0, Constant_191_0, Dot_1390_0);
 // name=ElementWiseFused_1806
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_109_0, Dot_1390_0, Add_1352_0, Add_1393_0);
 // name=Sum_1394
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1393_0, Sum_1394_0);
 // name=ElementWiseFused_1807
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1395_0, Sum_1394_0, Divide_1397_0);
 // name=/encoder/layer.11/layernorm_before/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1398(0, Divide_1397_0, Reshape_1398_0);
 // name=Reshape_1399
// eliminated: Reshape_float_float_cuda_lib_Reshape_1399(0, Reshape_1398_0, Reshape_1399_0);
 // name=ElementWiseFused_1808
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1399_0, Add_1393_0, Subtract_1401_0, Power_1403_0);
 // name=Sum_1404
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1403_0, Sum_1404_0);
 // name=ElementWiseFused_1809
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1405_0, Sum_1404_0, Sqrt_1412_0);
 // name=Reshape_1413
// eliminated: Reshape_float_float_cuda_lib_Reshape_1413(0, Sqrt_1412_0, Reshape_1413_0);
 // name=ElementWiseFused_1810
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_121_0, Constant_120_0, Reshape_1413_0, Subtract_1401_0, Add_1419_0);
 // name=/encoder/layer.11/attention/attention/value/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1419_0, Constant_194_0, Dot_1443_0);
 // name=ElementWiseFused_1813
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_116_0, Dot_1443_0, Add_1445_0);
 // name=/encoder/layer.11/attention/attention/Reshape_1_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1446(0, Add_1445_0, Reshape_1446_0);
 // name=/encoder/layer.11/attention/attention/Transpose_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1446_0, Reshape_1447_0);
 // name=Reshape_1449
// eliminated: Reshape_float_float_cuda_lib_Reshape_1449(0, Reshape_1447_0, Reshape_1449_0);
 // name=Broadcast_1451
// eliminated: Broadcast_float_float_cuda_Broadcast_1451_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1449_0, Broadcast_1451_0);
 // name=/encoder/layer.11/attention/attention/key/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1419_0, Constant_193_0, Dot_1420_0);
 // name=ElementWiseFused_1811
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_115_0, Dot_1420_0, Add_1422_0);
 // name=/encoder/layer.11/attention/attention/Reshape_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1423(0, Add_1422_0, Reshape_1423_0);
 // name=/encoder/layer.11/attention/attention/Transpose_2_output_0
Reshape_float_float_cuda_Reshape_247_Call(dim3(48, 13, 48), dim3(16, 16, 1), 0, 0, Reshape_1423_0, Reshape_1424_0);
 // name=ElementWiseFused_1815
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1424_0, Multiply_1427_0);
 // name=Reshape_1437
// eliminated: Reshape_float_float_cuda_lib_Reshape_1437(0, Multiply_1427_0, Reshape_1437_0);
 // name=Broadcast_1439
// eliminated: Broadcast_float_float_cuda_Broadcast_1439_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1437_0, Broadcast_1439_0);
 // name=/encoder/layer.11/attention/attention/query/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Add_1419_0, Constant_192_0, Dot_1428_0);
 // name=ElementWiseFused_1812
FusedKernel_float_float_float_cuda_Broadcast_Add_6_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_114_0, Dot_1428_0, Add_1430_0);
 // name=/encoder/layer.11/attention/attention/Reshape_2_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1431(0, Add_1430_0, Reshape_1431_0);
 // name=/encoder/layer.11/attention/attention/Transpose_1_output_0
Reshape_float_float_cuda_Reshape_270_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1431_0, Reshape_1432_0);
 // name=ElementWiseFused_1814
FusedKernel_float_float_float_cuda_Broadcast_Multiply_8_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Reshape_248_0, Reshape_1432_0, Multiply_1435_0);
 // name=Reshape_1436
// eliminated: Reshape_float_float_cuda_lib_Reshape_1436(0, Multiply_1435_0, Reshape_1436_0);
 // name=Broadcast_1438
// eliminated: Broadcast_float_float_cuda_Broadcast_1438_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1436_0, Broadcast_1438_0);
 // name=/encoder/layer.11/attention/attention/MatMul_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_263(cublas_handle_0, Broadcast_1438_0, Broadcast_1439_0, BatchMatMul_1440_0);
 // name=Reshape_1441
// eliminated: Reshape_float_float_cuda_lib_Reshape_1441(0, BatchMatMul_1440_0, Reshape_1441_0);
 // name=/encoder/layer.11/attention/attention/Softmax_output_0
Softmax_float_float_cuda_lib_Softmax_265(0, Reshape_1441_0, Softmax_1442_0);
 // name=Reshape_1448
// eliminated: Reshape_float_float_cuda_lib_Reshape_1448(0, Softmax_1442_0, Reshape_1448_0);
 // name=Broadcast_1450
// eliminated: Broadcast_float_float_cuda_Broadcast_1450_Call(dim3(349281, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1448_0, Broadcast_1450_0);
 // name=/encoder/layer.11/attention/attention/MatMul_1_output_0
BatchMatMul_float_float_float_cuda_lib_BatchMatMul_275(cublas_handle_0, Broadcast_1450_0, Broadcast_1451_0, BatchMatMul_1452_0);
 // name=Reshape_1453
// eliminated: Reshape_float_float_cuda_lib_Reshape_1453(0, BatchMatMul_1452_0, Reshape_1453_0);
 // name=/encoder/layer.11/attention/attention/Transpose_3_output_0
Reshape_float_float_cuda_Reshape_277_Call(dim3(113472, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_1453_0, Reshape_1454_0);
 // name=/encoder/layer.11/attention/attention/Reshape_3_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1455(0, Reshape_1454_0, Reshape_1455_0);
 // name=/encoder/layer.11/attention/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_266(cublas_handle_0, Reshape_1455_0, Constant_195_0, Dot_1456_0);
 // name=ElementWiseFused_1816
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_117_0, Dot_1456_0, Add_1393_0, Add_1459_0);
 // name=Sum_1460
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1459_0, Sum_1460_0);
 // name=ElementWiseFused_1817
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1461_0, Sum_1460_0, Divide_1463_0);
 // name=/encoder/layer.11/layernorm_after/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1464(0, Divide_1463_0, Reshape_1464_0);
 // name=Reshape_1465
// eliminated: Reshape_float_float_cuda_lib_Reshape_1465(0, Reshape_1464_0, Reshape_1465_0);
 // name=ElementWiseFused_1818
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1465_0, Add_1459_0, Subtract_1467_0, Power_1469_0);
 // name=Sum_1470
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1469_0, Sum_1470_0);
 // name=ElementWiseFused_1819
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1471_0, Sum_1470_0, Sqrt_1478_0);
 // name=Reshape_1479
// eliminated: Reshape_float_float_cuda_lib_Reshape_1479(0, Sqrt_1478_0, Reshape_1479_0);
 // name=ElementWiseFused_1820
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_123_0, Constant_122_0, Reshape_1479_0, Subtract_1467_0, Add_1485_0);
 // name=/encoder/layer.11/intermediate/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_309(cublas_handle_0, Add_1485_0, Constant_196_0, Dot_1486_0);
 // name=ElementWiseFused_1821
FusedKernel_float_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Broadcast_Add_Divide_Erf_Add_Multiply_Multiply_14_Call(dim3(56736, 1, 1), dim3(512, 1, 1), 0, 0, Constant_204_0, Constant_203_0, Constant_205_0, Constant_118_0, Dot_1486_0, Multiply_1496_0);
 // name=/encoder/layer.11/output/dense/MatMul_output_0
Dot_float_float_float_cuda_lib_Dot_320(cublas_handle_0, Multiply_1496_0, Constant_197_0, Dot_1497_0);
 // name=ElementWiseFused_1822
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Add_9_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_119_0, Dot_1497_0, Add_1459_0, Add_1500_0);
 // name=Sum_1501
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Add_1500_0, Sum_1501_0);
 // name=ElementWiseFused_1823
FusedKernel_float_float_float_cuda_Broadcast_Divide_0_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Constant_1502_0, Sum_1501_0, Divide_1504_0);
 // name=/layernorm/ReduceMean_output_0
// eliminated: Reshape_float_float_cuda_lib_Reshape_1505(0, Divide_1504_0, Reshape_1505_0);
 // name=Reshape_1506
// eliminated: Reshape_float_float_cuda_lib_Reshape_1506(0, Reshape_1505_0, Reshape_1506_0);
 // name=ElementWiseFused_1824
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Subtract_Power_1_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_206_0, Reshape_1506_0, Add_1500_0, Subtract_1508_0, Power_1510_0);
 // name=Sum_1511
Sum_float_float_cuda_Sum_217_Call(dim3(9456, 1, 1), dim3(512, 1, 1), 0, 0, Power_1510_0, Sum_1511_0);
 // name=ElementWiseFused_1825
FusedKernel_float_float_float_float_cuda_Broadcast_Broadcast_Divide_Reshape_Add_Sqrt_2_Call(dim3(24, 1, 1), dim3(394, 1, 1), 0, 0, Reshape_1516_0, Constant_1512_0, Sum_1511_0, Sqrt_1519_0);
 // name=Reshape_1520
// eliminated: Reshape_float_float_cuda_lib_Reshape_1520(0, Sqrt_1519_0, Reshape_1520_0);
 // name=ElementWiseFused_1826
FusedKernel_float_float_float_float_float_cuda_Broadcast_Broadcast_Broadcast_Divide_Multiply_Add_3_Call(dim3(14184, 1, 1), dim3(512, 1, 1), 0, 0, Constant_125_0, Constant_124_0, Reshape_1520_0, Subtract_1508_0, last_hidden_state);
 // name=Result_1527
Result_float_float_cuda_lib_Result_1527(last_hidden_state, Result_1527_0);
return 0;
}


extern "C" void cuda_free()
{
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

