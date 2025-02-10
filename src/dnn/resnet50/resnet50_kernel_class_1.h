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

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>
#include <vector>
#define MIN(a,b) ((a)>(b)?(b):(a))
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
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_272_0;
float* Constant_9_0;
float* Constant_8_0;
float* Constant_7_0;
float* Constant_6_0;
float* Constant_5_0;
float* Constant_19_0;
float* Constant_18_0;
float* Constant_17_0;
float* Constant_16_0;
float* Constant_15_0;
float* Constant_24_0;
float* Constant_23_0;
float* Constant_22_0;
float* Constant_21_0;
float* Constant_20_0;
float* Constant_29_0;
float* Constant_28_0;
float* Constant_27_0;
float* Constant_26_0;
float* Constant_25_0;
float* Constant_14_0;
float* Constant_13_0;
float* Constant_12_0;
float* Constant_11_0;
float* Constant_10_0;
float* Constant_34_0;
float* Constant_33_0;
float* Constant_32_0;
float* Constant_31_0;
float* Constant_30_0;
float* Constant_39_0;
float* Constant_38_0;
float* Constant_37_0;
float* Constant_36_0;
float* Constant_35_0;
float* Constant_44_0;
float* Constant_43_0;
float* Constant_42_0;
float* Constant_41_0;
float* Constant_40_0;
float* Constant_149_0;
float* Constant_148_0;
float* Constant_147_0;
float* Constant_146_0;
float* Constant_145_0;
float* Constant_154_0;
float* Constant_153_0;
float* Constant_152_0;
float* Constant_151_0;
float* Constant_150_0;
float* Constant_144_0;
float* Constant_143_0;
float* Constant_142_0;
float* Constant_141_0;
float* Constant_140_0;
float* Constant_164_0;
float* Constant_163_0;
float* Constant_162_0;
float* Constant_161_0;
float* Constant_160_0;
float* Constant_169_0;
float* Constant_168_0;
float* Constant_167_0;
float* Constant_166_0;
float* Constant_165_0;
float* Constant_174_0;
float* Constant_173_0;
float* Constant_172_0;
float* Constant_171_0;
float* Constant_170_0;
float* Constant_159_0;
float* Constant_158_0;
float* Constant_157_0;
float* Constant_156_0;
float* Constant_155_0;
float* Constant_179_0;
float* Constant_178_0;
float* Constant_177_0;
float* Constant_176_0;
float* Constant_175_0;
float* Constant_184_0;
float* Constant_183_0;
float* Constant_182_0;
float* Constant_181_0;
float* Constant_180_0;
float* Constant_189_0;
float* Constant_188_0;
float* Constant_187_0;
float* Constant_186_0;
float* Constant_185_0;
float* Constant_194_0;
float* Constant_193_0;
float* Constant_192_0;
float* Constant_191_0;
float* Constant_190_0;
float* Constant_199_0;
float* Constant_198_0;
float* Constant_197_0;
float* Constant_196_0;
float* Constant_195_0;
float* Constant_204_0;
float* Constant_203_0;
float* Constant_202_0;
float* Constant_201_0;
float* Constant_200_0;
float* Constant_209_0;
float* Constant_208_0;
float* Constant_207_0;
float* Constant_206_0;
float* Constant_205_0;
float* Constant_214_0;
float* Constant_213_0;
float* Constant_212_0;
float* Constant_211_0;
float* Constant_210_0;
float* Constant_219_0;
float* Constant_218_0;
float* Constant_217_0;
float* Constant_216_0;
float* Constant_215_0;
float* Constant_229_0;
float* Constant_228_0;
float* Constant_227_0;
float* Constant_226_0;
float* Constant_225_0;
float* Constant_234_0;
float* Constant_233_0;
float* Constant_232_0;
float* Constant_231_0;
float* Constant_230_0;
float* Constant_239_0;
float* Constant_238_0;
float* Constant_237_0;
float* Constant_236_0;
float* Constant_235_0;
float* Constant_224_0;
float* Constant_223_0;
float* Constant_222_0;
float* Constant_221_0;
float* Constant_220_0;
float* Constant_244_0;
float* Constant_243_0;
float* Constant_242_0;
float* Constant_241_0;
float* Constant_240_0;
float* Constant_249_0;
float* Constant_248_0;
float* Constant_247_0;
float* Constant_246_0;
float* Constant_245_0;
float* Constant_254_0;
float* Constant_253_0;
float* Constant_252_0;
float* Constant_251_0;
float* Constant_250_0;
float* Constant_259_0;
float* Constant_258_0;
float* Constant_257_0;
float* Constant_256_0;
float* Constant_255_0;
float* Constant_264_0;
float* Constant_263_0;
float* Constant_262_0;
float* Constant_261_0;
float* Constant_260_0;
float* Constant_269_0;
float* Constant_268_0;
float* Constant_267_0;
float* Constant_266_0;
float* Constant_265_0;
float* Constant_49_0;
float* Constant_48_0;
float* Constant_47_0;
float* Constant_46_0;
float* Constant_45_0;
float* Constant_54_0;
float* Constant_53_0;
float* Constant_52_0;
float* Constant_51_0;
float* Constant_50_0;
float* Constant_59_0;
float* Constant_58_0;
float* Constant_57_0;
float* Constant_56_0;
float* Constant_55_0;
float* Constant_64_0;
float* Constant_63_0;
float* Constant_62_0;
float* Constant_61_0;
float* Constant_60_0;
float* Constant_69_0;
float* Constant_68_0;
float* Constant_67_0;
float* Constant_66_0;
float* Constant_65_0;
float* Constant_74_0;
float* Constant_73_0;
float* Constant_72_0;
float* Constant_71_0;
float* Constant_70_0;
float* Constant_79_0;
float* Constant_78_0;
float* Constant_77_0;
float* Constant_76_0;
float* Constant_75_0;
float* Constant_84_0;
float* Constant_83_0;
float* Constant_82_0;
float* Constant_81_0;
float* Constant_80_0;
float* Constant_89_0;
float* Constant_88_0;
float* Constant_87_0;
float* Constant_86_0;
float* Constant_85_0;
float* Constant_99_0;
float* Constant_98_0;
float* Constant_97_0;
float* Constant_96_0;
float* Constant_95_0;
float* Constant_104_0;
float* Constant_103_0;
float* Constant_102_0;
float* Constant_101_0;
float* Constant_100_0;
float* Constant_109_0;
float* Constant_108_0;
float* Constant_107_0;
float* Constant_106_0;
float* Constant_105_0;
float* Constant_94_0;
float* Constant_93_0;
float* Constant_92_0;
float* Constant_91_0;
float* Constant_90_0;
float* Constant_114_0;
float* Constant_113_0;
float* Constant_112_0;
float* Constant_111_0;
float* Constant_110_0;
float* Constant_119_0;
float* Constant_118_0;
float* Constant_117_0;
float* Constant_116_0;
float* Constant_115_0;
float* Constant_124_0;
float* Constant_123_0;
float* Constant_122_0;
float* Constant_121_0;
float* Constant_120_0;
float* Constant_129_0;
float* Constant_128_0;
float* Constant_127_0;
float* Constant_126_0;
float* Constant_125_0;
float* Constant_134_0;
float* Constant_133_0;
float* Constant_132_0;
float* Constant_131_0;
float* Constant_130_0;
float* Constant_139_0;
float* Constant_138_0;
float* Constant_137_0;
float* Constant_136_0;
float* Constant_135_0;
float* Constant_500_0;
float* Constant_4_0;
float* Constant_3_0;
float* Broadcast_503_0;
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
}
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* Reshape_271_0;
float* Pad_273_0;
float* Reshape_274_0;
float* Convolution_275_0;
float* BatchNormInference_276_0;
float* Relu_277_0;
float* MaxPool_278_0;
float* Reshape_281_0;
float* Convolution_282_0;
float* BatchNormInference_284_0;
float* Relu_285_0;
float* Reshape_286_0;
float* Convolution_287_0;
float* BatchNormInference_288_0;
float* Relu_289_0;
float* Reshape_290_0;
float* Convolution_291_0;
float* BatchNormInference_292_0;
float* Reshape_279_0;
float* Convolution_280_0;
float* BatchNormInference_283_0;
float* Relu_294_0;
float* Reshape_295_0;
float* Convolution_296_0;
float* BatchNormInference_297_0;
float* Relu_298_0;
float* Reshape_299_0;
float* Convolution_300_0;
float* BatchNormInference_301_0;
float* Relu_302_0;
float* Reshape_303_0;
float* Convolution_304_0;
float* BatchNormInference_305_0;
float* Relu_307_0;
float* Reshape_308_0;
float* Convolution_309_0;
float* BatchNormInference_310_0;
float* Relu_311_0;
float* Reshape_312_0;
float* Convolution_313_0;
float* BatchNormInference_314_0;
float* Relu_315_0;
float* Reshape_316_0;
float* Convolution_317_0;
float* BatchNormInference_318_0;
float* Relu_320_0;
float* Reshape_323_0;
float* Convolution_324_0;
float* BatchNormInference_326_0;
float* Relu_327_0;
float* Reshape_328_0;
float* Convolution_329_0;
float* BatchNormInference_330_0;
float* Relu_331_0;
float* Reshape_332_0;
float* Convolution_333_0;
float* BatchNormInference_334_0;
float* Reshape_321_0;
float* Convolution_322_0;
float* BatchNormInference_325_0;
float* Relu_336_0;
float* Reshape_337_0;
float* Convolution_338_0;
float* BatchNormInference_339_0;
float* Relu_340_0;
float* Reshape_341_0;
float* Convolution_342_0;
float* BatchNormInference_343_0;
float* Relu_344_0;
float* Reshape_345_0;
float* Convolution_346_0;
float* BatchNormInference_347_0;
float* Relu_349_0;
float* Reshape_350_0;
float* Convolution_351_0;
float* BatchNormInference_352_0;
float* Relu_353_0;
float* Reshape_354_0;
float* Convolution_355_0;
float* BatchNormInference_356_0;
float* Relu_357_0;
float* Reshape_358_0;
float* Convolution_359_0;
float* BatchNormInference_360_0;
float* Relu_362_0;
float* Reshape_363_0;
float* Convolution_364_0;
float* BatchNormInference_365_0;
float* Relu_366_0;
float* Reshape_367_0;
float* Convolution_368_0;
float* BatchNormInference_369_0;
float* Relu_370_0;
float* Reshape_371_0;
float* Convolution_372_0;
float* BatchNormInference_373_0;
float* Relu_375_0;
float* Reshape_378_0;
float* Convolution_379_0;
float* BatchNormInference_381_0;
float* Relu_382_0;
float* Reshape_383_0;
float* Convolution_384_0;
float* BatchNormInference_385_0;
float* Relu_386_0;
float* Reshape_387_0;
float* Convolution_388_0;
float* BatchNormInference_389_0;
float* Reshape_376_0;
float* Convolution_377_0;
float* BatchNormInference_380_0;
float* Relu_391_0;
float* Reshape_392_0;
float* Convolution_393_0;
float* BatchNormInference_394_0;
float* Relu_395_0;
float* Reshape_396_0;
float* Convolution_397_0;
float* BatchNormInference_398_0;
float* Relu_399_0;
float* Reshape_400_0;
float* Convolution_401_0;
float* BatchNormInference_402_0;
float* Relu_404_0;
float* Reshape_405_0;
float* Convolution_406_0;
float* BatchNormInference_407_0;
float* Relu_408_0;
float* Reshape_409_0;
float* Convolution_410_0;
float* BatchNormInference_411_0;
float* Relu_412_0;
float* Reshape_413_0;
float* Convolution_414_0;
float* BatchNormInference_415_0;
float* Relu_417_0;
float* Reshape_418_0;
float* Convolution_419_0;
float* BatchNormInference_420_0;
float* Relu_421_0;
float* Reshape_422_0;
float* Convolution_423_0;
float* BatchNormInference_424_0;
float* Relu_425_0;
float* Reshape_426_0;
float* Convolution_427_0;
float* BatchNormInference_428_0;
float* Relu_430_0;
float* Reshape_431_0;
float* Convolution_432_0;
float* BatchNormInference_433_0;
float* Relu_434_0;
float* Reshape_435_0;
float* Convolution_436_0;
float* BatchNormInference_437_0;
float* Relu_438_0;
float* Reshape_439_0;
float* Convolution_440_0;
float* BatchNormInference_441_0;
float* Relu_443_0;
float* Reshape_444_0;
float* Convolution_445_0;
float* BatchNormInference_446_0;
float* Relu_447_0;
float* Reshape_448_0;
float* Convolution_449_0;
float* BatchNormInference_450_0;
float* Relu_451_0;
float* Reshape_452_0;
float* Convolution_453_0;
float* BatchNormInference_454_0;
float* Relu_456_0;
float* Reshape_459_0;
float* Convolution_460_0;
float* BatchNormInference_462_0;
float* Relu_463_0;
float* Reshape_464_0;
float* Convolution_465_0;
float* BatchNormInference_466_0;
float* Relu_467_0;
float* Reshape_468_0;
float* Convolution_469_0;
float* BatchNormInference_470_0;
float* Reshape_457_0;
float* Convolution_458_0;
float* BatchNormInference_461_0;
float* Relu_472_0;
float* Reshape_473_0;
float* Convolution_474_0;
float* BatchNormInference_475_0;
float* Relu_476_0;
float* Reshape_477_0;
float* Convolution_478_0;
float* BatchNormInference_479_0;
float* Relu_480_0;
float* Reshape_481_0;
float* Convolution_482_0;
float* BatchNormInference_483_0;
float* Relu_485_0;
float* Reshape_486_0;
float* Convolution_487_0;
float* BatchNormInference_488_0;
float* Relu_489_0;
float* Reshape_490_0;
float* Convolution_491_0;
float* BatchNormInference_492_0;
float* Relu_493_0;
float* Reshape_494_0;
float* Convolution_495_0;
float* BatchNormInference_496_0;
float* Relu_498_0;
float* Sum_499_0;
float* Divide_501_0;
float* Dot_502_0;
float* Add_504_0;

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

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
cublasHandle_t cublas_handle_0;
cudnnHandle_t cudnn_handle_0;
int num_SMs;
// Node name:	Reshape_387
// Description:	Reshape
// Input:
//	- name: Constant_239_0	type: float	shape: Shape{1, 1, 256, 1024}
// Output:
//	- name: Reshape_387_0	type: float	shape: Shape{1024, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_387(float* input0, float* output0)
{
    uint32_t input_strides0 = 1024;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 256;
    size_t nx = 1024;
    size_t ny = 256;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_387_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_387<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Relu_277
// Description:	Relu
// Input:
//	- name: BatchNormInference_276_0	type: float	shape: Shape{1, 64, 112, 112}
// Output:
//	- name: Relu_277_0	type: float	shape: Shape{1, 64, 112, 112}
extern "C" __launch_bounds__(512) __global__ void Relu_float_float_cuda_Relu_277(float* input0, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern void Relu_float_float_cuda_Relu_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Relu_float_float_cuda_Relu_277<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_332
// Description:	Reshape
// Input:
//	- name: Constant_174_0	type: float	shape: Shape{1, 1, 128, 512}
// Output:
//	- name: Reshape_332_0	type: float	shape: Shape{512, 128, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_332(float* input0, float* output0)
{
    uint32_t input_strides0 = 512;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 128;
    size_t nx = 512;
    size_t ny = 128;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_332_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_332<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_465
// Description:	Convolution
// Input:
//	- name: Relu_463_0	type: float	shape: Shape{1, 512, 7, 7}
//	- name: Reshape_464_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: Convolution_465_0	type: float	shape: Shape{1, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_465(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 512, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Constant_500
// Description:	Constant
// Input:
// Output:
//	- name: Constant_500_0	type: float	shape: Shape{1, 2048}
void Constant_float_cuda_Constant_500(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_500_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_500_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: Constant_40_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_118
// Description:	Constant
// Input:
// Output:
//	- name: Constant_118_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_118(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_118_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_118_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Sum_499
// Description:	Sum
// Input:
//	- name: Relu_498_0	type: float	shape: Shape{1, 2048, 7, 7}
// Output:
//	- name: Sum_499_0	type: float	shape: Shape{1, 2048}
extern "C" __launch_bounds__(32) __global__ void Sum_float_float_cuda_Sum_499(float* input0, float* output0)
{

    int width = 49;
    int block_size = 32;
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
extern void Sum_float_float_cuda_Sum_499_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Sum_float_float_cuda_Sum_499<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_139
// Description:	Constant
// Input:
// Output:
//	- name: Constant_139_0	type: float	shape: Shape{1, 1, 512, 2048}
void Constant_float_cuda_Constant_139(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_139_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_139_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: Constant_193_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_193(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_193_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_193_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: Constant_15_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_215
// Description:	Constant
// Input:
// Output:
//	- name: Constant_215_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_215(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_215_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_215_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_202
// Description:	Constant
// Input:
// Output:
//	- name: Constant_202_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_202(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_202_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_202_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_229
// Description:	Constant
// Input:
// Output:
//	- name: Constant_229_0	type: float	shape: Shape{1, 1, 512, 256}
void Constant_float_cuda_Constant_229(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_229_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_229_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[524288];
    bin_file.read(tmp_mem, 524288);
    cudaMemcpyAsync(output0, tmp_mem, 524288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: Constant_12_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_178
// Description:	Constant
// Input:
// Output:
//	- name: Constant_178_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_178(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_178_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_178_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: Constant_169_0	type: float	shape: Shape{3, 3, 128, 128}
void Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_169_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_258
// Description:	Constant
// Input:
// Output:
//	- name: Constant_258_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_258(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_258_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_258_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_249
// Description:	Constant
// Input:
// Output:
//	- name: Constant_249_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_249(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_249_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_249_0 failed.\n");
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
//	- name: Constant_218_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_218_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_155
// Description:	Constant
// Input:
// Output:
//	- name: Constant_155_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_155(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_155_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_155_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_161
// Description:	Constant
// Input:
// Output:
//	- name: Constant_161_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_161_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: Constant_72_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_72_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: Constant_7_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_16
// Description:	Constant
// Input:
// Output:
//	- name: Constant_16_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_16_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: Constant_37_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_22
// Description:	Constant
// Input:
// Output:
//	- name: Constant_22_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_22_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: Constant_84_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_171
// Description:	Constant
// Input:
// Output:
//	- name: Constant_171_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_171_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_122
// Description:	Constant
// Input:
// Output:
//	- name: Constant_122_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_122(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_122_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_122_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_142
// Description:	Constant
// Input:
// Output:
//	- name: Constant_142_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_142(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_142_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_142_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_17
// Description:	Constant
// Input:
// Output:
//	- name: Constant_17_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_203
// Description:	Constant
// Input:
// Output:
//	- name: Constant_203_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_203(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_203_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_203_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: Constant_51_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_173
// Description:	Constant
// Input:
// Output:
//	- name: Constant_173_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_173(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_173_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_173_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_130
// Description:	Constant
// Input:
// Output:
//	- name: Constant_130_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_130(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_130_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_130_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_255
// Description:	Constant
// Input:
// Output:
//	- name: Constant_255_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_255(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_255_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_255_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_128
// Description:	Constant
// Input:
// Output:
//	- name: Constant_128_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_128_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: Constant_32_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_156
// Description:	Constant
// Input:
// Output:
//	- name: Constant_156_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_156_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_89
// Description:	Constant
// Input:
// Output:
//	- name: Constant_89_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_89(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_89_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_89_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: Constant_42_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: Constant_43_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: Constant_61_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_236
// Description:	Constant
// Input:
// Output:
//	- name: Constant_236_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_236(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_236_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_236_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_149
// Description:	Constant
// Input:
// Output:
//	- name: Constant_149_0	type: float	shape: Shape{1, 1, 256, 64}
void Constant_float_cuda_Constant_149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_149_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_197
// Description:	Constant
// Input:
// Output:
//	- name: Constant_197_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_197_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: Constant_57_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_157
// Description:	Constant
// Input:
// Output:
//	- name: Constant_157_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_157(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_157_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_157_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_98
// Description:	Constant
// Input:
// Output:
//	- name: Constant_98_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_98_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_110
// Description:	Constant
// Input:
// Output:
//	- name: Constant_110_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_110_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_146
// Description:	Constant
// Input:
// Output:
//	- name: Constant_146_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_63
// Description:	Constant
// Input:
// Output:
//	- name: Constant_63_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_63(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_63_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_63_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_250
// Description:	Constant
// Input:
// Output:
//	- name: Constant_250_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_250(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_250_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_250_0 failed.\n");
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
//	- name: Constant_65_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_65_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: Constant_9_0	type: float	shape: Shape{7, 7, 3, 64}
void Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[37632];
    bin_file.read(tmp_mem, 37632);
    cudaMemcpyAsync(output0, tmp_mem, 37632, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_272
// Description:	Constant
// Input:
// Output:
//	- name: Constant_272_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_272(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_272_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_272_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_80
// Description:	Constant
// Input:
// Output:
//	- name: Constant_80_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_80(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_80_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_80_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: Constant_5_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_154
// Description:	Constant
// Input:
// Output:
//	- name: Constant_154_0	type: float	shape: Shape{3, 3, 64, 64}
void Constant_float_cuda_Constant_154(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_154_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_154_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: Constant_117_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_117(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_117_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_117_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: Constant_21_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_21_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_165
// Description:	Constant
// Input:
// Output:
//	- name: Constant_165_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_165(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_165_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_165_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_151
// Description:	Constant
// Input:
// Output:
//	- name: Constant_151_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_151(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_151_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_151_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: Constant_144_0	type: float	shape: Shape{1, 1, 64, 256}
void Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_144_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_56
// Description:	Constant
// Input:
// Output:
//	- name: Constant_56_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_56_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_141
// Description:	Constant
// Input:
// Output:
//	- name: Constant_141_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_141(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_141_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_141_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_74
// Description:	Constant
// Input:
// Output:
//	- name: Constant_74_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_74(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_74_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_74_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_150
// Description:	Constant
// Input:
// Output:
//	- name: Constant_150_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_246
// Description:	Constant
// Input:
// Output:
//	- name: Constant_246_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_246(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_246_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_246_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_129
// Description:	Constant
// Input:
// Output:
//	- name: Constant_129_0	type: float	shape: Shape{1, 1, 2048, 512}
void Constant_float_cuda_Constant_129(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_129_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_129_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_145
// Description:	Constant
// Input:
// Output:
//	- name: Constant_145_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_145(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_145_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_145_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_86
// Description:	Constant
// Input:
// Output:
//	- name: Constant_86_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: Constant_109_0	type: float	shape: Shape{1, 1, 512, 2048}
void Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_164_0	type: float	shape: Shape{1, 1, 256, 128}
void Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_252
// Description:	Constant
// Input:
// Output:
//	- name: Constant_252_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_252(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_252_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_252_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: Constant_29_0	type: float	shape: Shape{1, 1, 64, 256}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_205
// Description:	Constant
// Input:
// Output:
//	- name: Constant_205_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_205(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_205_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_205_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_50
// Description:	Constant
// Input:
// Output:
//	- name: Constant_50_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_50_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_81
// Description:	Constant
// Input:
// Output:
//	- name: Constant_81_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_81(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_81_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_81_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_101
// Description:	Constant
// Input:
// Output:
//	- name: Constant_101_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_101_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: Constant_167_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_167_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: Constant_59_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_59_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: Constant_88_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_88(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_88_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_108
// Description:	Constant
// Input:
// Output:
//	- name: Constant_108_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_108_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_70
// Description:	Constant
// Input:
// Output:
//	- name: Constant_70_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_70_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_66
// Description:	Constant
// Input:
// Output:
//	- name: Constant_66_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_66_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_120
// Description:	Constant
// Input:
// Output:
//	- name: Constant_120_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_266
// Description:	Constant
// Input:
// Output:
//	- name: Constant_266_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_266(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_266_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_266_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: Constant_92_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_92_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_114
// Description:	Constant
// Input:
// Output:
//	- name: Constant_114_0	type: float	shape: Shape{1, 1, 2048, 512}
void Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_114_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3_0	type: float	shape: Shape{1001}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_221
// Description:	Constant
// Input:
// Output:
//	- name: Constant_221_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_221(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_221_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_221_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: Constant_8_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_183
// Description:	Constant
// Input:
// Output:
//	- name: Constant_183_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_183(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_183_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_183_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_189
// Description:	Constant
// Input:
// Output:
//	- name: Constant_189_0	type: float	shape: Shape{1, 1, 128, 512}
void Constant_float_cuda_Constant_189(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_189_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_189_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_134
// Description:	Constant
// Input:
// Output:
//	- name: Constant_134_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_134(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_134_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_134_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_219
// Description:	Constant
// Input:
// Output:
//	- name: Constant_219_0	type: float	shape: Shape{1, 1, 128, 512}
void Constant_float_cuda_Constant_219(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_219_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_219_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_226
// Description:	Constant
// Input:
// Output:
//	- name: Constant_226_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_226(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_226_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_226_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: Constant_33_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_33_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_180
// Description:	Constant
// Input:
// Output:
//	- name: Constant_180_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_180(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_180_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_180_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_116
// Description:	Constant
// Input:
// Output:
//	- name: Constant_116_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_115
// Description:	Constant
// Input:
// Output:
//	- name: Constant_115_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_115(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_115_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_115_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_119
// Description:	Constant
// Input:
// Output:
//	- name: Constant_119_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_119_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: Constant_52_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_138
// Description:	Constant
// Input:
// Output:
//	- name: Constant_138_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_138(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_138_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_138_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_90
// Description:	Constant
// Input:
// Output:
//	- name: Constant_90_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_265
// Description:	Constant
// Input:
// Output:
//	- name: Constant_265_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_265(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_265_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_265_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_140
// Description:	Constant
// Input:
// Output:
//	- name: Constant_140_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_140_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_131
// Description:	Constant
// Input:
// Output:
//	- name: Constant_131_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_131(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_131_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_131_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_136
// Description:	Constant
// Input:
// Output:
//	- name: Constant_136_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_136(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_136_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_136_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: Constant_87_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_102
// Description:	Constant
// Input:
// Output:
//	- name: Constant_102_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_102(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_102_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_102_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_166
// Description:	Constant
// Input:
// Output:
//	- name: Constant_166_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_166(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_166_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_166_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_112_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_103
// Description:	Constant
// Input:
// Output:
//	- name: Constant_103_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_103(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_103_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_103_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_207
// Description:	Constant
// Input:
// Output:
//	- name: Constant_207_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_207(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_207_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_207_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: Constant_78_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_78_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: Constant_23_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_267
// Description:	Constant
// Input:
// Output:
//	- name: Constant_267_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_267(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_267_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_267_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_133
// Description:	Constant
// Input:
// Output:
//	- name: Constant_133_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_133(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_133_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_133_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_111
// Description:	Constant
// Input:
// Output:
//	- name: Constant_111_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_111_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_121
// Description:	Constant
// Input:
// Output:
//	- name: Constant_121_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_121(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_121_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_121_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_106
// Description:	Constant
// Input:
// Output:
//	- name: Constant_106_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_106(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_106_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_106_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_82
// Description:	Constant
// Input:
// Output:
//	- name: Constant_82_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_82(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_82_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_82_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_125
// Description:	Constant
// Input:
// Output:
//	- name: Constant_125_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_125_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: Constant_96_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_96_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_107
// Description:	Constant
// Input:
// Output:
//	- name: Constant_107_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_107_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_137
// Description:	Constant
// Input:
// Output:
//	- name: Constant_137_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_126_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: Constant_39_0	type: float	shape: Shape{3, 3, 64, 64}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_135
// Description:	Constant
// Input:
// Output:
//	- name: Constant_135_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_135(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_135_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_135_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_124
// Description:	Constant
// Input:
// Output:
//	- name: Constant_124_0	type: float	shape: Shape{1, 1, 512, 2048}
void Constant_float_cuda_Constant_124(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_124_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_124_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_71
// Description:	Constant
// Input:
// Output:
//	- name: Constant_71_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_73
// Description:	Constant
// Input:
// Output:
//	- name: Constant_73_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_73_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_85
// Description:	Constant
// Input:
// Output:
//	- name: Constant_85_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_69
// Description:	Constant
// Input:
// Output:
//	- name: Constant_69_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_69_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_94
// Description:	Constant
// Input:
// Output:
//	- name: Constant_94_0	type: float	shape: Shape{1, 1, 1024, 2048}
void Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8388608];
    bin_file.read(tmp_mem, 8388608);
    cudaMemcpyAsync(output0, tmp_mem, 8388608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_60
// Description:	Constant
// Input:
// Output:
//	- name: Constant_60_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_60_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_62
// Description:	Constant
// Input:
// Output:
//	- name: Constant_62_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_62_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_170_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: Constant_55_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_55_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_75
// Description:	Constant
// Input:
// Output:
//	- name: Constant_75_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_75(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_75_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_75_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_95
// Description:	Constant
// Input:
// Output:
//	- name: Constant_95_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_95_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: Constant_26_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_26_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_77
// Description:	Constant
// Input:
// Output:
//	- name: Constant_77_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_77(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_77_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_77_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_123
// Description:	Constant
// Input:
// Output:
//	- name: Constant_123_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_123(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_123_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_123_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_64
// Description:	Constant
// Input:
// Output:
//	- name: Constant_64_0	type: float	shape: Shape{1, 1, 1024, 256}
void Constant_float_cuda_Constant_64(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_64_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_64_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_222
// Description:	Constant
// Input:
// Output:
//	- name: Constant_222_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_222(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_222_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_222_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_41
// Description:	Constant
// Input:
// Output:
//	- name: Constant_41_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_41(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_41_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_41_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: Constant_4_0	type: float	shape: Shape{2048, 1001}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8200192];
    bin_file.read(tmp_mem, 8200192);
    cudaMemcpyAsync(output0, tmp_mem, 8200192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_162
// Description:	Constant
// Input:
// Output:
//	- name: Constant_162_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_162_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_162_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: Constant_58_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: Constant_24_0	type: float	shape: Shape{3, 3, 64, 64}
void Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_152
// Description:	Constant
// Input:
// Output:
//	- name: Constant_152_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_152(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_152_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_152_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: Constant_53_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: Constant_54_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: Constant_47_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_49
// Description:	Constant
// Input:
// Output:
//	- name: Constant_49_0	type: float	shape: Shape{1, 1, 1024, 256}
void Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_49_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_163
// Description:	Constant
// Input:
// Output:
//	- name: Constant_163_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_163(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_163_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_163_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_97
// Description:	Constant
// Input:
// Output:
//	- name: Constant_97_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_97_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_268
// Description:	Constant
// Input:
// Output:
//	- name: Constant_268_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_268(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_268_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_268_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_256
// Description:	Constant
// Input:
// Output:
//	- name: Constant_256_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_256(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_256_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_256_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_172
// Description:	Constant
// Input:
// Output:
//	- name: Constant_172_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_172(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_172_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_172_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_83
// Description:	Constant
// Input:
// Output:
//	- name: Constant_83_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_83(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_83_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_83_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_179
// Description:	Constant
// Input:
// Output:
//	- name: Constant_179_0	type: float	shape: Shape{1, 1, 512, 128}
void Constant_float_cuda_Constant_179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_179_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_243
// Description:	Constant
// Input:
// Output:
//	- name: Constant_243_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_243(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_243_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_243_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_153
// Description:	Constant
// Input:
// Output:
//	- name: Constant_153_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_153(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_153_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_153_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: Constant_19_0	type: float	shape: Shape{1, 1, 64, 64}
void Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_174
// Description:	Constant
// Input:
// Output:
//	- name: Constant_174_0	type: float	shape: Shape{1, 1, 128, 512}
void Constant_float_cuda_Constant_174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_174_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_227
// Description:	Constant
// Input:
// Output:
//	- name: Constant_227_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_227(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_227_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_227_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: Constant_100_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_100_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: Constant_67_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_67_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_264
// Description:	Constant
// Input:
// Output:
//	- name: Constant_264_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_264(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_264_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_264_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_143
// Description:	Constant
// Input:
// Output:
//	- name: Constant_143_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_143_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_196
// Description:	Constant
// Input:
// Output:
//	- name: Constant_196_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_196(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_196_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_196_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_34
// Description:	Constant
// Input:
// Output:
//	- name: Constant_34_0	type: float	shape: Shape{1, 1, 256, 64}
void Constant_float_cuda_Constant_34(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_34_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_132
// Description:	Constant
// Input:
// Output:
//	- name: Constant_132_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_132(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_132_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_132_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_148
// Description:	Constant
// Input:
// Output:
//	- name: Constant_148_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_148_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_20
// Description:	Constant
// Input:
// Output:
//	- name: Constant_20_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_20_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: Constant_38_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_99
// Description:	Constant
// Input:
// Output:
//	- name: Constant_99_0	type: float	shape: Shape{1, 1, 1024, 512}
void Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_99_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_185
// Description:	Constant
// Input:
// Output:
//	- name: Constant_185_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_185_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: Constant_11_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: Constant_36_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_36_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_10
// Description:	Constant
// Input:
// Output:
//	- name: Constant_10_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_234
// Description:	Constant
// Input:
// Output:
//	- name: Constant_234_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_234(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_234_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_234_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_13
// Description:	Constant
// Input:
// Output:
//	- name: Constant_13_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_13_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_168
// Description:	Constant
// Input:
// Output:
//	- name: Constant_168_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_168(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_168_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_168_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: Constant_31_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: Constant_25_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_190
// Description:	Constant
// Input:
// Output:
//	- name: Constant_190_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_190_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: Constant_35_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_177
// Description:	Constant
// Input:
// Output:
//	- name: Constant_177_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_177_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: Constant_68_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_68_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_176
// Description:	Constant
// Input:
// Output:
//	- name: Constant_176_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_176_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: Constant_44_0	type: float	shape: Shape{1, 1, 64, 256}
void Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_259
// Description:	Constant
// Input:
// Output:
//	- name: Constant_259_0	type: float	shape: Shape{1, 1, 1024, 256}
void Constant_float_cuda_Constant_259(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_259_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_259_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_175
// Description:	Constant
// Input:
// Output:
//	- name: Constant_175_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_175(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_175_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_175_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_231
// Description:	Constant
// Input:
// Output:
//	- name: Constant_231_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_231(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_231_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_231_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_184
// Description:	Constant
// Input:
// Output:
//	- name: Constant_184_0	type: float	shape: Shape{3, 3, 128, 128}
void Constant_float_cuda_Constant_184(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_184_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_184_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_182_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_182_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_181
// Description:	Constant
// Input:
// Output:
//	- name: Constant_181_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_181_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_204
// Description:	Constant
// Input:
// Output:
//	- name: Constant_204_0	type: float	shape: Shape{1, 1, 128, 512}
void Constant_float_cuda_Constant_204(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_204_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_204_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_188
// Description:	Constant
// Input:
// Output:
//	- name: Constant_188_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_188_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: Constant_93_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_186
// Description:	Constant
// Input:
// Output:
//	- name: Constant_186_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_186(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_186_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_186_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_263
// Description:	Constant
// Input:
// Output:
//	- name: Constant_263_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_263(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_263_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_263_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_160
// Description:	Constant
// Input:
// Output:
//	- name: Constant_160_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_160(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_160_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_160_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_194
// Description:	Constant
// Input:
// Output:
//	- name: Constant_194_0	type: float	shape: Shape{1, 1, 512, 128}
void Constant_float_cuda_Constant_194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_194_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_220
// Description:	Constant
// Input:
// Output:
//	- name: Constant_220_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_220(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_220_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_220_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{1, 1, 64, 256}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_192
// Description:	Constant
// Input:
// Output:
//	- name: Constant_192_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_192(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_192_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_192_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_191
// Description:	Constant
// Input:
// Output:
//	- name: Constant_191_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_257
// Description:	Constant
// Input:
// Output:
//	- name: Constant_257_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_257(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_257_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_257_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_199
// Description:	Constant
// Input:
// Output:
//	- name: Constant_199_0	type: float	shape: Shape{3, 3, 128, 128}
void Constant_float_cuda_Constant_199(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_199_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_199_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: Constant_45_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_198
// Description:	Constant
// Input:
// Output:
//	- name: Constant_198_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_198(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_198_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_198_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_201
// Description:	Constant
// Input:
// Output:
//	- name: Constant_201_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_201(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_201_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_201_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_200
// Description:	Constant
// Input:
// Output:
//	- name: Constant_200_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_200(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_200_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_200_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_159
// Description:	Constant
// Input:
// Output:
//	- name: Constant_159_0	type: float	shape: Shape{1, 1, 256, 512}
void Constant_float_cuda_Constant_159(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_159_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_159_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[524288];
    bin_file.read(tmp_mem, 524288);
    cudaMemcpyAsync(output0, tmp_mem, 524288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_209
// Description:	Constant
// Input:
// Output:
//	- name: Constant_209_0	type: float	shape: Shape{1, 1, 512, 128}
void Constant_float_cuda_Constant_209(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_209_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_209_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_244
// Description:	Constant
// Input:
// Output:
//	- name: Constant_244_0	type: float	shape: Shape{1, 1, 1024, 256}
void Constant_float_cuda_Constant_244(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_244_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_244_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_104_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_208
// Description:	Constant
// Input:
// Output:
//	- name: Constant_208_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_208(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_208_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_208_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_206
// Description:	Constant
// Input:
// Output:
//	- name: Constant_206_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_206(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_206_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_206_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_187
// Description:	Constant
// Input:
// Output:
//	- name: Constant_187_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_187(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_187_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_187_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_261
// Description:	Constant
// Input:
// Output:
//	- name: Constant_261_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_261(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_261_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_261_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_232
// Description:	Constant
// Input:
// Output:
//	- name: Constant_232_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_232(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_232_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_232_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: Constant_48_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_48_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_224
// Description:	Constant
// Input:
// Output:
//	- name: Constant_224_0	type: float	shape: Shape{1, 1, 512, 1024}
void Constant_float_cuda_Constant_224(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_224_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_224_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_230
// Description:	Constant
// Input:
// Output:
//	- name: Constant_230_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_230(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_230_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_230_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_214
// Description:	Constant
// Input:
// Output:
//	- name: Constant_214_0	type: float	shape: Shape{3, 3, 128, 128}
void Constant_float_cuda_Constant_214(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_214_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_214_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: Constant_91_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: Constant_76_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_213
// Description:	Constant
// Input:
// Output:
//	- name: Constant_213_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_213(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_213_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_213_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_147
// Description:	Constant
// Input:
// Output:
//	- name: Constant_147_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_147(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_147_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_147_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_158
// Description:	Constant
// Input:
// Output:
//	- name: Constant_158_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_195
// Description:	Constant
// Input:
// Output:
//	- name: Constant_195_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_195(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_195_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_195_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_211
// Description:	Constant
// Input:
// Output:
//	- name: Constant_211_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_211(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_211_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_211_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_210
// Description:	Constant
// Input:
// Output:
//	- name: Constant_210_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_210(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_210_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_210_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_238
// Description:	Constant
// Input:
// Output:
//	- name: Constant_238_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_238(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_238_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_238_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_217
// Description:	Constant
// Input:
// Output:
//	- name: Constant_217_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_217(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_217_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_217_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_269
// Description:	Constant
// Input:
// Output:
//	- name: Constant_269_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_269(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_269_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_269_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_216
// Description:	Constant
// Input:
// Output:
//	- name: Constant_216_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_216(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_216_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_216_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_105
// Description:	Constant
// Input:
// Output:
//	- name: Constant_105_0	type: float	shape: Shape{2048}
void Constant_float_cuda_Constant_105(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_105_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_105_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_228
// Description:	Constant
// Input:
// Output:
//	- name: Constant_228_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_228(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_228_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_228_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_225
// Description:	Constant
// Input:
// Output:
//	- name: Constant_225_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_225(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_225_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_225_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_113
// Description:	Constant
// Input:
// Output:
//	- name: Constant_113_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_113(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_113_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_113_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_240
// Description:	Constant
// Input:
// Output:
//	- name: Constant_240_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_240(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_240_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_240_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_233
// Description:	Constant
// Input:
// Output:
//	- name: Constant_233_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_233(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_233_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_233_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_239
// Description:	Constant
// Input:
// Output:
//	- name: Constant_239_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_239(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_239_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_239_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_237
// Description:	Constant
// Input:
// Output:
//	- name: Constant_237_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_237(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_237_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_237_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_235
// Description:	Constant
// Input:
// Output:
//	- name: Constant_235_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_235(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_235_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_235_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_223
// Description:	Constant
// Input:
// Output:
//	- name: Constant_223_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_223(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_223_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_223_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_242
// Description:	Constant
// Input:
// Output:
//	- name: Constant_242_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_242(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_242_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_242_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_241
// Description:	Constant
// Input:
// Output:
//	- name: Constant_241_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_241(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_241_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_241_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_245
// Description:	Constant
// Input:
// Output:
//	- name: Constant_245_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_245(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_245_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_245_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_248
// Description:	Constant
// Input:
// Output:
//	- name: Constant_248_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_248(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_248_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_248_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_247
// Description:	Constant
// Input:
// Output:
//	- name: Constant_247_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_247(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_247_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_247_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_254
// Description:	Constant
// Input:
// Output:
//	- name: Constant_254_0	type: float	shape: Shape{1, 1, 256, 1024}
void Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_254_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_253
// Description:	Constant
// Input:
// Output:
//	- name: Constant_253_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_253(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_253_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_253_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_127
// Description:	Constant
// Input:
// Output:
//	- name: Constant_127_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_127(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_127_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_127_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_79
// Description:	Constant
// Input:
// Output:
//	- name: Constant_79_0	type: float	shape: Shape{1, 1, 1024, 256}
void Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_79_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_251
// Description:	Constant
// Input:
// Output:
//	- name: Constant_251_0	type: float	shape: Shape{1024}
void Constant_float_cuda_Constant_251(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_251_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_251_0 failed.\n");
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
//	- name: Constant_262_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_262(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_262_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_262_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_260
// Description:	Constant
// Input:
// Output:
//	- name: Constant_260_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_260(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_260_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_260_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Reshape_468
// Description:	Reshape
// Input:
//	- name: Constant_109_0	type: float	shape: Shape{1, 1, 512, 2048}
// Output:
//	- name: Reshape_468_0	type: float	shape: Shape{2048, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_468(float* input0, float* output0)
{
    uint32_t input_strides0 = 2048;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 512;
    size_t nx = 2048;
    size_t ny = 512;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_468_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_468<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_276
// Description:	BatchNormInference
// Input:
//	- name: Constant_6_0	type: float	shape: Shape{64}
//	- name: Constant_5_0	type: float	shape: Shape{64}
//	- name: Convolution_275_0	type: float	shape: Shape{1, 64, 112, 112}
//	- name: Constant_7_0	type: float	shape: Shape{64}
//	- name: Constant_8_0	type: float	shape: Shape{64}
// Output:
//	- name: BatchNormInference_276_0	type: float	shape: Shape{1, 64, 112, 112}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 112 * 112;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 112 * 112; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Divide_501
// Description:	Divide
// Input:
//	- name: Sum_499_0	type: float	shape: Shape{1, 2048}
//	- name: Constant_500_0	type: float	shape: Shape{1, 2048}
// Output:
//	- name: Divide_501_0	type: float	shape: Shape{1, 2048}
extern "C" __launch_bounds__(512) __global__ void Divide_float_float_float_cuda_Divide_501(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = fdividef(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void Divide_float_float_float_cuda_Divide_501_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Divide_float_float_float_cuda_Divide_501<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_291
// Description:	Convolution
// Input:
//	- name: Relu_289_0	type: float	shape: Shape{1, 64, 56, 56}
//	- name: Reshape_290_0	type: float	shape: Shape{256, 64, 1, 1}
// Output:
//	- name: Convolution_291_0	type: float	shape: Shape{1, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_291(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_324
// Description:	Convolution
// Input:
//	- name: Relu_320_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: Reshape_323_0	type: float	shape: Shape{128, 256, 1, 1}
// Output:
//	- name: Convolution_324_0	type: float	shape: Shape{1, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_324(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_296
// Description:	Convolution
// Input:
//	- name: Relu_294_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: Reshape_295_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_296_0	type: float	shape: Shape{1, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_296(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_284
// Description:	BatchNormInference
// Input:
//	- name: Constant_16_0	type: float	shape: Shape{64}
//	- name: Constant_15_0	type: float	shape: Shape{64}
//	- name: Convolution_282_0	type: float	shape: Shape{1, 64, 56, 56}
//	- name: Constant_17_0	type: float	shape: Shape{64}
//	- name: Constant_18_0	type: float	shape: Shape{64}
// Output:
//	- name: BatchNormInference_284_0	type: float	shape: Shape{1, 64, 56, 56}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 56 * 56;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 56 * 56; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_274
// Description:	Reshape
// Input:
//	- name: Constant_9_0	type: float	shape: Shape{7, 7, 3, 64}
// Output:
//	- name: Reshape_274_0	type: float	shape: Shape{64, 3, 7, 7}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_274(float* input0, float* output0)
{
    uint32_t input_strides0 = 192;
    uint32_t input_strides1 = 64;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 49;
    uint32_t trans_strides2 = 147;
    size_t nx = 64;
    size_t ny = 3;
    size_t nz = 49;
    __shared__ float tile[16][1][17];
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
    otid2 = tid0;
    otid0 = tid2;
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
extern void Reshape_float_float_cuda_Reshape_274_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_274<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_334
// Description:	BatchNormInference
// Input:
//	- name: Constant_171_0	type: float	shape: Shape{512}
//	- name: Constant_170_0	type: float	shape: Shape{512}
//	- name: Convolution_333_0	type: float	shape: Shape{1, 512, 28, 28}
//	- name: Constant_172_0	type: float	shape: Shape{512}
//	- name: Constant_173_0	type: float	shape: Shape{512}
// Output:
//	- name: BatchNormInference_334_0	type: float	shape: Shape{1, 512, 28, 28}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 28 * 28;
    const int c_id = blockIdx.x % 512;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 28 * 28; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Pad_273
// Description:	Pad
// Input:
//	- name: Reshape_271_0	type: float	shape: Shape{1, 3, 224, 224}
//	- name: Constant_272_0	type: float	shape: Shape{}
// Output:
//	- name: Pad_273_0	type: float	shape: Shape{1, 3, 230, 230}
extern "C" __launch_bounds__(64) __global__ void Pad_float_float_float_cuda_Pad_273(float* input0, float* input1, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* in = input0;
    float* pad = input1;
    float* out = output0;
    if (tid < 158700)
    {
        size_t input_shape0 = 1;
        size_t input_shape1 = 3;
        size_t input_shape2 = 224;
        size_t input_shape3 = 224;
        uint32_t input_strides0 = 150528;
        uint32_t input_strides1 = 50176;
        uint32_t input_strides2 = 224;
        uint32_t input_strides3 = 1;
        uint32_t output_strides0 = 158700;
        uint32_t output_strides1 = 52900;
        uint32_t output_strides2 = 230;
        uint32_t output_strides3 = 1;
        uint32_t padding_below0 = 0;
        uint32_t padding_below1 = 0;
        uint32_t padding_below2 = 3;
        uint32_t padding_below3 = 3;
        uint32_t padding_interior0 = 0;
        uint32_t padding_interior1 = 0;
        uint32_t padding_interior2 = 0;
        uint32_t padding_interior3 = 0;
        bool in_bounds = true;
        uint32_t output_pixel = tid;
        uint32_t input_pixel = 0;
        int32_t input, input_dil;
        input_dil = output_pixel / output_strides0 - padding_below0;
        input = input_dil / (padding_interior0 + 1);
        input_dil %= (padding_interior0 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape0) && (input_dil == 0);
        input_pixel += input * input_strides0;
        output_pixel %= output_strides0;
        input_dil = output_pixel / output_strides1 - padding_below1;
        input = input_dil / (padding_interior1 + 1);
        input_dil %= (padding_interior1 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape1) && (input_dil == 0);
        input_pixel += input * input_strides1;
        output_pixel %= output_strides1;
        input_dil = output_pixel / output_strides2 - padding_below2;
        input = input_dil / (padding_interior2 + 1);
        input_dil %= (padding_interior2 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape2) && (input_dil == 0);
        input_pixel += input * input_strides2;
        output_pixel %= output_strides2;
        input_dil = output_pixel / output_strides3 - padding_below3;
        input = input_dil / (padding_interior3 + 1);
        input_dil %= (padding_interior3 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape3) && (input_dil == 0);
        input_pixel += input * input_strides3;
        out[tid] = (in_bounds) ? in[input_pixel] : *pad;
    }

}
extern void Pad_float_float_float_cuda_Pad_273_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Pad_float_float_float_cuda_Pad_273<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_275
// Description:	Convolution
// Input:
//	- name: Pad_273_0	type: float	shape: Shape{1, 3, 230, 230}
//	- name: Reshape_274_0	type: float	shape: Shape{64, 3, 7, 7}
// Output:
//	- name: Convolution_275_0	type: float	shape: Shape{1, 64, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_275(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 230, 230));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 7, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_378
// Description:	Reshape
// Input:
//	- name: Constant_229_0	type: float	shape: Shape{1, 1, 512, 256}
// Output:
//	- name: Reshape_378_0	type: float	shape: Shape{256, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_378(float* input0, float* output0)
{
    uint32_t input_strides0 = 256;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 512;
    size_t nx = 256;
    size_t ny = 512;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_378_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_378<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_328
// Description:	Reshape
// Input:
//	- name: Constant_169_0	type: float	shape: Shape{3, 3, 128, 128}
// Output:
//	- name: Reshape_328_0	type: float	shape: Shape{128, 128, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_328(float* input0, float* output0)
{
    uint32_t input_strides0 = 16384;
    uint32_t input_strides1 = 128;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 1152;
    size_t nx = 128;
    size_t ny = 128;
    size_t nz = 9;
    __shared__ float tile[16][1][17];
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
    otid2 = tid0;
    otid0 = tid2;
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
extern void Reshape_float_float_cuda_Reshape_328_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_328<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_392
// Description:	Reshape
// Input:
//	- name: Constant_244_0	type: float	shape: Shape{1, 1, 1024, 256}
// Output:
//	- name: Reshape_392_0	type: float	shape: Shape{256, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_392(float* input0, float* output0)
{
    uint32_t input_strides0 = 256;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1024;
    size_t nx = 256;
    size_t ny = 1024;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_392<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_271
// Description:	Reshape
// Input:
//	- name: Parameter_270_0	type: float	shape: Shape{1, 224, 224, 3}
// Output:
//	- name: Reshape_271_0	type: float	shape: Shape{1, 3, 224, 224}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_271(float* input0, float* output0)
{
    uint32_t input_strides0 = 3;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 50176;
    size_t nx = 3;
    size_t ny = 50176;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_271_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_271<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_287
// Description:	Convolution
// Input:
//	- name: Relu_285_0	type: float	shape: Shape{1, 64, 56, 56}
//	- name: Reshape_286_0	type: float	shape: Shape{64, 64, 3, 3}
// Output:
//	- name: Convolution_287_0	type: float	shape: Shape{1, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_287(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Constant_212
// Description:	Constant
// Input:
// Output:
//	- name: Constant_212_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_212(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/resnet50/Constant/Constant_212_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_212_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Reshape_286
// Description:	Reshape
// Input:
//	- name: Constant_24_0	type: float	shape: Shape{3, 3, 64, 64}
// Output:
//	- name: Reshape_286_0	type: float	shape: Shape{64, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_286(float* input0, float* output0)
{
    uint32_t input_strides0 = 4096;
    uint32_t input_strides1 = 64;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 576;
    size_t nx = 64;
    size_t ny = 64;
    size_t nz = 9;
    __shared__ float tile[16][1][17];
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
    otid2 = tid0;
    otid0 = tid2;
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
extern void Reshape_float_float_cuda_Reshape_286_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_286<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: BatchNormInference_283_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: BatchNormInference_292_0	type: float	shape: Shape{1, 256, 56, 56}
// Output:
//	- name: Relu_294_0	type: float	shape: Shape{1, 256, 56, 56}
// Fused functions:
// Add_float_float_float_cuda_Add_293<<<dim3(1568, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_283_0, BatchNormInference_292_0, Add_293_0);
// Relu_float_float_cuda_Relu_294<<<dim3(1568, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_293_0, Relu_294_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Add_Relu_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Add_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Add_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_473
// Description:	Reshape
// Input:
//	- name: Constant_114_0	type: float	shape: Shape{1, 1, 2048, 512}
// Output:
//	- name: Reshape_473_0	type: float	shape: Shape{512, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_473(float* input0, float* output0)
{
    uint32_t input_strides0 = 512;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 2048;
    size_t nx = 512;
    size_t ny = 2048;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_473_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_473<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_329
// Description:	Convolution
// Input:
//	- name: Relu_327_0	type: float	shape: Shape{1, 128, 28, 28}
//	- name: Reshape_328_0	type: float	shape: Shape{128, 128, 3, 3}
// Output:
//	- name: Convolution_329_0	type: float	shape: Shape{1, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_329(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_389
// Description:	BatchNormInference
// Input:
//	- name: Constant_236_0	type: float	shape: Shape{1024}
//	- name: Constant_235_0	type: float	shape: Shape{1024}
//	- name: Convolution_388_0	type: float	shape: Shape{1, 1024, 14, 14}
//	- name: Constant_237_0	type: float	shape: Shape{1024}
//	- name: Constant_238_0	type: float	shape: Shape{1024}
// Output:
//	- name: BatchNormInference_389_0	type: float	shape: Shape{1, 1024, 14, 14}
extern "C" __launch_bounds__(196) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 14 * 14;
    const int c_id = blockIdx.x % 1024;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 14 * 14; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

extern "C" void cuda_init()
{
// total memory:114302720

CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,102457088));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 102457088));
Constant_272_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37696);
Constant_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37952);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38208);
Constant_5_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38464);
Constant_19_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38720);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+55104);
Constant_17_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+55360);
Constant_16_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+55616);
Constant_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+55872);
Constant_24_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+56128);
Constant_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+203584);
Constant_22_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+203840);
Constant_21_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+204096);
Constant_20_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+204352);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+204608);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+270144);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+271168);
Constant_26_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+272192);
Constant_25_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+273216);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+274240);
Constant_13_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+339776);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+340800);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+341824);
Constant_10_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+342848);
Constant_34_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+343872);
Constant_33_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+409408);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+409664);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+409920);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+410176);
Constant_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+410432);
Constant_38_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+557888);
Constant_37_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+558144);
Constant_36_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+558400);
Constant_35_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+558656);
Constant_44_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+558912);
Constant_43_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+624448);
Constant_42_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+625472);
Constant_41_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+626496);
Constant_40_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+627520);
Constant_149_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+628544);
Constant_148_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+694080);
Constant_147_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+694336);
Constant_146_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+694592);
Constant_145_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+694848);
Constant_154_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+695104);
Constant_153_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+842560);
Constant_152_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+842816);
Constant_151_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+843072);
Constant_150_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+843328);
Constant_144_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+843584);
Constant_143_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+909120);
Constant_142_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+910144);
Constant_141_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+911168);
Constant_140_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+912192);
Constant_164_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+913216);
Constant_163_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1044288);
Constant_162_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1044800);
Constant_161_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1045312);
Constant_160_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1045824);
Constant_169_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1046336);
Constant_168_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1636160);
Constant_167_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1636672);
Constant_166_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1637184);
Constant_165_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1637696);
Constant_174_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1638208);
Constant_173_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1900352);
Constant_172_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1902400);
Constant_171_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1904448);
Constant_170_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1906496);
Constant_159_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+1908544);
Constant_158_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2432832);
Constant_157_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2434880);
Constant_156_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2436928);
Constant_155_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2438976);
Constant_179_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2441024);
Constant_178_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2703168);
Constant_177_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2703680);
Constant_176_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2704192);
Constant_175_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2704704);
Constant_184_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+2705216);
Constant_183_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3295040);
Constant_182_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3295552);
Constant_181_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3296064);
Constant_180_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3296576);
Constant_189_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3297088);
Constant_188_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3559232);
Constant_187_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3561280);
Constant_186_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3563328);
Constant_185_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3565376);
Constant_194_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3567424);
Constant_193_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3829568);
Constant_192_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3830080);
Constant_191_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3830592);
Constant_190_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3831104);
Constant_199_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3831616);
Constant_198_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4421440);
Constant_197_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4421952);
Constant_196_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4422464);
Constant_195_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4422976);
Constant_204_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4423488);
Constant_203_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4685632);
Constant_202_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4687680);
Constant_201_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4689728);
Constant_200_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4691776);
Constant_209_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4693824);
Constant_208_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4955968);
Constant_207_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4956480);
Constant_206_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4956992);
Constant_205_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4957504);
Constant_214_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4958016);
Constant_213_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5547840);
Constant_212_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5548352);
Constant_211_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5548864);
Constant_210_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5549376);
Constant_219_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5549888);
Constant_218_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5812032);
Constant_217_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5814080);
Constant_216_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5816128);
Constant_215_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5818176);
Constant_229_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+5820224);
Constant_228_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+6344512);
Constant_227_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+6345536);
Constant_226_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+6346560);
Constant_225_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+6347584);
Constant_234_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+6348608);
Constant_233_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8707904);
Constant_232_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8708928);
Constant_231_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8709952);
Constant_230_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8710976);
Constant_239_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8712000);
Constant_238_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9760576);
Constant_237_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9764672);
Constant_236_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9768768);
Constant_235_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9772864);
Constant_224_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9776960);
Constant_223_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11874112);
Constant_222_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11878208);
Constant_221_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11882304);
Constant_220_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11886400);
Constant_244_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11890496);
Constant_243_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12939072);
Constant_242_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12940096);
Constant_241_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12941120);
Constant_240_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12942144);
Constant_249_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12943168);
Constant_248_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+15302464);
Constant_247_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+15303488);
Constant_246_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+15304512);
Constant_245_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+15305536);
Constant_254_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+15306560);
Constant_253_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16355136);
Constant_252_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16359232);
Constant_251_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16363328);
Constant_250_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16367424);
Constant_259_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16371520);
Constant_258_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17420096);
Constant_257_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17421120);
Constant_256_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17422144);
Constant_255_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17423168);
Constant_264_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17424192);
Constant_263_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19783488);
Constant_262_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19784512);
Constant_261_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19785536);
Constant_260_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19786560);
Constant_269_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19787584);
Constant_268_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20836160);
Constant_267_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20840256);
Constant_266_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20844352);
Constant_265_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20848448);
Constant_49_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20852544);
Constant_48_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21901120);
Constant_47_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21902144);
Constant_46_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21903168);
Constant_45_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21904192);
Constant_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21905216);
Constant_53_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24264512);
Constant_52_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24265536);
Constant_51_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24266560);
Constant_50_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24267584);
Constant_59_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24268608);
Constant_58_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25317184);
Constant_57_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25321280);
Constant_56_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25325376);
Constant_55_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25329472);
Constant_64_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25333568);
Constant_63_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26382144);
Constant_62_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26383168);
Constant_61_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26384192);
Constant_60_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26385216);
Constant_69_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26386240);
Constant_68_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28745536);
Constant_67_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28746560);
Constant_66_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28747584);
Constant_65_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28748608);
Constant_74_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28749632);
Constant_73_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29798208);
Constant_72_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29802304);
Constant_71_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29806400);
Constant_70_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29810496);
Constant_79_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29814592);
Constant_78_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30863168);
Constant_77_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30864192);
Constant_76_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30865216);
Constant_75_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30866240);
Constant_84_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30867264);
Constant_83_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33226560);
Constant_82_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33227584);
Constant_81_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33228608);
Constant_80_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33229632);
Constant_89_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33230656);
Constant_88_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34279232);
Constant_87_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34283328);
Constant_86_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34287424);
Constant_85_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34291520);
Constant_99_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34295616);
Constant_98_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36392768);
Constant_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36394816);
Constant_96_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36396864);
Constant_95_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36398912);
Constant_104_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36400960);
Constant_103_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45838144);
Constant_102_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45840192);
Constant_101_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45842240);
Constant_100_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45844288);
Constant_109_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45846336);
Constant_108_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50040640);
Constant_107_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50048832);
Constant_106_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50057024);
Constant_105_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50065216);
Constant_94_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50073408);
Constant_93_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+58462016);
Constant_92_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+58470208);
Constant_91_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+58478400);
Constant_90_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+58486592);
Constant_114_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+58494784);
Constant_113_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62689088);
Constant_112_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62691136);
Constant_111_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62693184);
Constant_110_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62695232);
Constant_119_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62697280);
Constant_118_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72134464);
Constant_117_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72136512);
Constant_116_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72138560);
Constant_115_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72140608);
Constant_124_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72142656);
Constant_123_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76336960);
Constant_122_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76345152);
Constant_121_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76353344);
Constant_120_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76361536);
Constant_129_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76369728);
Constant_128_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+80564032);
Constant_127_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+80566080);
Constant_126_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+80568128);
Constant_125_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+80570176);
Constant_134_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+80572224);
Constant_133_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90009408);
Constant_132_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90011456);
Constant_131_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90013504);
Constant_130_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90015552);
Constant_139_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+90017600);
Constant_138_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94211904);
Constant_137_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94220096);
Constant_136_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94228288);
Constant_135_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94236480);
Constant_500_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94244672);
Constant_4_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+94252864);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+102453056);
Broadcast_503_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+102453056);

CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,11845632));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 11845632));
Reshape_271_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Pad_273_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+602112);
Reshape_274_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_275_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1236928);
BatchNormInference_276_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4448192);
Relu_277_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4448192);
MaxPool_278_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_281_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_282_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+819200);
BatchNormInference_284_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1622016);
Relu_285_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1622016);
Reshape_286_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_287_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2424832);
BatchNormInference_288_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Relu_289_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Reshape_290_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_291_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1671168);
BatchNormInference_292_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4882432);
Reshape_279_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_280_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+868352);
BatchNormInference_283_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+8093696);
Relu_294_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_295_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Convolution_296_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3276800);
BatchNormInference_297_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4079616);
Relu_298_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4079616);
Reshape_299_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Convolution_300_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4882432);
BatchNormInference_301_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Relu_302_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Reshape_303_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4014080);
Convolution_304_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4079616);
BatchNormInference_305_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7290880);
Relu_307_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Reshape_308_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_309_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
BatchNormInference_310_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+868352);
Relu_311_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+868352);
Reshape_312_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_313_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1671168);
BatchNormInference_314_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_315_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_316_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_317_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6422528);
BatchNormInference_318_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_320_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6422528);
Reshape_323_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_324_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+131072);
BatchNormInference_326_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+532480);
Relu_327_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+532480);
Reshape_328_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+933888);
Convolution_329_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_330_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Relu_331_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Reshape_332_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_333_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
BatchNormInference_334_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Reshape_321_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_322_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+524288);
BatchNormInference_325_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4014080);
Relu_336_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_337_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_338_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1867776);
BatchNormInference_339_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2269184);
Relu_340_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2269184);
Reshape_341_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_342_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2670592);
BatchNormInference_343_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Relu_344_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_345_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2007040);
Convolution_346_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2269184);
BatchNormInference_347_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3874816);
Relu_349_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_350_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_351_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+262144);
BatchNormInference_352_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+663552);
Relu_353_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+663552);
Reshape_354_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_355_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1064960);
BatchNormInference_356_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_357_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_358_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Convolution_359_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
BatchNormInference_360_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_362_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Reshape_363_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_364_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+262144);
BatchNormInference_365_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+663552);
Relu_366_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+663552);
Reshape_367_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_368_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1064960);
BatchNormInference_369_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_370_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_371_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Convolution_372_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+663552);
BatchNormInference_373_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4816896);
Relu_375_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_378_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_379_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2129920);
BatchNormInference_381_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Relu_382_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_383_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1806336);
Convolution_384_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4165632);
BatchNormInference_385_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Relu_386_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_387_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1806336);
Convolution_388_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2854912);
BatchNormInference_389_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_376_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_377_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4505600);
BatchNormInference_380_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_391_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Reshape_392_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_393_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_394_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_395_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_396_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_397_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_398_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_399_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_400_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_401_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2654208);
BatchNormInference_402_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_404_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_405_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_406_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1048576);
BatchNormInference_407_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_408_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_409_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_410_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
BatchNormInference_411_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_412_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_413_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Convolution_414_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
BatchNormInference_415_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_417_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Reshape_418_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_419_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_420_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_421_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_422_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_423_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_424_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_425_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_426_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_427_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2654208);
BatchNormInference_428_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_430_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_431_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_432_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1048576);
BatchNormInference_433_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_434_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_435_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_436_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
BatchNormInference_437_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_438_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_439_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Convolution_440_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
BatchNormInference_441_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_443_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Reshape_444_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_445_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_446_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_447_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_448_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_449_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_450_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Relu_451_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
Reshape_452_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_453_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2654208);
BatchNormInference_454_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_456_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_459_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_460_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_462_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Relu_463_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Reshape_464_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_465_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_466_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Relu_467_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Reshape_468_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_469_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
BatchNormInference_470_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+602112);
Reshape_457_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2408448);
Convolution_458_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_461_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1003520);
Relu_472_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_473_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Convolution_474_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4595712);
BatchNormInference_475_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Relu_476_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Reshape_477_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+501760);
Convolution_478_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9938944);
BatchNormInference_479_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Relu_480_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Reshape_481_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+501760);
Convolution_482_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4696064);
BatchNormInference_483_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Relu_485_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Reshape_486_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1204224);
Convolution_487_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_488_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Relu_489_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Reshape_490_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1204224);
Convolution_491_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BatchNormInference_492_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Relu_493_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Reshape_494_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1204224);
Convolution_495_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+200704);
BatchNormInference_496_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1204224);
Relu_498_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Sum_499_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Divide_501_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Dot_502_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_504_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
 // name=Constant_272
Constant_float_cuda_Constant_272(0, Constant_272_0);
 // name=cg/conv0/conv2d/kernel/read/_6__cf__6
Constant_float_cuda_Constant_9(0, Constant_9_0);
 // name=cg/conv0/batchnorm0/moving_variance/read/_5__cf__5
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=cg/conv0/batchnorm0/moving_mean/read/_4__cf__4
Constant_float_cuda_Constant_7(0, Constant_7_0);
 // name=cg/conv0/batchnorm0/gamma/read/_3__cf__3
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=cg/conv0/batchnorm0/beta/read/_2__cf__2
Constant_float_cuda_Constant_5(0, Constant_5_0);
 // name=cg/resnet_v10/conv2/conv2d/kernel/read/_16__cf__16
Constant_float_cuda_Constant_19(0, Constant_19_0);
 // name=cg/resnet_v10/conv2/batchnorm2/moving_variance/read/_15__cf__15
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=cg/resnet_v10/conv2/batchnorm2/moving_mean/read/_14__cf__14
Constant_float_cuda_Constant_17(0, Constant_17_0);
 // name=cg/resnet_v10/conv2/batchnorm2/gamma/read/_13__cf__13
Constant_float_cuda_Constant_16(0, Constant_16_0);
 // name=cg/resnet_v10/conv2/batchnorm2/beta/read/_12__cf__12
Constant_float_cuda_Constant_15(0, Constant_15_0);
 // name=cg/resnet_v10/conv3/conv2d/kernel/read/_21__cf__21
Constant_float_cuda_Constant_24(0, Constant_24_0);
 // name=cg/resnet_v10/conv3/batchnorm3/moving_variance/read/_20__cf__20
Constant_float_cuda_Constant_23(0, Constant_23_0);
 // name=cg/resnet_v10/conv3/batchnorm3/moving_mean/read/_19__cf__19
Constant_float_cuda_Constant_22(0, Constant_22_0);
 // name=cg/resnet_v10/conv3/batchnorm3/gamma/read/_18__cf__18
Constant_float_cuda_Constant_21(0, Constant_21_0);
 // name=cg/resnet_v10/conv3/batchnorm3/beta/read/_17__cf__17
Constant_float_cuda_Constant_20(0, Constant_20_0);
 // name=cg/resnet_v10/conv4/conv2d/kernel/read/_26__cf__26
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=cg/resnet_v10/conv4/batchnorm4/moving_variance/read/_25__cf__25
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=cg/resnet_v10/conv4/batchnorm4/moving_mean/read/_24__cf__24
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=cg/resnet_v10/conv4/batchnorm4/gamma/read/_23__cf__23
Constant_float_cuda_Constant_26(0, Constant_26_0);
 // name=cg/resnet_v10/conv4/batchnorm4/beta/read/_22__cf__22
Constant_float_cuda_Constant_25(0, Constant_25_0);
 // name=cg/resnet_v10/conv1/conv2d/kernel/read/_11__cf__11
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=cg/resnet_v10/conv1/batchnorm1/moving_variance/read/_10__cf__10
Constant_float_cuda_Constant_13(0, Constant_13_0);
 // name=cg/resnet_v10/conv1/batchnorm1/moving_mean/read/_9__cf__9
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=cg/resnet_v10/conv1/batchnorm1/gamma/read/_8__cf__8
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=cg/resnet_v10/conv1/batchnorm1/beta/read/_7__cf__7
Constant_float_cuda_Constant_10(0, Constant_10_0);
 // name=cg/resnet_v11/conv5/conv2d/kernel/read/_31__cf__31
Constant_float_cuda_Constant_34(0, Constant_34_0);
 // name=cg/resnet_v11/conv5/batchnorm5/moving_variance/read/_30__cf__30
Constant_float_cuda_Constant_33(0, Constant_33_0);
 // name=cg/resnet_v11/conv5/batchnorm5/moving_mean/read/_29__cf__29
Constant_float_cuda_Constant_32(0, Constant_32_0);
 // name=cg/resnet_v11/conv5/batchnorm5/gamma/read/_28__cf__28
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=cg/resnet_v11/conv5/batchnorm5/beta/read/_27__cf__27
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=cg/resnet_v11/conv6/conv2d/kernel/read/_36__cf__36
Constant_float_cuda_Constant_39(0, Constant_39_0);
 // name=cg/resnet_v11/conv6/batchnorm6/moving_variance/read/_35__cf__35
Constant_float_cuda_Constant_38(0, Constant_38_0);
 // name=cg/resnet_v11/conv6/batchnorm6/moving_mean/read/_34__cf__34
Constant_float_cuda_Constant_37(0, Constant_37_0);
 // name=cg/resnet_v11/conv6/batchnorm6/gamma/read/_33__cf__33
Constant_float_cuda_Constant_36(0, Constant_36_0);
 // name=cg/resnet_v11/conv6/batchnorm6/beta/read/_32__cf__32
Constant_float_cuda_Constant_35(0, Constant_35_0);
 // name=cg/resnet_v11/conv7/conv2d/kernel/read/_41__cf__41
Constant_float_cuda_Constant_44(0, Constant_44_0);
 // name=cg/resnet_v11/conv7/batchnorm7/moving_variance/read/_40__cf__40
Constant_float_cuda_Constant_43(0, Constant_43_0);
 // name=cg/resnet_v11/conv7/batchnorm7/moving_mean/read/_39__cf__39
Constant_float_cuda_Constant_42(0, Constant_42_0);
 // name=cg/resnet_v11/conv7/batchnorm7/gamma/read/_38__cf__38
Constant_float_cuda_Constant_41(0, Constant_41_0);
 // name=cg/resnet_v11/conv7/batchnorm7/beta/read/_37__cf__37
Constant_float_cuda_Constant_40(0, Constant_40_0);
 // name=cg/resnet_v12/conv8/conv2d/kernel/read/_146__cf__146
Constant_float_cuda_Constant_149(0, Constant_149_0);
 // name=cg/resnet_v12/conv8/batchnorm8/moving_variance/read/_145__cf__145
Constant_float_cuda_Constant_148(0, Constant_148_0);
 // name=cg/resnet_v12/conv8/batchnorm8/moving_mean/read/_144__cf__144
Constant_float_cuda_Constant_147(0, Constant_147_0);
 // name=cg/resnet_v12/conv8/batchnorm8/gamma/read/_143__cf__143
Constant_float_cuda_Constant_146(0, Constant_146_0);
 // name=cg/resnet_v12/conv8/batchnorm8/beta/read/_142__cf__142
Constant_float_cuda_Constant_145(0, Constant_145_0);
 // name=cg/resnet_v12/conv9/conv2d/kernel/read/_151__cf__151
Constant_float_cuda_Constant_154(0, Constant_154_0);
 // name=cg/resnet_v12/conv9/batchnorm9/moving_variance/read/_150__cf__150
Constant_float_cuda_Constant_153(0, Constant_153_0);
 // name=cg/resnet_v12/conv9/batchnorm9/moving_mean/read/_149__cf__149
Constant_float_cuda_Constant_152(0, Constant_152_0);
 // name=cg/resnet_v12/conv9/batchnorm9/gamma/read/_148__cf__148
Constant_float_cuda_Constant_151(0, Constant_151_0);
 // name=cg/resnet_v12/conv9/batchnorm9/beta/read/_147__cf__147
Constant_float_cuda_Constant_150(0, Constant_150_0);
 // name=cg/resnet_v12/conv10/conv2d/kernel/read/_141__cf__141
Constant_float_cuda_Constant_144(0, Constant_144_0);
 // name=cg/resnet_v12/conv10/batchnorm10/moving_variance/read/_140__cf__140
Constant_float_cuda_Constant_143(0, Constant_143_0);
 // name=cg/resnet_v12/conv10/batchnorm10/moving_mean/read/_139__cf__139
Constant_float_cuda_Constant_142(0, Constant_142_0);
 // name=cg/resnet_v12/conv10/batchnorm10/gamma/read/_138__cf__138
Constant_float_cuda_Constant_141(0, Constant_141_0);
 // name=cg/resnet_v12/conv10/batchnorm10/beta/read/_137__cf__137
Constant_float_cuda_Constant_140(0, Constant_140_0);
 // name=cg/resnet_v13/conv12/conv2d/kernel/read/_161__cf__161
Constant_float_cuda_Constant_164(0, Constant_164_0);
 // name=cg/resnet_v13/conv12/batchnorm12/moving_variance/read/_160__cf__160
Constant_float_cuda_Constant_163(0, Constant_163_0);
 // name=cg/resnet_v13/conv12/batchnorm12/moving_mean/read/_159__cf__159
Constant_float_cuda_Constant_162(0, Constant_162_0);
 // name=cg/resnet_v13/conv12/batchnorm12/gamma/read/_158__cf__158
Constant_float_cuda_Constant_161(0, Constant_161_0);
 // name=cg/resnet_v13/conv12/batchnorm12/beta/read/_157__cf__157
Constant_float_cuda_Constant_160(0, Constant_160_0);
 // name=cg/resnet_v13/conv13/conv2d/kernel/read/_166__cf__166
Constant_float_cuda_Constant_169(0, Constant_169_0);
 // name=cg/resnet_v13/conv13/batchnorm13/moving_variance/read/_165__cf__165
Constant_float_cuda_Constant_168(0, Constant_168_0);
 // name=cg/resnet_v13/conv13/batchnorm13/moving_mean/read/_164__cf__164
Constant_float_cuda_Constant_167(0, Constant_167_0);
 // name=cg/resnet_v13/conv13/batchnorm13/gamma/read/_163__cf__163
Constant_float_cuda_Constant_166(0, Constant_166_0);
 // name=cg/resnet_v13/conv13/batchnorm13/beta/read/_162__cf__162
Constant_float_cuda_Constant_165(0, Constant_165_0);
 // name=cg/resnet_v13/conv14/conv2d/kernel/read/_171__cf__171
Constant_float_cuda_Constant_174(0, Constant_174_0);
 // name=cg/resnet_v13/conv14/batchnorm14/moving_variance/read/_170__cf__170
Constant_float_cuda_Constant_173(0, Constant_173_0);
 // name=cg/resnet_v13/conv14/batchnorm14/moving_mean/read/_169__cf__169
Constant_float_cuda_Constant_172(0, Constant_172_0);
 // name=cg/resnet_v13/conv14/batchnorm14/gamma/read/_168__cf__168
Constant_float_cuda_Constant_171(0, Constant_171_0);
 // name=cg/resnet_v13/conv14/batchnorm14/beta/read/_167__cf__167
Constant_float_cuda_Constant_170(0, Constant_170_0);
 // name=cg/resnet_v13/conv11/conv2d/kernel/read/_156__cf__156
Constant_float_cuda_Constant_159(0, Constant_159_0);
 // name=cg/resnet_v13/conv11/batchnorm11/moving_variance/read/_155__cf__155
Constant_float_cuda_Constant_158(0, Constant_158_0);
 // name=cg/resnet_v13/conv11/batchnorm11/moving_mean/read/_154__cf__154
Constant_float_cuda_Constant_157(0, Constant_157_0);
 // name=cg/resnet_v13/conv11/batchnorm11/gamma/read/_153__cf__153
Constant_float_cuda_Constant_156(0, Constant_156_0);
 // name=cg/resnet_v13/conv11/batchnorm11/beta/read/_152__cf__152
Constant_float_cuda_Constant_155(0, Constant_155_0);
 // name=cg/resnet_v14/conv15/conv2d/kernel/read/_176__cf__176
Constant_float_cuda_Constant_179(0, Constant_179_0);
 // name=cg/resnet_v14/conv15/batchnorm15/moving_variance/read/_175__cf__175
Constant_float_cuda_Constant_178(0, Constant_178_0);
 // name=cg/resnet_v14/conv15/batchnorm15/moving_mean/read/_174__cf__174
Constant_float_cuda_Constant_177(0, Constant_177_0);
 // name=cg/resnet_v14/conv15/batchnorm15/gamma/read/_173__cf__173
Constant_float_cuda_Constant_176(0, Constant_176_0);
 // name=cg/resnet_v14/conv15/batchnorm15/beta/read/_172__cf__172
Constant_float_cuda_Constant_175(0, Constant_175_0);
 // name=cg/resnet_v14/conv16/conv2d/kernel/read/_181__cf__181
Constant_float_cuda_Constant_184(0, Constant_184_0);
 // name=cg/resnet_v14/conv16/batchnorm16/moving_variance/read/_180__cf__180
Constant_float_cuda_Constant_183(0, Constant_183_0);
 // name=cg/resnet_v14/conv16/batchnorm16/moving_mean/read/_179__cf__179
Constant_float_cuda_Constant_182(0, Constant_182_0);
 // name=cg/resnet_v14/conv16/batchnorm16/gamma/read/_178__cf__178
Constant_float_cuda_Constant_181(0, Constant_181_0);
 // name=cg/resnet_v14/conv16/batchnorm16/beta/read/_177__cf__177
Constant_float_cuda_Constant_180(0, Constant_180_0);
 // name=cg/resnet_v14/conv17/conv2d/kernel/read/_186__cf__186
Constant_float_cuda_Constant_189(0, Constant_189_0);
 // name=cg/resnet_v14/conv17/batchnorm17/moving_variance/read/_185__cf__185
Constant_float_cuda_Constant_188(0, Constant_188_0);
 // name=cg/resnet_v14/conv17/batchnorm17/moving_mean/read/_184__cf__184
Constant_float_cuda_Constant_187(0, Constant_187_0);
 // name=cg/resnet_v14/conv17/batchnorm17/gamma/read/_183__cf__183
Constant_float_cuda_Constant_186(0, Constant_186_0);
 // name=cg/resnet_v14/conv17/batchnorm17/beta/read/_182__cf__182
Constant_float_cuda_Constant_185(0, Constant_185_0);
 // name=cg/resnet_v15/conv18/conv2d/kernel/read/_191__cf__191
Constant_float_cuda_Constant_194(0, Constant_194_0);
 // name=cg/resnet_v15/conv18/batchnorm18/moving_variance/read/_190__cf__190
Constant_float_cuda_Constant_193(0, Constant_193_0);
 // name=cg/resnet_v15/conv18/batchnorm18/moving_mean/read/_189__cf__189
Constant_float_cuda_Constant_192(0, Constant_192_0);
 // name=cg/resnet_v15/conv18/batchnorm18/gamma/read/_188__cf__188
Constant_float_cuda_Constant_191(0, Constant_191_0);
 // name=cg/resnet_v15/conv18/batchnorm18/beta/read/_187__cf__187
Constant_float_cuda_Constant_190(0, Constant_190_0);
 // name=cg/resnet_v15/conv19/conv2d/kernel/read/_196__cf__196
Constant_float_cuda_Constant_199(0, Constant_199_0);
 // name=cg/resnet_v15/conv19/batchnorm19/moving_variance/read/_195__cf__195
Constant_float_cuda_Constant_198(0, Constant_198_0);
 // name=cg/resnet_v15/conv19/batchnorm19/moving_mean/read/_194__cf__194
Constant_float_cuda_Constant_197(0, Constant_197_0);
 // name=cg/resnet_v15/conv19/batchnorm19/gamma/read/_193__cf__193
Constant_float_cuda_Constant_196(0, Constant_196_0);
 // name=cg/resnet_v15/conv19/batchnorm19/beta/read/_192__cf__192
Constant_float_cuda_Constant_195(0, Constant_195_0);
 // name=cg/resnet_v15/conv20/conv2d/kernel/read/_201__cf__201
Constant_float_cuda_Constant_204(0, Constant_204_0);
 // name=cg/resnet_v15/conv20/batchnorm20/moving_variance/read/_200__cf__200
Constant_float_cuda_Constant_203(0, Constant_203_0);
 // name=cg/resnet_v15/conv20/batchnorm20/moving_mean/read/_199__cf__199
Constant_float_cuda_Constant_202(0, Constant_202_0);
 // name=cg/resnet_v15/conv20/batchnorm20/gamma/read/_198__cf__198
Constant_float_cuda_Constant_201(0, Constant_201_0);
 // name=cg/resnet_v15/conv20/batchnorm20/beta/read/_197__cf__197
Constant_float_cuda_Constant_200(0, Constant_200_0);
 // name=cg/resnet_v16/conv21/conv2d/kernel/read/_206__cf__206
Constant_float_cuda_Constant_209(0, Constant_209_0);
 // name=cg/resnet_v16/conv21/batchnorm21/moving_variance/read/_205__cf__205
Constant_float_cuda_Constant_208(0, Constant_208_0);
 // name=cg/resnet_v16/conv21/batchnorm21/moving_mean/read/_204__cf__204
Constant_float_cuda_Constant_207(0, Constant_207_0);
 // name=cg/resnet_v16/conv21/batchnorm21/gamma/read/_203__cf__203
Constant_float_cuda_Constant_206(0, Constant_206_0);
 // name=cg/resnet_v16/conv21/batchnorm21/beta/read/_202__cf__202
Constant_float_cuda_Constant_205(0, Constant_205_0);
 // name=cg/resnet_v16/conv22/conv2d/kernel/read/_211__cf__211
Constant_float_cuda_Constant_214(0, Constant_214_0);
 // name=cg/resnet_v16/conv22/batchnorm22/moving_variance/read/_210__cf__210
Constant_float_cuda_Constant_213(0, Constant_213_0);
 // name=cg/resnet_v16/conv22/batchnorm22/moving_mean/read/_209__cf__209
Constant_float_cuda_Constant_212(0, Constant_212_0);
 // name=cg/resnet_v16/conv22/batchnorm22/gamma/read/_208__cf__208
Constant_float_cuda_Constant_211(0, Constant_211_0);
 // name=cg/resnet_v16/conv22/batchnorm22/beta/read/_207__cf__207
Constant_float_cuda_Constant_210(0, Constant_210_0);
 // name=cg/resnet_v16/conv23/conv2d/kernel/read/_216__cf__216
Constant_float_cuda_Constant_219(0, Constant_219_0);
 // name=cg/resnet_v16/conv23/batchnorm23/moving_variance/read/_215__cf__215
Constant_float_cuda_Constant_218(0, Constant_218_0);
 // name=cg/resnet_v16/conv23/batchnorm23/moving_mean/read/_214__cf__214
Constant_float_cuda_Constant_217(0, Constant_217_0);
 // name=cg/resnet_v16/conv23/batchnorm23/gamma/read/_213__cf__213
Constant_float_cuda_Constant_216(0, Constant_216_0);
 // name=cg/resnet_v16/conv23/batchnorm23/beta/read/_212__cf__212
Constant_float_cuda_Constant_215(0, Constant_215_0);
 // name=cg/resnet_v17/conv25/conv2d/kernel/read/_226__cf__226
Constant_float_cuda_Constant_229(0, Constant_229_0);
 // name=cg/resnet_v17/conv25/batchnorm25/moving_variance/read/_225__cf__225
Constant_float_cuda_Constant_228(0, Constant_228_0);
 // name=cg/resnet_v17/conv25/batchnorm25/moving_mean/read/_224__cf__224
Constant_float_cuda_Constant_227(0, Constant_227_0);
 // name=cg/resnet_v17/conv25/batchnorm25/gamma/read/_223__cf__223
Constant_float_cuda_Constant_226(0, Constant_226_0);
 // name=cg/resnet_v17/conv25/batchnorm25/beta/read/_222__cf__222
Constant_float_cuda_Constant_225(0, Constant_225_0);
 // name=cg/resnet_v17/conv26/conv2d/kernel/read/_231__cf__231
Constant_float_cuda_Constant_234(0, Constant_234_0);
 // name=cg/resnet_v17/conv26/batchnorm26/moving_variance/read/_230__cf__230
Constant_float_cuda_Constant_233(0, Constant_233_0);
 // name=cg/resnet_v17/conv26/batchnorm26/moving_mean/read/_229__cf__229
Constant_float_cuda_Constant_232(0, Constant_232_0);
 // name=cg/resnet_v17/conv26/batchnorm26/gamma/read/_228__cf__228
Constant_float_cuda_Constant_231(0, Constant_231_0);
 // name=cg/resnet_v17/conv26/batchnorm26/beta/read/_227__cf__227
Constant_float_cuda_Constant_230(0, Constant_230_0);
 // name=cg/resnet_v17/conv27/conv2d/kernel/read/_236__cf__236
Constant_float_cuda_Constant_239(0, Constant_239_0);
 // name=cg/resnet_v17/conv27/batchnorm27/moving_variance/read/_235__cf__235
Constant_float_cuda_Constant_238(0, Constant_238_0);
 // name=cg/resnet_v17/conv27/batchnorm27/moving_mean/read/_234__cf__234
Constant_float_cuda_Constant_237(0, Constant_237_0);
 // name=cg/resnet_v17/conv27/batchnorm27/gamma/read/_233__cf__233
Constant_float_cuda_Constant_236(0, Constant_236_0);
 // name=cg/resnet_v17/conv27/batchnorm27/beta/read/_232__cf__232
Constant_float_cuda_Constant_235(0, Constant_235_0);
 // name=cg/resnet_v17/conv24/conv2d/kernel/read/_221__cf__221
Constant_float_cuda_Constant_224(0, Constant_224_0);
 // name=cg/resnet_v17/conv24/batchnorm24/moving_variance/read/_220__cf__220
Constant_float_cuda_Constant_223(0, Constant_223_0);
 // name=cg/resnet_v17/conv24/batchnorm24/moving_mean/read/_219__cf__219
Constant_float_cuda_Constant_222(0, Constant_222_0);
 // name=cg/resnet_v17/conv24/batchnorm24/gamma/read/_218__cf__218
Constant_float_cuda_Constant_221(0, Constant_221_0);
 // name=cg/resnet_v17/conv24/batchnorm24/beta/read/_217__cf__217
Constant_float_cuda_Constant_220(0, Constant_220_0);
 // name=cg/resnet_v18/conv28/conv2d/kernel/read/_241__cf__241
Constant_float_cuda_Constant_244(0, Constant_244_0);
 // name=cg/resnet_v18/conv28/batchnorm28/moving_variance/read/_240__cf__240
Constant_float_cuda_Constant_243(0, Constant_243_0);
 // name=cg/resnet_v18/conv28/batchnorm28/moving_mean/read/_239__cf__239
Constant_float_cuda_Constant_242(0, Constant_242_0);
 // name=cg/resnet_v18/conv28/batchnorm28/gamma/read/_238__cf__238
Constant_float_cuda_Constant_241(0, Constant_241_0);
 // name=cg/resnet_v18/conv28/batchnorm28/beta/read/_237__cf__237
Constant_float_cuda_Constant_240(0, Constant_240_0);
 // name=cg/resnet_v18/conv29/conv2d/kernel/read/_246__cf__246
Constant_float_cuda_Constant_249(0, Constant_249_0);
 // name=cg/resnet_v18/conv29/batchnorm29/moving_variance/read/_245__cf__245
Constant_float_cuda_Constant_248(0, Constant_248_0);
 // name=cg/resnet_v18/conv29/batchnorm29/moving_mean/read/_244__cf__244
Constant_float_cuda_Constant_247(0, Constant_247_0);
 // name=cg/resnet_v18/conv29/batchnorm29/gamma/read/_243__cf__243
Constant_float_cuda_Constant_246(0, Constant_246_0);
 // name=cg/resnet_v18/conv29/batchnorm29/beta/read/_242__cf__242
Constant_float_cuda_Constant_245(0, Constant_245_0);
 // name=cg/resnet_v18/conv30/conv2d/kernel/read/_251__cf__251
Constant_float_cuda_Constant_254(0, Constant_254_0);
 // name=cg/resnet_v18/conv30/batchnorm30/moving_variance/read/_250__cf__250
Constant_float_cuda_Constant_253(0, Constant_253_0);
 // name=cg/resnet_v18/conv30/batchnorm30/moving_mean/read/_249__cf__249
Constant_float_cuda_Constant_252(0, Constant_252_0);
 // name=cg/resnet_v18/conv30/batchnorm30/gamma/read/_248__cf__248
Constant_float_cuda_Constant_251(0, Constant_251_0);
 // name=cg/resnet_v18/conv30/batchnorm30/beta/read/_247__cf__247
Constant_float_cuda_Constant_250(0, Constant_250_0);
 // name=cg/resnet_v19/conv31/conv2d/kernel/read/_256__cf__256
Constant_float_cuda_Constant_259(0, Constant_259_0);
 // name=cg/resnet_v19/conv31/batchnorm31/moving_variance/read/_255__cf__255
Constant_float_cuda_Constant_258(0, Constant_258_0);
 // name=cg/resnet_v19/conv31/batchnorm31/moving_mean/read/_254__cf__254
Constant_float_cuda_Constant_257(0, Constant_257_0);
 // name=cg/resnet_v19/conv31/batchnorm31/gamma/read/_253__cf__253
Constant_float_cuda_Constant_256(0, Constant_256_0);
 // name=cg/resnet_v19/conv31/batchnorm31/beta/read/_252__cf__252
Constant_float_cuda_Constant_255(0, Constant_255_0);
 // name=cg/resnet_v19/conv32/conv2d/kernel/read/_261__cf__261
Constant_float_cuda_Constant_264(0, Constant_264_0);
 // name=cg/resnet_v19/conv32/batchnorm32/moving_variance/read/_260__cf__260
Constant_float_cuda_Constant_263(0, Constant_263_0);
 // name=cg/resnet_v19/conv32/batchnorm32/moving_mean/read/_259__cf__259
Constant_float_cuda_Constant_262(0, Constant_262_0);
 // name=cg/resnet_v19/conv32/batchnorm32/gamma/read/_258__cf__258
Constant_float_cuda_Constant_261(0, Constant_261_0);
 // name=cg/resnet_v19/conv32/batchnorm32/beta/read/_257__cf__257
Constant_float_cuda_Constant_260(0, Constant_260_0);
 // name=cg/resnet_v19/conv33/conv2d/kernel/read/_266__cf__266
Constant_float_cuda_Constant_269(0, Constant_269_0);
 // name=cg/resnet_v19/conv33/batchnorm33/moving_variance/read/_265__cf__265
Constant_float_cuda_Constant_268(0, Constant_268_0);
 // name=cg/resnet_v19/conv33/batchnorm33/moving_mean/read/_264__cf__264
Constant_float_cuda_Constant_267(0, Constant_267_0);
 // name=cg/resnet_v19/conv33/batchnorm33/gamma/read/_263__cf__263
Constant_float_cuda_Constant_266(0, Constant_266_0);
 // name=cg/resnet_v19/conv33/batchnorm33/beta/read/_262__cf__262
Constant_float_cuda_Constant_265(0, Constant_265_0);
 // name=cg/resnet_v110/conv34/conv2d/kernel/read/_46__cf__46
Constant_float_cuda_Constant_49(0, Constant_49_0);
 // name=cg/resnet_v110/conv34/batchnorm34/moving_variance/read/_45__cf__45
Constant_float_cuda_Constant_48(0, Constant_48_0);
 // name=cg/resnet_v110/conv34/batchnorm34/moving_mean/read/_44__cf__44
Constant_float_cuda_Constant_47(0, Constant_47_0);
 // name=cg/resnet_v110/conv34/batchnorm34/gamma/read/_43__cf__43
Constant_float_cuda_Constant_46(0, Constant_46_0);
 // name=cg/resnet_v110/conv34/batchnorm34/beta/read/_42__cf__42
Constant_float_cuda_Constant_45(0, Constant_45_0);
 // name=cg/resnet_v110/conv35/conv2d/kernel/read/_51__cf__51
Constant_float_cuda_Constant_54(0, Constant_54_0);
 // name=cg/resnet_v110/conv35/batchnorm35/moving_variance/read/_50__cf__50
Constant_float_cuda_Constant_53(0, Constant_53_0);
 // name=cg/resnet_v110/conv35/batchnorm35/moving_mean/read/_49__cf__49
Constant_float_cuda_Constant_52(0, Constant_52_0);
 // name=cg/resnet_v110/conv35/batchnorm35/gamma/read/_48__cf__48
Constant_float_cuda_Constant_51(0, Constant_51_0);
 // name=cg/resnet_v110/conv35/batchnorm35/beta/read/_47__cf__47
Constant_float_cuda_Constant_50(0, Constant_50_0);
 // name=cg/resnet_v110/conv36/conv2d/kernel/read/_56__cf__56
Constant_float_cuda_Constant_59(0, Constant_59_0);
 // name=cg/resnet_v110/conv36/batchnorm36/moving_variance/read/_55__cf__55
Constant_float_cuda_Constant_58(0, Constant_58_0);
 // name=cg/resnet_v110/conv36/batchnorm36/moving_mean/read/_54__cf__54
Constant_float_cuda_Constant_57(0, Constant_57_0);
 // name=cg/resnet_v110/conv36/batchnorm36/gamma/read/_53__cf__53
Constant_float_cuda_Constant_56(0, Constant_56_0);
 // name=cg/resnet_v110/conv36/batchnorm36/beta/read/_52__cf__52
Constant_float_cuda_Constant_55(0, Constant_55_0);
 // name=cg/resnet_v111/conv37/conv2d/kernel/read/_61__cf__61
Constant_float_cuda_Constant_64(0, Constant_64_0);
 // name=cg/resnet_v111/conv37/batchnorm37/moving_variance/read/_60__cf__60
Constant_float_cuda_Constant_63(0, Constant_63_0);
 // name=cg/resnet_v111/conv37/batchnorm37/moving_mean/read/_59__cf__59
Constant_float_cuda_Constant_62(0, Constant_62_0);
 // name=cg/resnet_v111/conv37/batchnorm37/gamma/read/_58__cf__58
Constant_float_cuda_Constant_61(0, Constant_61_0);
 // name=cg/resnet_v111/conv37/batchnorm37/beta/read/_57__cf__57
Constant_float_cuda_Constant_60(0, Constant_60_0);
 // name=cg/resnet_v111/conv38/conv2d/kernel/read/_66__cf__66
Constant_float_cuda_Constant_69(0, Constant_69_0);
 // name=cg/resnet_v111/conv38/batchnorm38/moving_variance/read/_65__cf__65
Constant_float_cuda_Constant_68(0, Constant_68_0);
 // name=cg/resnet_v111/conv38/batchnorm38/moving_mean/read/_64__cf__64
Constant_float_cuda_Constant_67(0, Constant_67_0);
 // name=cg/resnet_v111/conv38/batchnorm38/gamma/read/_63__cf__63
Constant_float_cuda_Constant_66(0, Constant_66_0);
 // name=cg/resnet_v111/conv38/batchnorm38/beta/read/_62__cf__62
Constant_float_cuda_Constant_65(0, Constant_65_0);
 // name=cg/resnet_v111/conv39/conv2d/kernel/read/_71__cf__71
Constant_float_cuda_Constant_74(0, Constant_74_0);
 // name=cg/resnet_v111/conv39/batchnorm39/moving_variance/read/_70__cf__70
Constant_float_cuda_Constant_73(0, Constant_73_0);
 // name=cg/resnet_v111/conv39/batchnorm39/moving_mean/read/_69__cf__69
Constant_float_cuda_Constant_72(0, Constant_72_0);
 // name=cg/resnet_v111/conv39/batchnorm39/gamma/read/_68__cf__68
Constant_float_cuda_Constant_71(0, Constant_71_0);
 // name=cg/resnet_v111/conv39/batchnorm39/beta/read/_67__cf__67
Constant_float_cuda_Constant_70(0, Constant_70_0);
 // name=cg/resnet_v112/conv40/conv2d/kernel/read/_76__cf__76
Constant_float_cuda_Constant_79(0, Constant_79_0);
 // name=cg/resnet_v112/conv40/batchnorm40/moving_variance/read/_75__cf__75
Constant_float_cuda_Constant_78(0, Constant_78_0);
 // name=cg/resnet_v112/conv40/batchnorm40/moving_mean/read/_74__cf__74
Constant_float_cuda_Constant_77(0, Constant_77_0);
 // name=cg/resnet_v112/conv40/batchnorm40/gamma/read/_73__cf__73
Constant_float_cuda_Constant_76(0, Constant_76_0);
 // name=cg/resnet_v112/conv40/batchnorm40/beta/read/_72__cf__72
Constant_float_cuda_Constant_75(0, Constant_75_0);
 // name=cg/resnet_v112/conv41/conv2d/kernel/read/_81__cf__81
Constant_float_cuda_Constant_84(0, Constant_84_0);
 // name=cg/resnet_v112/conv41/batchnorm41/moving_variance/read/_80__cf__80
Constant_float_cuda_Constant_83(0, Constant_83_0);
 // name=cg/resnet_v112/conv41/batchnorm41/moving_mean/read/_79__cf__79
Constant_float_cuda_Constant_82(0, Constant_82_0);
 // name=cg/resnet_v112/conv41/batchnorm41/gamma/read/_78__cf__78
Constant_float_cuda_Constant_81(0, Constant_81_0);
 // name=cg/resnet_v112/conv41/batchnorm41/beta/read/_77__cf__77
Constant_float_cuda_Constant_80(0, Constant_80_0);
 // name=cg/resnet_v112/conv42/conv2d/kernel/read/_86__cf__86
Constant_float_cuda_Constant_89(0, Constant_89_0);
 // name=cg/resnet_v112/conv42/batchnorm42/moving_variance/read/_85__cf__85
Constant_float_cuda_Constant_88(0, Constant_88_0);
 // name=cg/resnet_v112/conv42/batchnorm42/moving_mean/read/_84__cf__84
Constant_float_cuda_Constant_87(0, Constant_87_0);
 // name=cg/resnet_v112/conv42/batchnorm42/gamma/read/_83__cf__83
Constant_float_cuda_Constant_86(0, Constant_86_0);
 // name=cg/resnet_v112/conv42/batchnorm42/beta/read/_82__cf__82
Constant_float_cuda_Constant_85(0, Constant_85_0);
 // name=cg/resnet_v113/conv44/conv2d/kernel/read/_96__cf__96
Constant_float_cuda_Constant_99(0, Constant_99_0);
 // name=cg/resnet_v113/conv44/batchnorm44/moving_variance/read/_95__cf__95
Constant_float_cuda_Constant_98(0, Constant_98_0);
 // name=cg/resnet_v113/conv44/batchnorm44/moving_mean/read/_94__cf__94
Constant_float_cuda_Constant_97(0, Constant_97_0);
 // name=cg/resnet_v113/conv44/batchnorm44/gamma/read/_93__cf__93
Constant_float_cuda_Constant_96(0, Constant_96_0);
 // name=cg/resnet_v113/conv44/batchnorm44/beta/read/_92__cf__92
Constant_float_cuda_Constant_95(0, Constant_95_0);
 // name=cg/resnet_v113/conv45/conv2d/kernel/read/_101__cf__101
Constant_float_cuda_Constant_104(0, Constant_104_0);
 // name=cg/resnet_v113/conv45/batchnorm45/moving_variance/read/_100__cf__100
Constant_float_cuda_Constant_103(0, Constant_103_0);
 // name=cg/resnet_v113/conv45/batchnorm45/moving_mean/read/_99__cf__99
Constant_float_cuda_Constant_102(0, Constant_102_0);
 // name=cg/resnet_v113/conv45/batchnorm45/gamma/read/_98__cf__98
Constant_float_cuda_Constant_101(0, Constant_101_0);
 // name=cg/resnet_v113/conv45/batchnorm45/beta/read/_97__cf__97
Constant_float_cuda_Constant_100(0, Constant_100_0);
 // name=cg/resnet_v113/conv46/conv2d/kernel/read/_106__cf__106
Constant_float_cuda_Constant_109(0, Constant_109_0);
 // name=cg/resnet_v113/conv46/batchnorm46/moving_variance/read/_105__cf__105
Constant_float_cuda_Constant_108(0, Constant_108_0);
 // name=cg/resnet_v113/conv46/batchnorm46/moving_mean/read/_104__cf__104
Constant_float_cuda_Constant_107(0, Constant_107_0);
 // name=cg/resnet_v113/conv46/batchnorm46/gamma/read/_103__cf__103
Constant_float_cuda_Constant_106(0, Constant_106_0);
 // name=cg/resnet_v113/conv46/batchnorm46/beta/read/_102__cf__102
Constant_float_cuda_Constant_105(0, Constant_105_0);
 // name=cg/resnet_v113/conv43/conv2d/kernel/read/_91__cf__91
Constant_float_cuda_Constant_94(0, Constant_94_0);
 // name=cg/resnet_v113/conv43/batchnorm43/moving_variance/read/_90__cf__90
Constant_float_cuda_Constant_93(0, Constant_93_0);
 // name=cg/resnet_v113/conv43/batchnorm43/moving_mean/read/_89__cf__89
Constant_float_cuda_Constant_92(0, Constant_92_0);
 // name=cg/resnet_v113/conv43/batchnorm43/gamma/read/_88__cf__88
Constant_float_cuda_Constant_91(0, Constant_91_0);
 // name=cg/resnet_v113/conv43/batchnorm43/beta/read/_87__cf__87
Constant_float_cuda_Constant_90(0, Constant_90_0);
 // name=cg/resnet_v114/conv47/conv2d/kernel/read/_111__cf__111
Constant_float_cuda_Constant_114(0, Constant_114_0);
 // name=cg/resnet_v114/conv47/batchnorm47/moving_variance/read/_110__cf__110
Constant_float_cuda_Constant_113(0, Constant_113_0);
 // name=cg/resnet_v114/conv47/batchnorm47/moving_mean/read/_109__cf__109
Constant_float_cuda_Constant_112(0, Constant_112_0);
 // name=cg/resnet_v114/conv47/batchnorm47/gamma/read/_108__cf__108
Constant_float_cuda_Constant_111(0, Constant_111_0);
 // name=cg/resnet_v114/conv47/batchnorm47/beta/read/_107__cf__107
Constant_float_cuda_Constant_110(0, Constant_110_0);
 // name=cg/resnet_v114/conv48/conv2d/kernel/read/_116__cf__116
Constant_float_cuda_Constant_119(0, Constant_119_0);
 // name=cg/resnet_v114/conv48/batchnorm48/moving_variance/read/_115__cf__115
Constant_float_cuda_Constant_118(0, Constant_118_0);
 // name=cg/resnet_v114/conv48/batchnorm48/moving_mean/read/_114__cf__114
Constant_float_cuda_Constant_117(0, Constant_117_0);
 // name=cg/resnet_v114/conv48/batchnorm48/gamma/read/_113__cf__113
Constant_float_cuda_Constant_116(0, Constant_116_0);
 // name=cg/resnet_v114/conv48/batchnorm48/beta/read/_112__cf__112
Constant_float_cuda_Constant_115(0, Constant_115_0);
 // name=cg/resnet_v114/conv49/conv2d/kernel/read/_121__cf__121
Constant_float_cuda_Constant_124(0, Constant_124_0);
 // name=cg/resnet_v114/conv49/batchnorm49/moving_variance/read/_120__cf__120
Constant_float_cuda_Constant_123(0, Constant_123_0);
 // name=cg/resnet_v114/conv49/batchnorm49/moving_mean/read/_119__cf__119
Constant_float_cuda_Constant_122(0, Constant_122_0);
 // name=cg/resnet_v114/conv49/batchnorm49/gamma/read/_118__cf__118
Constant_float_cuda_Constant_121(0, Constant_121_0);
 // name=cg/resnet_v114/conv49/batchnorm49/beta/read/_117__cf__117
Constant_float_cuda_Constant_120(0, Constant_120_0);
 // name=cg/resnet_v115/conv50/conv2d/kernel/read/_126__cf__126
Constant_float_cuda_Constant_129(0, Constant_129_0);
 // name=cg/resnet_v115/conv50/batchnorm50/moving_variance/read/_125__cf__125
Constant_float_cuda_Constant_128(0, Constant_128_0);
 // name=cg/resnet_v115/conv50/batchnorm50/moving_mean/read/_124__cf__124
Constant_float_cuda_Constant_127(0, Constant_127_0);
 // name=cg/resnet_v115/conv50/batchnorm50/gamma/read/_123__cf__123
Constant_float_cuda_Constant_126(0, Constant_126_0);
 // name=cg/resnet_v115/conv50/batchnorm50/beta/read/_122__cf__122
Constant_float_cuda_Constant_125(0, Constant_125_0);
 // name=cg/resnet_v115/conv51/conv2d/kernel/read/_131__cf__131
Constant_float_cuda_Constant_134(0, Constant_134_0);
 // name=cg/resnet_v115/conv51/batchnorm51/moving_variance/read/_130__cf__130
Constant_float_cuda_Constant_133(0, Constant_133_0);
 // name=cg/resnet_v115/conv51/batchnorm51/moving_mean/read/_129__cf__129
Constant_float_cuda_Constant_132(0, Constant_132_0);
 // name=cg/resnet_v115/conv51/batchnorm51/gamma/read/_128__cf__128
Constant_float_cuda_Constant_131(0, Constant_131_0);
 // name=cg/resnet_v115/conv51/batchnorm51/beta/read/_127__cf__127
Constant_float_cuda_Constant_130(0, Constant_130_0);
 // name=cg/resnet_v115/conv52/conv2d/kernel/read/_136__cf__136
Constant_float_cuda_Constant_139(0, Constant_139_0);
 // name=cg/resnet_v115/conv52/batchnorm52/moving_variance/read/_135__cf__135
Constant_float_cuda_Constant_138(0, Constant_138_0);
 // name=cg/resnet_v115/conv52/batchnorm52/moving_mean/read/_134__cf__134
Constant_float_cuda_Constant_137(0, Constant_137_0);
 // name=cg/resnet_v115/conv52/batchnorm52/gamma/read/_133__cf__133
Constant_float_cuda_Constant_136(0, Constant_136_0);
 // name=cg/resnet_v115/conv52/batchnorm52/beta/read/_132__cf__132
Constant_float_cuda_Constant_135(0, Constant_135_0);
 // name=Constant_500
Constant_float_cuda_Constant_500(0, Constant_500_0);
 // name=cg/affine0/weights/read/_1__cf__1
Constant_float_cuda_Constant_4(0, Constant_4_0);
 // name=cg/affine0/biases/read/_0__cf__0
Constant_float_cuda_Constant_3(0, Constant_3_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
}

// Node name:	Result_505
// Description:	Result
// Input:
//	- name: Add_504_0	type: float	shape: Shape{1, 1001}
// Output:
//	- name: Result_505_0	type: float	shape: Shape{1, 1001}
void Result_float_float_cuda_lib_Result_505(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Convolution_282
// Description:	Convolution
// Input:
//	- name: MaxPool_278_0	type: float	shape: Shape{1, 64, 56, 56}
//	- name: Reshape_281_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_282_0	type: float	shape: Shape{1, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_282(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_457
// Description:	Reshape
// Input:
//	- name: Constant_94_0	type: float	shape: Shape{1, 1, 1024, 2048}
// Output:
//	- name: Reshape_457_0	type: float	shape: Shape{2048, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_457(float* input0, float* output0)
{
    uint32_t input_strides0 = 2048;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1024;
    size_t nx = 2048;
    size_t ny = 1024;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_457_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_457<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_460
// Description:	Convolution
// Input:
//	- name: Relu_456_0	type: float	shape: Shape{1, 1024, 14, 14}
//	- name: Reshape_459_0	type: float	shape: Shape{512, 1024, 1, 1}
// Output:
//	- name: Convolution_460_0	type: float	shape: Shape{1, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_460(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_458
// Description:	Convolution
// Input:
//	- name: Relu_456_0	type: float	shape: Shape{1, 1024, 14, 14}
//	- name: Reshape_457_0	type: float	shape: Shape{2048, 1024, 1, 1}
// Output:
//	- name: Convolution_458_0	type: float	shape: Shape{1, 2048, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_458(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_459
// Description:	Reshape
// Input:
//	- name: Constant_99_0	type: float	shape: Shape{1, 1, 1024, 512}
// Output:
//	- name: Reshape_459_0	type: float	shape: Shape{512, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_459(float* input0, float* output0)
{
    uint32_t input_strides0 = 512;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1024;
    size_t nx = 512;
    size_t ny = 1024;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_459_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_459<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_377
// Description:	Convolution
// Input:
//	- name: Relu_375_0	type: float	shape: Shape{1, 512, 28, 28}
//	- name: Reshape_376_0	type: float	shape: Shape{1024, 512, 1, 1}
// Output:
//	- name: Convolution_377_0	type: float	shape: Shape{1, 1024, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_377(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_469
// Description:	Convolution
// Input:
//	- name: Relu_467_0	type: float	shape: Shape{1, 512, 7, 7}
//	- name: Reshape_468_0	type: float	shape: Shape{2048, 512, 1, 1}
// Output:
//	- name: Convolution_469_0	type: float	shape: Shape{1, 2048, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_469(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_290
// Description:	Reshape
// Input:
//	- name: Constant_29_0	type: float	shape: Shape{1, 1, 64, 256}
// Output:
//	- name: Reshape_290_0	type: float	shape: Shape{256, 64, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_290(float* input0, float* output0)
{
    uint32_t input_strides0 = 256;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 64;
    size_t nx = 256;
    size_t ny = 64;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_290_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_290<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_322
// Description:	Convolution
// Input:
//	- name: Relu_320_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: Reshape_321_0	type: float	shape: Shape{512, 256, 1, 1}
// Output:
//	- name: Convolution_322_0	type: float	shape: Shape{1, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_322(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_281
// Description:	Reshape
// Input:
//	- name: Constant_19_0	type: float	shape: Shape{1, 1, 64, 64}
// Output:
//	- name: Reshape_281_0	type: float	shape: Shape{64, 64, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_281(float* input0, float* output0)
{
    uint32_t input_strides0 = 64;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 64;
    size_t nx = 64;
    size_t ny = 64;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_281_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_281<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_292
// Description:	BatchNormInference
// Input:
//	- name: Constant_26_0	type: float	shape: Shape{256}
//	- name: Constant_25_0	type: float	shape: Shape{256}
//	- name: Convolution_291_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: Constant_27_0	type: float	shape: Shape{256}
//	- name: Constant_28_0	type: float	shape: Shape{256}
// Output:
//	- name: BatchNormInference_292_0	type: float	shape: Shape{1, 256, 56, 56}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 56 * 56;
    const int c_id = blockIdx.x % 256;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 56 * 56; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_323
// Description:	Reshape
// Input:
//	- name: Constant_164_0	type: float	shape: Shape{1, 1, 256, 128}
// Output:
//	- name: Reshape_323_0	type: float	shape: Shape{128, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_323(float* input0, float* output0)
{
    uint32_t input_strides0 = 128;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 256;
    size_t nx = 128;
    size_t ny = 256;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_323_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_323<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_502
// Description:	Dot
// Input:
//	- name: Divide_501_0	type: float	shape: Shape{1, 2048}
//	- name: Constant_4_0	type: float	shape: Shape{2048, 1001}
// Output:
//	- name: Dot_502_0	type: float	shape: Shape{1, 1001}
void Dot_float_float_float_cuda_lib_Dot_502(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 1, 2048, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 2048, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Convolution_388
// Description:	Convolution
// Input:
//	- name: Relu_386_0	type: float	shape: Shape{1, 256, 14, 14}
//	- name: Reshape_387_0	type: float	shape: Shape{1024, 256, 1, 1}
// Output:
//	- name: Convolution_388_0	type: float	shape: Shape{1, 1024, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_388(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_326
// Description:	BatchNormInference
// Input:
//	- name: Constant_161_0	type: float	shape: Shape{128}
//	- name: Constant_160_0	type: float	shape: Shape{128}
//	- name: Convolution_324_0	type: float	shape: Shape{1, 128, 28, 28}
//	- name: Constant_162_0	type: float	shape: Shape{128}
//	- name: Constant_163_0	type: float	shape: Shape{128}
// Output:
//	- name: BatchNormInference_326_0	type: float	shape: Shape{1, 128, 28, 28}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 28 * 28;
    const int c_id = blockIdx.x % 128;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 28 * 28; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_333
// Description:	Convolution
// Input:
//	- name: Relu_331_0	type: float	shape: Shape{1, 128, 28, 28}
//	- name: Reshape_332_0	type: float	shape: Shape{512, 128, 1, 1}
// Output:
//	- name: Convolution_333_0	type: float	shape: Shape{1, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_333(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 128, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	MaxPool_278
// Description:	MaxPool
// Input:
//	- name: Relu_277_0	type: float	shape: Shape{1, 64, 112, 112}
// Output:
//	- name: MaxPool_278_0	type: float	shape: Shape{1, 64, 56, 56}
void MaxPool_float_float_cuda_lib_MaxPool_278(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(static bool selected_algo = true(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(static bool selected_algo = true(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Reshape_464
// Description:	Reshape
// Input:
//	- name: Constant_104_0	type: float	shape: Shape{3, 3, 512, 512}
// Output:
//	- name: Reshape_464_0	type: float	shape: Shape{512, 512, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_464(float* input0, float* output0)
{
    uint32_t input_strides0 = 262144;
    uint32_t input_strides1 = 512;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 4608;
    size_t nx = 512;
    size_t ny = 512;
    size_t nz = 9;
    __shared__ float tile[16][1][17];
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
    otid2 = tid0;
    otid0 = tid2;
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
extern void Reshape_float_float_cuda_Reshape_464_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_464<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_462
// Description:	BatchNormInference
// Input:
//	- name: Constant_96_0	type: float	shape: Shape{512}
//	- name: Constant_95_0	type: float	shape: Shape{512}
//	- name: Convolution_460_0	type: float	shape: Shape{1, 512, 7, 7}
//	- name: Constant_97_0	type: float	shape: Shape{512}
//	- name: Constant_98_0	type: float	shape: Shape{512}
// Output:
//	- name: BatchNormInference_462_0	type: float	shape: Shape{1, 512, 7, 7}
extern "C" __launch_bounds__(49) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 7 * 7;
    const int c_id = blockIdx.x % 512;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 7 * 7; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_321
// Description:	Reshape
// Input:
//	- name: Constant_159_0	type: float	shape: Shape{1, 1, 256, 512}
// Output:
//	- name: Reshape_321_0	type: float	shape: Shape{512, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_321(float* input0, float* output0)
{
    uint32_t input_strides0 = 512;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 256;
    size_t nx = 512;
    size_t ny = 256;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_321_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_321<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_474
// Description:	Convolution
// Input:
//	- name: Relu_472_0	type: float	shape: Shape{1, 2048, 7, 7}
//	- name: Reshape_473_0	type: float	shape: Shape{512, 2048, 1, 1}
// Output:
//	- name: Convolution_474_0	type: float	shape: Shape{1, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_474(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 2048, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_337
// Description:	Reshape
// Input:
//	- name: Constant_179_0	type: float	shape: Shape{1, 1, 512, 128}
// Output:
//	- name: Reshape_337_0	type: float	shape: Shape{128, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_337(float* input0, float* output0)
{
    uint32_t input_strides0 = 128;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 512;
    size_t nx = 128;
    size_t ny = 512;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_337_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_337<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_338
// Description:	Convolution
// Input:
//	- name: Relu_336_0	type: float	shape: Shape{1, 512, 28, 28}
//	- name: Reshape_337_0	type: float	shape: Shape{128, 512, 1, 1}
// Output:
//	- name: Convolution_338_0	type: float	shape: Shape{1, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_338(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_379
// Description:	Convolution
// Input:
//	- name: Relu_375_0	type: float	shape: Shape{1, 512, 28, 28}
//	- name: Reshape_378_0	type: float	shape: Shape{256, 512, 1, 1}
// Output:
//	- name: Convolution_379_0	type: float	shape: Shape{1, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_379(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_393
// Description:	Convolution
// Input:
//	- name: Relu_391_0	type: float	shape: Shape{1, 1024, 14, 14}
//	- name: Reshape_392_0	type: float	shape: Shape{256, 1024, 1, 1}
// Output:
//	- name: Convolution_393_0	type: float	shape: Shape{1, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_393(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_381
// Description:	BatchNormInference
// Input:
//	- name: Constant_226_0	type: float	shape: Shape{256}
//	- name: Constant_225_0	type: float	shape: Shape{256}
//	- name: Convolution_379_0	type: float	shape: Shape{1, 256, 14, 14}
//	- name: Constant_227_0	type: float	shape: Shape{256}
//	- name: Constant_228_0	type: float	shape: Shape{256}
// Output:
//	- name: BatchNormInference_381_0	type: float	shape: Shape{1, 256, 14, 14}
extern "C" __launch_bounds__(196) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 14 * 14;
    const int c_id = blockIdx.x % 256;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 14 * 14; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_383
// Description:	Reshape
// Input:
//	- name: Constant_234_0	type: float	shape: Shape{3, 3, 256, 256}
// Output:
//	- name: Reshape_383_0	type: float	shape: Shape{256, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_383(float* input0, float* output0)
{
    uint32_t input_strides0 = 65536;
    uint32_t input_strides1 = 256;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 2304;
    size_t nx = 256;
    size_t ny = 256;
    size_t nz = 9;
    __shared__ float tile[16][1][17];
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
    otid2 = tid0;
    otid0 = tid2;
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
extern void Reshape_float_float_cuda_Reshape_383_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_383<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Add_504
// Description:	Add
// Input:
//	- name: Dot_502_0	type: float	shape: Shape{1, 1001}
//	- name: Broadcast_503_0	type: float	shape: Shape{1, 1001}
// Output:
//	- name: Add_504_0	type: float	shape: Shape{1, 1001}
extern "C" __launch_bounds__(143) __global__ void Add_float_float_float_cuda_Add_504(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 143 + threadIdx.x] = add(input0[blockIdx.x * 143 + threadIdx.x], input1[blockIdx.x * 143 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_504_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_504<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_384
// Description:	Convolution
// Input:
//	- name: Relu_382_0	type: float	shape: Shape{1, 256, 14, 14}
//	- name: Reshape_383_0	type: float	shape: Shape{256, 256, 3, 3}
// Output:
//	- name: Convolution_384_0	type: float	shape: Shape{1, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_384(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 256, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_376
// Description:	Reshape
// Input:
//	- name: Constant_224_0	type: float	shape: Shape{1, 1, 512, 1024}
// Output:
//	- name: Reshape_376_0	type: float	shape: Shape{1024, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_376(float* input0, float* output0)
{
    uint32_t input_strides0 = 1024;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 512;
    size_t nx = 1024;
    size_t ny = 512;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_376_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_376<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_295
// Description:	Reshape
// Input:
//	- name: Constant_34_0	type: float	shape: Shape{1, 1, 256, 64}
// Output:
//	- name: Reshape_295_0	type: float	shape: Shape{64, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_295(float* input0, float* output0)
{
    uint32_t input_strides0 = 64;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 256;
    size_t nx = 64;
    size_t ny = 256;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = input0[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output0[output_idx] = tile[tid1][tid0];
    }

}
extern void Reshape_float_float_cuda_Reshape_295_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_295<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_470
// Description:	BatchNormInference
// Input:
//	- name: Constant_106_0	type: float	shape: Shape{2048}
//	- name: Constant_105_0	type: float	shape: Shape{2048}
//	- name: Convolution_469_0	type: float	shape: Shape{1, 2048, 7, 7}
//	- name: Constant_107_0	type: float	shape: Shape{2048}
//	- name: Constant_108_0	type: float	shape: Shape{2048}
// Output:
//	- name: BatchNormInference_470_0	type: float	shape: Shape{1, 2048, 7, 7}
extern "C" __launch_bounds__(49) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 7 * 7;
    const int c_id = blockIdx.x % 2048;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 7 * 7; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {1, 224, 224, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {1, 1001}
#endif




extern "C" void resnet50_cuda_free()
{

CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));
}

#include "./include/dnn.h"


class Reshape_float_float_cuda_Reshape_271_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_271_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_271_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_271_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_271<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_271_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Pad_float_float_float_cuda_Pad_273_CallKernel : public Kernel {
public:
    Pad_float_float_float_cuda_Pad_273_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Pad_float_float_float_cuda_Pad_273_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Pad_float_float_float_cuda_Pad_273_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Pad_float_float_float_cuda_Pad_273<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        Pad_float_float_float_cuda_Pad_273_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_274_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_274_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_274_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_274_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_274<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_274_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_275Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_275Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_275";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 3, 230, 230, 64, 3, 7, 7, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_275(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 230, 230));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 7, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_275(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Relu_float_float_cuda_Relu_277_CallKernel : public Kernel {
public:
    Relu_float_float_cuda_Relu_277_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Relu_float_float_cuda_Relu_277_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Relu_float_float_cuda_Relu_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Relu_float_float_cuda_Relu_277<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Relu_float_float_cuda_Relu_277_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_278Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_278Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_278";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_278(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(static bool selected_algo = true(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(static bool selected_algo = true(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        MaxPool_float_float_cuda_lib_MaxPool_278(cudnn_handle, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_281_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_281_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_281_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_281_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_281<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_281_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_282Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_282Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_282";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 64, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_282(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_282(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_286_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_286_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_286_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_286_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_286<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_286_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_287Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_287Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_287";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_287(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_287(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_290_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_290_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_290_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_290_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_290<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_290_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_291Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_291Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_291";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 64, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_291(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_291(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel : public Kernel {
public:
    FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "FusedKernel_float_float_float_cuda_Add_Relu_0_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Add_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Add_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        FusedKernel_float_float_float_cuda_Add_Relu_0_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_295_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_295_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_295_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_295_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_295<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_295_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_296Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_296Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_296";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_296(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_296(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_323_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_323_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_323_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_323_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_323<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_323_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_324Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_324Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_324";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 56, 56, 128, 256, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_324(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_324(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_328_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_328_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_328_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_328_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_328<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_328_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_329Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_329Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_329";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_329(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_329(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_332_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_332_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_332_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_332_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_332<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_332_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_333Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_333Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_333";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 128, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_333(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 128, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_333(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_321_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_321_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_321_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_321_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_321<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_321_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_322Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_322Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_322";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 56, 56, 512, 256, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_322(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_322(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_337_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_337_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_337_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_337_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_337<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_337_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_338Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_338Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_338";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_338(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_338(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_378_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_378_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_378_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_378_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_378<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_378_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_379Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_379Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_379";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 28, 28, 256, 512, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_379(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_379(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_383_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_383_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_383_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_383_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_383<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_383_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_384Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_384Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_384";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_384(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 256, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_384(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_387_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_387_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_387_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_387_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_387<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_387_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_388Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_388Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_388";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_388(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_388(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_376_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_376_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_376_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_376_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_376<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_376_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_377Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_377Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_377";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 28, 28, 1024, 512, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_377(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_377(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_392_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_392_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_392_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_392<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_392_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_393Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_393Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_393";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 1024, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_393(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_393(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_459_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_459_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_459_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_459_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_459<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_459_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_460Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_460Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_460";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 1024, 14, 14, 512, 1024, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_460(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_460(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_464_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_464_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_464_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_464_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_464<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_464_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_465Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_465Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_465";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_465(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 512, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_465(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_468_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_468_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_468_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_468_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_468<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_468_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_469Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_469Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_469";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_469(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_469(cudnn_handle, input0, input1, output0);
    }
};


class BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel : public Kernel {
public:
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class Reshape_float_float_cuda_Reshape_457_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_457_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_457_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_457_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_457<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_457_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_458Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_458Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_458";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 1024, 14, 14, 2048, 1024, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_458(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_458(cudnn_handle, input0, input1, output0);
    }
};


class Reshape_float_float_cuda_Reshape_473_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_473_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_473_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_473_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_473<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_473_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_474Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_474Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_474";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 2048, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_474(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2048, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(static bool selected_algo = true(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 2048, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    CUDNN_SAFE_CALL(mycudnnConvolutionForward(cudnn_handle, &alpha, tensor_desc_0, input0,filter_desc, input1, conv_desc, conv_fwd_algo, workspace_ptr_0, workspace_size_in_bytes, &beta, tensor_desc_1, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

}

    void executeImpl(cudaStream_t stream) {
        Convolution_float_float_float_cuda_lib_Convolution_474(cudnn_handle, input0, input1, output0);
    }
};


class Sum_float_float_cuda_Sum_499_CallKernel : public Kernel {
public:
    Sum_float_float_cuda_Sum_499_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Sum_float_float_cuda_Sum_499_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Sum_float_float_cuda_Sum_499_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Sum_float_float_cuda_Sum_499<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Sum_float_float_cuda_Sum_499_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Divide_float_float_float_cuda_Divide_501_CallKernel : public Kernel {
public:
    Divide_float_float_float_cuda_Divide_501_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Divide_float_float_float_cuda_Divide_501_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Divide_float_float_float_cuda_Divide_501_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Divide_float_float_float_cuda_Divide_501<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        Divide_float_float_float_cuda_Divide_501_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class Dot_float_float_float_cuda_lib_Dot_502Kernel : public Kernel {
public:
    Dot_float_float_float_cuda_lib_Dot_502Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Dot_float_float_float_cuda_lib_Dot_502";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1001;
    ret[1] = 1;
    ret[2] = 2048;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_502(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 1, 2048, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 2048, &beta, static_cast<float*>(output0), 1001));

}

    void executeImpl(cudaStream_t stream) {
        Dot_float_float_float_cuda_lib_Dot_502(cublas_handle, input0, input1, output0);
    }
};


class Add_float_float_float_cuda_Add_504_CallKernel : public Kernel {
public:
    Add_float_float_float_cuda_Add_504_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Add_float_float_float_cuda_Add_504_Call";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_504_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_504<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        Add_float_float_float_cuda_Add_504_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class Result_float_float_cuda_lib_Result_505Kernel : public Kernel {
public:
    Result_float_float_cuda_lib_Result_505Kernel(float*  input0, float**  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "Result_float_float_cuda_lib_Result_505";
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
    float*  Parameter_270_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_505(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        Result_float_float_cuda_lib_Result_505(input0, output0);
    }
};
void Resnet50::gen_vector(float*  Parameter_270_0, float**  Result_505_0) {
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_271_CallKernel(dim3(1, 3136, 1), dim3(16, 16, 1), 0, nullptr, std::move(Parameter_270_0), std::move(Reshape_271_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Pad_float_float_float_cuda_Pad_273_CallKernel(dim3(2480, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Reshape_271_0), std::move(Constant_272_0), std::move(Pad_273_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_274_CallKernel(dim3(4, 3, 4), dim3(16, 1, 16), 0, nullptr, std::move(Constant_9_0), std::move(Reshape_274_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_275Kernel(std::move(cudnn_handle_0), std::move(Pad_273_0), std::move(Reshape_274_0), std::move(Convolution_275_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_6_0), std::move(Constant_5_0), std::move(Convolution_275_0), std::move(Constant_7_0), std::move(Constant_8_0), std::move(BatchNormInference_276_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_276_0), std::move(Relu_277_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_278Kernel(std::move(cudnn_handle_0), std::move(Relu_277_0), std::move(MaxPool_278_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_281_CallKernel(dim3(4, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_19_0), std::move(Reshape_281_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_282Kernel(std::move(cudnn_handle_0), std::move(MaxPool_278_0), std::move(Reshape_281_0), std::move(Convolution_282_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_16_0), std::move(Constant_15_0), std::move(Convolution_282_0), std::move(Constant_17_0), std::move(Constant_18_0), std::move(BatchNormInference_284_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_284_0), std::move(Relu_285_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_24_0), std::move(Reshape_286_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(cudnn_handle_0), std::move(Relu_285_0), std::move(Reshape_286_0), std::move(Convolution_287_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_21_0), std::move(Constant_20_0), std::move(Convolution_287_0), std::move(Constant_22_0), std::move(Constant_23_0), std::move(BatchNormInference_288_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_288_0), std::move(Relu_289_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_29_0), std::move(Reshape_290_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(cudnn_handle_0), std::move(Relu_289_0), std::move(Reshape_290_0), std::move(Convolution_291_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_26_0), std::move(Constant_25_0), std::move(Convolution_291_0), std::move(Constant_27_0), std::move(Constant_28_0), std::move(BatchNormInference_292_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_14_0), std::move(Reshape_279_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(cudnn_handle_0), std::move(MaxPool_278_0), std::move(Reshape_279_0), std::move(Convolution_280_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_11_0), std::move(Constant_10_0), std::move(Convolution_280_0), std::move(Constant_12_0), std::move(Constant_13_0), std::move(BatchNormInference_283_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_283_0), std::move(BatchNormInference_292_0), std::move(Relu_294_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_295_CallKernel(dim3(4, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_34_0), std::move(Reshape_295_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_296Kernel(std::move(cudnn_handle_0), std::move(Relu_294_0), std::move(Reshape_295_0), std::move(Convolution_296_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_31_0), std::move(Constant_30_0), std::move(Convolution_296_0), std::move(Constant_32_0), std::move(Constant_33_0), std::move(BatchNormInference_297_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_297_0), std::move(Relu_298_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_39_0), std::move(Reshape_299_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(cudnn_handle_0), std::move(Relu_298_0), std::move(Reshape_299_0), std::move(Convolution_300_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_36_0), std::move(Constant_35_0), std::move(Convolution_300_0), std::move(Constant_37_0), std::move(Constant_38_0), std::move(BatchNormInference_301_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_301_0), std::move(Relu_302_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_44_0), std::move(Reshape_303_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(cudnn_handle_0), std::move(Relu_302_0), std::move(Reshape_303_0), std::move(Convolution_304_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_41_0), std::move(Constant_40_0), std::move(Convolution_304_0), std::move(Constant_42_0), std::move(Constant_43_0), std::move(BatchNormInference_305_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_294_0), std::move(BatchNormInference_305_0), std::move(Relu_307_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_295_CallKernel(dim3(4, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_149_0), std::move(Reshape_308_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_296Kernel(std::move(cudnn_handle_0), std::move(Relu_307_0), std::move(Reshape_308_0), std::move(Convolution_309_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_146_0), std::move(Constant_145_0), std::move(Convolution_309_0), std::move(Constant_147_0), std::move(Constant_148_0), std::move(BatchNormInference_310_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_310_0), std::move(Relu_311_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_154_0), std::move(Reshape_312_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(cudnn_handle_0), std::move(Relu_311_0), std::move(Reshape_312_0), std::move(Convolution_313_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(64, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_151_0), std::move(Constant_150_0), std::move(Convolution_313_0), std::move(Constant_152_0), std::move(Constant_153_0), std::move(BatchNormInference_314_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_314_0), std::move(Relu_315_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_144_0), std::move(Reshape_316_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(cudnn_handle_0), std::move(Relu_315_0), std::move(Reshape_316_0), std::move(Convolution_317_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_141_0), std::move(Constant_140_0), std::move(Convolution_317_0), std::move(Constant_142_0), std::move(Constant_143_0), std::move(BatchNormInference_318_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_307_0), std::move(BatchNormInference_318_0), std::move(Relu_320_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_323_CallKernel(dim3(8, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_164_0), std::move(Reshape_323_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_324Kernel(std::move(cudnn_handle_0), std::move(Relu_320_0), std::move(Reshape_323_0), std::move(Convolution_324_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_161_0), std::move(Constant_160_0), std::move(Convolution_324_0), std::move(Constant_162_0), std::move(Constant_163_0), std::move(BatchNormInference_326_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_326_0), std::move(Relu_327_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_169_0), std::move(Reshape_328_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(cudnn_handle_0), std::move(Relu_327_0), std::move(Reshape_328_0), std::move(Convolution_329_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_166_0), std::move(Constant_165_0), std::move(Convolution_329_0), std::move(Constant_167_0), std::move(Constant_168_0), std::move(BatchNormInference_330_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_330_0), std::move(Relu_331_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_174_0), std::move(Reshape_332_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(cudnn_handle_0), std::move(Relu_331_0), std::move(Reshape_332_0), std::move(Convolution_333_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_171_0), std::move(Constant_170_0), std::move(Convolution_333_0), std::move(Constant_172_0), std::move(Constant_173_0), std::move(BatchNormInference_334_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_321_CallKernel(dim3(32, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_159_0), std::move(Reshape_321_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_322Kernel(std::move(cudnn_handle_0), std::move(Relu_320_0), std::move(Reshape_321_0), std::move(Convolution_322_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_156_0), std::move(Constant_155_0), std::move(Convolution_322_0), std::move(Constant_157_0), std::move(Constant_158_0), std::move(BatchNormInference_325_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_325_0), std::move(BatchNormInference_334_0), std::move(Relu_336_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_179_0), std::move(Reshape_337_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(cudnn_handle_0), std::move(Relu_336_0), std::move(Reshape_337_0), std::move(Convolution_338_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_176_0), std::move(Constant_175_0), std::move(Convolution_338_0), std::move(Constant_177_0), std::move(Constant_178_0), std::move(BatchNormInference_339_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_339_0), std::move(Relu_340_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_184_0), std::move(Reshape_341_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(cudnn_handle_0), std::move(Relu_340_0), std::move(Reshape_341_0), std::move(Convolution_342_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_181_0), std::move(Constant_180_0), std::move(Convolution_342_0), std::move(Constant_182_0), std::move(Constant_183_0), std::move(BatchNormInference_343_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_343_0), std::move(Relu_344_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_189_0), std::move(Reshape_345_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(cudnn_handle_0), std::move(Relu_344_0), std::move(Reshape_345_0), std::move(Convolution_346_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_186_0), std::move(Constant_185_0), std::move(Convolution_346_0), std::move(Constant_187_0), std::move(Constant_188_0), std::move(BatchNormInference_347_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_336_0), std::move(BatchNormInference_347_0), std::move(Relu_349_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_194_0), std::move(Reshape_350_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(cudnn_handle_0), std::move(Relu_349_0), std::move(Reshape_350_0), std::move(Convolution_351_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_191_0), std::move(Constant_190_0), std::move(Convolution_351_0), std::move(Constant_192_0), std::move(Constant_193_0), std::move(BatchNormInference_352_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_352_0), std::move(Relu_353_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_199_0), std::move(Reshape_354_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(cudnn_handle_0), std::move(Relu_353_0), std::move(Reshape_354_0), std::move(Convolution_355_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_196_0), std::move(Constant_195_0), std::move(Convolution_355_0), std::move(Constant_197_0), std::move(Constant_198_0), std::move(BatchNormInference_356_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_356_0), std::move(Relu_357_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_204_0), std::move(Reshape_358_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(cudnn_handle_0), std::move(Relu_357_0), std::move(Reshape_358_0), std::move(Convolution_359_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_201_0), std::move(Constant_200_0), std::move(Convolution_359_0), std::move(Constant_202_0), std::move(Constant_203_0), std::move(BatchNormInference_360_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_349_0), std::move(BatchNormInference_360_0), std::move(Relu_362_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_209_0), std::move(Reshape_363_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(cudnn_handle_0), std::move(Relu_362_0), std::move(Reshape_363_0), std::move(Convolution_364_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_206_0), std::move(Constant_205_0), std::move(Convolution_364_0), std::move(Constant_207_0), std::move(Constant_208_0), std::move(BatchNormInference_365_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_365_0), std::move(Relu_366_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_214_0), std::move(Reshape_367_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(cudnn_handle_0), std::move(Relu_366_0), std::move(Reshape_367_0), std::move(Convolution_368_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(128, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_211_0), std::move(Constant_210_0), std::move(Convolution_368_0), std::move(Constant_212_0), std::move(Constant_213_0), std::move(BatchNormInference_369_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_369_0), std::move(Relu_370_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_219_0), std::move(Reshape_371_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(cudnn_handle_0), std::move(Relu_370_0), std::move(Reshape_371_0), std::move(Convolution_372_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(512, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Constant_216_0), std::move(Constant_215_0), std::move(Convolution_372_0), std::move(Constant_217_0), std::move(Constant_218_0), std::move(BatchNormInference_373_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_362_0), std::move(BatchNormInference_373_0), std::move(Relu_375_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_378_CallKernel(dim3(16, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_229_0), std::move(Reshape_378_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_379Kernel(std::move(cudnn_handle_0), std::move(Relu_375_0), std::move(Reshape_378_0), std::move(Convolution_379_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_226_0), std::move(Constant_225_0), std::move(Convolution_379_0), std::move(Constant_227_0), std::move(Constant_228_0), std::move(BatchNormInference_381_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_381_0), std::move(Relu_382_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_234_0), std::move(Reshape_383_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_382_0), std::move(Reshape_383_0), std::move(Convolution_384_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_231_0), std::move(Constant_230_0), std::move(Convolution_384_0), std::move(Constant_232_0), std::move(Constant_233_0), std::move(BatchNormInference_385_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_385_0), std::move(Relu_386_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_239_0), std::move(Reshape_387_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_386_0), std::move(Reshape_387_0), std::move(Convolution_388_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_236_0), std::move(Constant_235_0), std::move(Convolution_388_0), std::move(Constant_237_0), std::move(Constant_238_0), std::move(BatchNormInference_389_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_376_CallKernel(dim3(64, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_224_0), std::move(Reshape_376_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_377Kernel(std::move(cudnn_handle_0), std::move(Relu_375_0), std::move(Reshape_376_0), std::move(Convolution_377_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_221_0), std::move(Constant_220_0), std::move(Convolution_377_0), std::move(Constant_222_0), std::move(Constant_223_0), std::move(BatchNormInference_380_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_380_0), std::move(BatchNormInference_389_0), std::move(Relu_391_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_244_0), std::move(Reshape_392_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(cudnn_handle_0), std::move(Relu_391_0), std::move(Reshape_392_0), std::move(Convolution_393_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_241_0), std::move(Constant_240_0), std::move(Convolution_393_0), std::move(Constant_242_0), std::move(Constant_243_0), std::move(BatchNormInference_394_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_394_0), std::move(Relu_395_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_249_0), std::move(Reshape_396_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_395_0), std::move(Reshape_396_0), std::move(Convolution_397_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_246_0), std::move(Constant_245_0), std::move(Convolution_397_0), std::move(Constant_247_0), std::move(Constant_248_0), std::move(BatchNormInference_398_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_398_0), std::move(Relu_399_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_254_0), std::move(Reshape_400_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_399_0), std::move(Reshape_400_0), std::move(Convolution_401_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_251_0), std::move(Constant_250_0), std::move(Convolution_401_0), std::move(Constant_252_0), std::move(Constant_253_0), std::move(BatchNormInference_402_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_391_0), std::move(BatchNormInference_402_0), std::move(Relu_404_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_259_0), std::move(Reshape_405_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(cudnn_handle_0), std::move(Relu_404_0), std::move(Reshape_405_0), std::move(Convolution_406_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_256_0), std::move(Constant_255_0), std::move(Convolution_406_0), std::move(Constant_257_0), std::move(Constant_258_0), std::move(BatchNormInference_407_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_407_0), std::move(Relu_408_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_264_0), std::move(Reshape_409_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_408_0), std::move(Reshape_409_0), std::move(Convolution_410_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_261_0), std::move(Constant_260_0), std::move(Convolution_410_0), std::move(Constant_262_0), std::move(Constant_263_0), std::move(BatchNormInference_411_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_411_0), std::move(Relu_412_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_269_0), std::move(Reshape_413_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_412_0), std::move(Reshape_413_0), std::move(Convolution_414_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_266_0), std::move(Constant_265_0), std::move(Convolution_414_0), std::move(Constant_267_0), std::move(Constant_268_0), std::move(BatchNormInference_415_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_404_0), std::move(BatchNormInference_415_0), std::move(Relu_417_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_49_0), std::move(Reshape_418_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(cudnn_handle_0), std::move(Relu_417_0), std::move(Reshape_418_0), std::move(Convolution_419_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_46_0), std::move(Constant_45_0), std::move(Convolution_419_0), std::move(Constant_47_0), std::move(Constant_48_0), std::move(BatchNormInference_420_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_420_0), std::move(Relu_421_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_54_0), std::move(Reshape_422_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_421_0), std::move(Reshape_422_0), std::move(Convolution_423_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_51_0), std::move(Constant_50_0), std::move(Convolution_423_0), std::move(Constant_52_0), std::move(Constant_53_0), std::move(BatchNormInference_424_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_424_0), std::move(Relu_425_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_59_0), std::move(Reshape_426_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_425_0), std::move(Reshape_426_0), std::move(Convolution_427_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_56_0), std::move(Constant_55_0), std::move(Convolution_427_0), std::move(Constant_57_0), std::move(Constant_58_0), std::move(BatchNormInference_428_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_417_0), std::move(BatchNormInference_428_0), std::move(Relu_430_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_64_0), std::move(Reshape_431_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(cudnn_handle_0), std::move(Relu_430_0), std::move(Reshape_431_0), std::move(Convolution_432_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_61_0), std::move(Constant_60_0), std::move(Convolution_432_0), std::move(Constant_62_0), std::move(Constant_63_0), std::move(BatchNormInference_433_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_433_0), std::move(Relu_434_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_69_0), std::move(Reshape_435_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_434_0), std::move(Reshape_435_0), std::move(Convolution_436_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_66_0), std::move(Constant_65_0), std::move(Convolution_436_0), std::move(Constant_67_0), std::move(Constant_68_0), std::move(BatchNormInference_437_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_437_0), std::move(Relu_438_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_74_0), std::move(Reshape_439_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_438_0), std::move(Reshape_439_0), std::move(Convolution_440_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_71_0), std::move(Constant_70_0), std::move(Convolution_440_0), std::move(Constant_72_0), std::move(Constant_73_0), std::move(BatchNormInference_441_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_430_0), std::move(BatchNormInference_441_0), std::move(Relu_443_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_79_0), std::move(Reshape_444_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(cudnn_handle_0), std::move(Relu_443_0), std::move(Reshape_444_0), std::move(Convolution_445_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_76_0), std::move(Constant_75_0), std::move(Convolution_445_0), std::move(Constant_77_0), std::move(Constant_78_0), std::move(BatchNormInference_446_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_446_0), std::move(Relu_447_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_84_0), std::move(Reshape_448_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(cudnn_handle_0), std::move(Relu_447_0), std::move(Reshape_448_0), std::move(Convolution_449_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(256, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_81_0), std::move(Constant_80_0), std::move(Convolution_449_0), std::move(Constant_82_0), std::move(Constant_83_0), std::move(BatchNormInference_450_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(98, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_450_0), std::move(Relu_451_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_89_0), std::move(Reshape_452_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(cudnn_handle_0), std::move(Relu_451_0), std::move(Reshape_452_0), std::move(Convolution_453_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(1024, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(Constant_86_0), std::move(Constant_85_0), std::move(Convolution_453_0), std::move(Constant_87_0), std::move(Constant_88_0), std::move(BatchNormInference_454_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(392, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_443_0), std::move(BatchNormInference_454_0), std::move(Relu_456_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_459_CallKernel(dim3(32, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_99_0), std::move(Reshape_459_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_460Kernel(std::move(cudnn_handle_0), std::move(Relu_456_0), std::move(Reshape_459_0), std::move(Convolution_460_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_96_0), std::move(Constant_95_0), std::move(Convolution_460_0), std::move(Constant_97_0), std::move(Constant_98_0), std::move(BatchNormInference_462_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_462_0), std::move(Relu_463_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_104_0), std::move(Reshape_464_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(cudnn_handle_0), std::move(Relu_463_0), std::move(Reshape_464_0), std::move(Convolution_465_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_101_0), std::move(Constant_100_0), std::move(Convolution_465_0), std::move(Constant_102_0), std::move(Constant_103_0), std::move(BatchNormInference_466_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_466_0), std::move(Relu_467_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_109_0), std::move(Reshape_468_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(cudnn_handle_0), std::move(Relu_467_0), std::move(Reshape_468_0), std::move(Convolution_469_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(2048, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_106_0), std::move(Constant_105_0), std::move(Convolution_469_0), std::move(Constant_107_0), std::move(Constant_108_0), std::move(BatchNormInference_470_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_457_CallKernel(dim3(128, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_94_0), std::move(Reshape_457_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_458Kernel(std::move(cudnn_handle_0), std::move(Relu_456_0), std::move(Reshape_457_0), std::move(Convolution_458_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(2048, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_91_0), std::move(Constant_90_0), std::move(Convolution_458_0), std::move(Constant_92_0), std::move(Constant_93_0), std::move(BatchNormInference_461_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_461_0), std::move(BatchNormInference_470_0), std::move(Relu_472_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_473_CallKernel(dim3(32, 128, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_114_0), std::move(Reshape_473_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_474Kernel(std::move(cudnn_handle_0), std::move(Relu_472_0), std::move(Reshape_473_0), std::move(Convolution_474_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_111_0), std::move(Constant_110_0), std::move(Convolution_474_0), std::move(Constant_112_0), std::move(Constant_113_0), std::move(BatchNormInference_475_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_475_0), std::move(Relu_476_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_119_0), std::move(Reshape_477_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(cudnn_handle_0), std::move(Relu_476_0), std::move(Reshape_477_0), std::move(Convolution_478_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_116_0), std::move(Constant_115_0), std::move(Convolution_478_0), std::move(Constant_117_0), std::move(Constant_118_0), std::move(BatchNormInference_479_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_479_0), std::move(Relu_480_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_124_0), std::move(Reshape_481_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(cudnn_handle_0), std::move(Relu_480_0), std::move(Reshape_481_0), std::move(Convolution_482_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(2048, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_121_0), std::move(Constant_120_0), std::move(Convolution_482_0), std::move(Constant_122_0), std::move(Constant_123_0), std::move(BatchNormInference_483_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_472_0), std::move(BatchNormInference_483_0), std::move(Relu_485_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_473_CallKernel(dim3(32, 128, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_129_0), std::move(Reshape_486_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_474Kernel(std::move(cudnn_handle_0), std::move(Relu_485_0), std::move(Reshape_486_0), std::move(Convolution_487_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_126_0), std::move(Constant_125_0), std::move(Convolution_487_0), std::move(Constant_127_0), std::move(Constant_128_0), std::move(BatchNormInference_488_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_488_0), std::move(Relu_489_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_134_0), std::move(Reshape_490_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(cudnn_handle_0), std::move(Relu_489_0), std::move(Reshape_490_0), std::move(Convolution_491_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(512, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_131_0), std::move(Constant_130_0), std::move(Convolution_491_0), std::move(Constant_132_0), std::move(Constant_133_0), std::move(BatchNormInference_492_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Relu_float_float_cuda_Relu_277_CallKernel(dim3(49, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(BatchNormInference_492_0), std::move(Relu_493_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(Constant_139_0), std::move(Reshape_494_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(cudnn_handle_0), std::move(Relu_493_0), std::move(Reshape_494_0), std::move(Convolution_495_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(2048, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(Constant_136_0), std::move(Constant_135_0), std::move(Convolution_495_0), std::move(Constant_137_0), std::move(Constant_138_0), std::move(BatchNormInference_496_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Relu_485_0), std::move(BatchNormInference_496_0), std::move(Relu_498_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Sum_float_float_cuda_Sum_499_CallKernel(dim3(2048, 1, 1), dim3(32, 1, 1), 0, nullptr, std::move(Relu_498_0), std::move(Sum_499_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Divide_float_float_float_cuda_Divide_501_CallKernel(dim3(4, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Sum_499_0), std::move(Constant_500_0), std::move(Divide_501_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Dot_float_float_float_cuda_lib_Dot_502Kernel(std::move(cublas_handle_0), std::move(Divide_501_0), std::move(Constant_4_0), std::move(Dot_502_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Add_float_float_float_cuda_Add_504_CallKernel(dim3(7, 1, 1), dim3(143, 1, 1), 0, nullptr, std::move(Dot_502_0), std::move(Broadcast_503_0), std::move(Add_504_0), std::move(Parameter_270_0), std::move(Result_505_0)));
    kernels.emplace_back(new Result_float_float_cuda_lib_Result_505Kernel(std::move(Add_504_0), std::move(Result_505_0), std::move(Parameter_270_0), std::move(Result_505_0)));
}
