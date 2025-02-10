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
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
char* resnet50_group_persist_CUDA_GPU0_allocator_memory_pool;
float* resnet50_Constant_272_0;
float* resnet50_Constant_3_0;
float* resnet50_Constant_7_0;
float* resnet50_Constant_6_0;
float* resnet50_Constant_4_0;
float* resnet50_Constant_5_0;
float* resnet50_Constant_13_0;
float* resnet50_Constant_17_0;
float* resnet50_Constant_16_0;
float* resnet50_Constant_14_0;
float* resnet50_Constant_15_0;
float* resnet50_Constant_18_0;
float* resnet50_Constant_22_0;
float* resnet50_Constant_21_0;
float* resnet50_Constant_19_0;
float* resnet50_Constant_20_0;
float* resnet50_Constant_23_0;
float* resnet50_Constant_27_0;
float* resnet50_Constant_26_0;
float* resnet50_Constant_24_0;
float* resnet50_Constant_25_0;
float* resnet50_Constant_8_0;
float* resnet50_Constant_12_0;
float* resnet50_Constant_11_0;
float* resnet50_Constant_9_0;
float* resnet50_Constant_10_0;
float* resnet50_Constant_28_0;
float* resnet50_Constant_32_0;
float* resnet50_Constant_31_0;
float* resnet50_Constant_29_0;
float* resnet50_Constant_30_0;
float* resnet50_Constant_33_0;
float* resnet50_Constant_37_0;
float* resnet50_Constant_36_0;
float* resnet50_Constant_34_0;
float* resnet50_Constant_35_0;
float* resnet50_Constant_38_0;
float* resnet50_Constant_42_0;
float* resnet50_Constant_41_0;
float* resnet50_Constant_39_0;
float* resnet50_Constant_40_0;
float* resnet50_Constant_43_0;
float* resnet50_Constant_47_0;
float* resnet50_Constant_46_0;
float* resnet50_Constant_44_0;
float* resnet50_Constant_45_0;
float* resnet50_Constant_48_0;
float* resnet50_Constant_52_0;
float* resnet50_Constant_51_0;
float* resnet50_Constant_49_0;
float* resnet50_Constant_50_0;
float* resnet50_Constant_53_0;
float* resnet50_Constant_57_0;
float* resnet50_Constant_56_0;
float* resnet50_Constant_54_0;
float* resnet50_Constant_55_0;
float* resnet50_Constant_63_0;
float* resnet50_Constant_67_0;
float* resnet50_Constant_66_0;
float* resnet50_Constant_64_0;
float* resnet50_Constant_65_0;
float* resnet50_Constant_68_0;
float* resnet50_Constant_72_0;
float* resnet50_Constant_71_0;
float* resnet50_Constant_69_0;
float* resnet50_Constant_70_0;
float* resnet50_Constant_73_0;
float* resnet50_Constant_77_0;
float* resnet50_Constant_76_0;
float* resnet50_Constant_74_0;
float* resnet50_Constant_75_0;
float* resnet50_Constant_58_0;
float* resnet50_Constant_62_0;
float* resnet50_Constant_61_0;
float* resnet50_Constant_59_0;
float* resnet50_Constant_60_0;
float* resnet50_Constant_78_0;
float* resnet50_Constant_82_0;
float* resnet50_Constant_81_0;
float* resnet50_Constant_79_0;
float* resnet50_Constant_80_0;
float* resnet50_Constant_83_0;
float* resnet50_Constant_87_0;
float* resnet50_Constant_86_0;
float* resnet50_Constant_84_0;
float* resnet50_Constant_85_0;
float* resnet50_Constant_88_0;
float* resnet50_Constant_92_0;
float* resnet50_Constant_91_0;
float* resnet50_Constant_89_0;
float* resnet50_Constant_90_0;
float* resnet50_Constant_93_0;
float* resnet50_Constant_97_0;
float* resnet50_Constant_96_0;
float* resnet50_Constant_94_0;
float* resnet50_Constant_95_0;
float* resnet50_Constant_98_0;
float* resnet50_Constant_102_0;
float* resnet50_Constant_101_0;
float* resnet50_Constant_99_0;
float* resnet50_Constant_100_0;
float* resnet50_Constant_103_0;
float* resnet50_Constant_107_0;
float* resnet50_Constant_106_0;
float* resnet50_Constant_104_0;
float* resnet50_Constant_105_0;
float* resnet50_Constant_108_0;
float* resnet50_Constant_112_0;
float* resnet50_Constant_111_0;
float* resnet50_Constant_109_0;
float* resnet50_Constant_110_0;
float* resnet50_Constant_113_0;
float* resnet50_Constant_117_0;
float* resnet50_Constant_116_0;
float* resnet50_Constant_114_0;
float* resnet50_Constant_115_0;
float* resnet50_Constant_118_0;
float* resnet50_Constant_122_0;
float* resnet50_Constant_121_0;
float* resnet50_Constant_119_0;
float* resnet50_Constant_120_0;
float* resnet50_Constant_128_0;
float* resnet50_Constant_132_0;
float* resnet50_Constant_131_0;
float* resnet50_Constant_129_0;
float* resnet50_Constant_130_0;
float* resnet50_Constant_133_0;
float* resnet50_Constant_137_0;
float* resnet50_Constant_136_0;
float* resnet50_Constant_134_0;
float* resnet50_Constant_135_0;
float* resnet50_Constant_138_0;
float* resnet50_Constant_142_0;
float* resnet50_Constant_141_0;
float* resnet50_Constant_139_0;
float* resnet50_Constant_140_0;
float* resnet50_Constant_123_0;
float* resnet50_Constant_127_0;
float* resnet50_Constant_126_0;
float* resnet50_Constant_124_0;
float* resnet50_Constant_125_0;
float* resnet50_Constant_143_0;
float* resnet50_Constant_147_0;
float* resnet50_Constant_146_0;
float* resnet50_Constant_144_0;
float* resnet50_Constant_145_0;
float* resnet50_Constant_148_0;
float* resnet50_Constant_152_0;
float* resnet50_Constant_151_0;
float* resnet50_Constant_149_0;
float* resnet50_Constant_150_0;
float* resnet50_Constant_153_0;
float* resnet50_Constant_157_0;
float* resnet50_Constant_156_0;
float* resnet50_Constant_154_0;
float* resnet50_Constant_155_0;
float* resnet50_Constant_158_0;
float* resnet50_Constant_162_0;
float* resnet50_Constant_161_0;
float* resnet50_Constant_159_0;
float* resnet50_Constant_160_0;
float* resnet50_Constant_163_0;
float* resnet50_Constant_167_0;
float* resnet50_Constant_166_0;
float* resnet50_Constant_164_0;
float* resnet50_Constant_165_0;
float* resnet50_Constant_168_0;
float* resnet50_Constant_172_0;
float* resnet50_Constant_171_0;
float* resnet50_Constant_169_0;
float* resnet50_Constant_170_0;
float* resnet50_Constant_173_0;
float* resnet50_Constant_177_0;
float* resnet50_Constant_176_0;
float* resnet50_Constant_174_0;
float* resnet50_Constant_175_0;
float* resnet50_Constant_178_0;
float* resnet50_Constant_182_0;
float* resnet50_Constant_181_0;
float* resnet50_Constant_179_0;
float* resnet50_Constant_180_0;
float* resnet50_Constant_183_0;
float* resnet50_Constant_187_0;
float* resnet50_Constant_186_0;
float* resnet50_Constant_184_0;
float* resnet50_Constant_185_0;
float* resnet50_Constant_188_0;
float* resnet50_Constant_192_0;
float* resnet50_Constant_191_0;
float* resnet50_Constant_189_0;
float* resnet50_Constant_190_0;
float* resnet50_Constant_193_0;
float* resnet50_Constant_197_0;
float* resnet50_Constant_196_0;
float* resnet50_Constant_194_0;
float* resnet50_Constant_195_0;
float* resnet50_Constant_198_0;
float* resnet50_Constant_202_0;
float* resnet50_Constant_201_0;
float* resnet50_Constant_199_0;
float* resnet50_Constant_200_0;
float* resnet50_Constant_203_0;
float* resnet50_Constant_207_0;
float* resnet50_Constant_206_0;
float* resnet50_Constant_204_0;
float* resnet50_Constant_205_0;
float* resnet50_Constant_208_0;
float* resnet50_Constant_212_0;
float* resnet50_Constant_211_0;
float* resnet50_Constant_209_0;
float* resnet50_Constant_210_0;
float* resnet50_Constant_213_0;
float* resnet50_Constant_217_0;
float* resnet50_Constant_216_0;
float* resnet50_Constant_214_0;
float* resnet50_Constant_215_0;
float* resnet50_Constant_223_0;
float* resnet50_Constant_227_0;
float* resnet50_Constant_226_0;
float* resnet50_Constant_224_0;
float* resnet50_Constant_225_0;
float* resnet50_Constant_228_0;
float* resnet50_Constant_232_0;
float* resnet50_Constant_231_0;
float* resnet50_Constant_229_0;
float* resnet50_Constant_230_0;
float* resnet50_Constant_233_0;
float* resnet50_Constant_237_0;
float* resnet50_Constant_236_0;
float* resnet50_Constant_234_0;
float* resnet50_Constant_235_0;
float* resnet50_Constant_218_0;
float* resnet50_Constant_222_0;
float* resnet50_Constant_221_0;
float* resnet50_Constant_219_0;
float* resnet50_Constant_220_0;
float* resnet50_Constant_238_0;
float* resnet50_Constant_242_0;
float* resnet50_Constant_241_0;
float* resnet50_Constant_239_0;
float* resnet50_Constant_240_0;
float* resnet50_Constant_243_0;
float* resnet50_Constant_247_0;
float* resnet50_Constant_246_0;
float* resnet50_Constant_244_0;
float* resnet50_Constant_245_0;
float* resnet50_Constant_248_0;
float* resnet50_Constant_252_0;
float* resnet50_Constant_251_0;
float* resnet50_Constant_249_0;
float* resnet50_Constant_250_0;
float* resnet50_Constant_253_0;
float* resnet50_Constant_257_0;
float* resnet50_Constant_256_0;
float* resnet50_Constant_254_0;
float* resnet50_Constant_255_0;
float* resnet50_Constant_258_0;
float* resnet50_Constant_262_0;
float* resnet50_Constant_261_0;
float* resnet50_Constant_259_0;
float* resnet50_Constant_260_0;
float* resnet50_Constant_263_0;
float* resnet50_Constant_267_0;
float* resnet50_Constant_266_0;
float* resnet50_Constant_264_0;
float* resnet50_Constant_265_0;
float* resnet50_Constant_500_0;
float* resnet50_Constant_269_0;
float* resnet50_Constant_270_0;
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
}
char* resnet50_group_0_CUDA_GPU0_allocator_memory_pool;
float* resnet50_Reshape_271_0;
float* resnet50_Pad_273_0;
float* resnet50_Reshape_274_0;
float* resnet50_Convolution_275_0;
float* resnet50_BatchNormInference_276_0;
float* resnet50_Relu_277_0;
float* resnet50_MaxPool_278_0;
float* resnet50_Reshape_281_0;
float* resnet50_Convolution_282_0;
float* resnet50_BatchNormInference_284_0;
float* resnet50_Relu_285_0;
float* resnet50_Reshape_286_0;
float* resnet50_Convolution_287_0;
float* resnet50_BatchNormInference_288_0;
float* resnet50_Relu_289_0;
float* resnet50_Reshape_290_0;
float* resnet50_Convolution_291_0;
float* resnet50_BatchNormInference_292_0;
float* resnet50_Reshape_279_0;
float* resnet50_Convolution_280_0;
float* resnet50_BatchNormInference_283_0;
float* resnet50_Relu_294_0;
float* resnet50_Reshape_295_0;
float* resnet50_Convolution_296_0;
float* resnet50_BatchNormInference_297_0;
float* resnet50_Relu_298_0;
float* resnet50_Reshape_299_0;
float* resnet50_Convolution_300_0;
float* resnet50_BatchNormInference_301_0;
float* resnet50_Relu_302_0;
float* resnet50_Reshape_303_0;
float* resnet50_Convolution_304_0;
float* resnet50_BatchNormInference_305_0;
float* resnet50_Relu_307_0;
float* resnet50_Reshape_308_0;
float* resnet50_Convolution_309_0;
float* resnet50_BatchNormInference_310_0;
float* resnet50_Relu_311_0;
float* resnet50_Reshape_312_0;
float* resnet50_Convolution_313_0;
float* resnet50_BatchNormInference_314_0;
float* resnet50_Relu_315_0;
float* resnet50_Reshape_316_0;
float* resnet50_Convolution_317_0;
float* resnet50_BatchNormInference_318_0;
float* resnet50_Relu_320_0;
float* resnet50_Reshape_323_0;
float* resnet50_Convolution_324_0;
float* resnet50_BatchNormInference_326_0;
float* resnet50_Relu_327_0;
float* resnet50_Reshape_328_0;
float* resnet50_Convolution_329_0;
float* resnet50_BatchNormInference_330_0;
float* resnet50_Relu_331_0;
float* resnet50_Reshape_332_0;
float* resnet50_Convolution_333_0;
float* resnet50_BatchNormInference_334_0;
float* resnet50_Reshape_321_0;
float* resnet50_Convolution_322_0;
float* resnet50_BatchNormInference_325_0;
float* resnet50_Relu_336_0;
float* resnet50_Reshape_337_0;
float* resnet50_Convolution_338_0;
float* resnet50_BatchNormInference_339_0;
float* resnet50_Relu_340_0;
float* resnet50_Reshape_341_0;
float* resnet50_Convolution_342_0;
float* resnet50_BatchNormInference_343_0;
float* resnet50_Relu_344_0;
float* resnet50_Reshape_345_0;
float* resnet50_Convolution_346_0;
float* resnet50_BatchNormInference_347_0;
float* resnet50_Relu_349_0;
float* resnet50_Reshape_350_0;
float* resnet50_Convolution_351_0;
float* resnet50_BatchNormInference_352_0;
float* resnet50_Relu_353_0;
float* resnet50_Reshape_354_0;
float* resnet50_Convolution_355_0;
float* resnet50_BatchNormInference_356_0;
float* resnet50_Relu_357_0;
float* resnet50_Reshape_358_0;
float* resnet50_Convolution_359_0;
float* resnet50_BatchNormInference_360_0;
float* resnet50_Relu_362_0;
float* resnet50_Reshape_363_0;
float* resnet50_Convolution_364_0;
float* resnet50_BatchNormInference_365_0;
float* resnet50_Relu_366_0;
float* resnet50_Reshape_367_0;
float* resnet50_Convolution_368_0;
float* resnet50_BatchNormInference_369_0;
float* resnet50_Relu_370_0;
float* resnet50_Reshape_371_0;
float* resnet50_Convolution_372_0;
float* resnet50_BatchNormInference_373_0;
float* resnet50_Relu_375_0;
float* resnet50_Reshape_378_0;
float* resnet50_Convolution_379_0;
float* resnet50_BatchNormInference_381_0;
float* resnet50_Relu_382_0;
float* resnet50_Reshape_383_0;
float* resnet50_Convolution_384_0;
float* resnet50_BatchNormInference_385_0;
float* resnet50_Relu_386_0;
float* resnet50_Reshape_387_0;
float* resnet50_Convolution_388_0;
float* resnet50_BatchNormInference_389_0;
float* resnet50_Reshape_376_0;
float* resnet50_Convolution_377_0;
float* resnet50_BatchNormInference_380_0;
float* resnet50_Relu_391_0;
float* resnet50_Reshape_392_0;
float* resnet50_Convolution_393_0;
float* resnet50_BatchNormInference_394_0;
float* resnet50_Relu_395_0;
float* resnet50_Reshape_396_0;
float* resnet50_Convolution_397_0;
float* resnet50_BatchNormInference_398_0;
float* resnet50_Relu_399_0;
float* resnet50_Reshape_400_0;
float* resnet50_Convolution_401_0;
float* resnet50_BatchNormInference_402_0;
float* resnet50_Relu_404_0;
float* resnet50_Reshape_405_0;
float* resnet50_Convolution_406_0;
float* resnet50_BatchNormInference_407_0;
float* resnet50_Relu_408_0;
float* resnet50_Reshape_409_0;
float* resnet50_Convolution_410_0;
float* resnet50_BatchNormInference_411_0;
float* resnet50_Relu_412_0;
float* resnet50_Reshape_413_0;
float* resnet50_Convolution_414_0;
float* resnet50_BatchNormInference_415_0;
float* resnet50_Relu_417_0;
float* resnet50_Reshape_418_0;
float* resnet50_Convolution_419_0;
float* resnet50_BatchNormInference_420_0;
float* resnet50_Relu_421_0;
float* resnet50_Reshape_422_0;
float* resnet50_Convolution_423_0;
float* resnet50_BatchNormInference_424_0;
float* resnet50_Relu_425_0;
float* resnet50_Reshape_426_0;
float* resnet50_Convolution_427_0;
float* resnet50_BatchNormInference_428_0;
float* resnet50_Relu_430_0;
float* resnet50_Reshape_431_0;
float* resnet50_Convolution_432_0;
float* resnet50_BatchNormInference_433_0;
float* resnet50_Relu_434_0;
float* resnet50_Reshape_435_0;
float* resnet50_Convolution_436_0;
float* resnet50_BatchNormInference_437_0;
float* resnet50_Relu_438_0;
float* resnet50_Reshape_439_0;
float* resnet50_Convolution_440_0;
float* resnet50_BatchNormInference_441_0;
float* resnet50_Relu_443_0;
float* resnet50_Reshape_444_0;
float* resnet50_Convolution_445_0;
float* resnet50_BatchNormInference_446_0;
float* resnet50_Relu_447_0;
float* resnet50_Reshape_448_0;
float* resnet50_Convolution_449_0;
float* resnet50_BatchNormInference_450_0;
float* resnet50_Relu_451_0;
float* resnet50_Reshape_452_0;
float* resnet50_Convolution_453_0;
float* resnet50_BatchNormInference_454_0;
float* resnet50_Relu_456_0;
float* resnet50_Reshape_459_0;
float* resnet50_Convolution_460_0;
float* resnet50_BatchNormInference_462_0;
float* resnet50_Relu_463_0;
float* resnet50_Reshape_464_0;
float* resnet50_Convolution_465_0;
float* resnet50_BatchNormInference_466_0;
float* resnet50_Relu_467_0;
float* resnet50_Reshape_468_0;
float* resnet50_Convolution_469_0;
float* resnet50_BatchNormInference_470_0;
float* resnet50_Reshape_457_0;
float* resnet50_Convolution_458_0;
float* resnet50_BatchNormInference_461_0;
float* resnet50_Relu_472_0;
float* resnet50_Reshape_473_0;
float* resnet50_Convolution_474_0;
float* resnet50_BatchNormInference_475_0;
float* resnet50_Relu_476_0;
float* resnet50_Reshape_477_0;
float* resnet50_Convolution_478_0;
float* resnet50_BatchNormInference_479_0;
float* resnet50_Relu_480_0;
float* resnet50_Reshape_481_0;
float* resnet50_Convolution_482_0;
float* resnet50_BatchNormInference_483_0;
float* resnet50_Relu_485_0;
float* resnet50_Reshape_486_0;
float* resnet50_Convolution_487_0;
float* resnet50_BatchNormInference_488_0;
float* resnet50_Relu_489_0;
float* resnet50_Reshape_490_0;
float* resnet50_Convolution_491_0;
float* resnet50_BatchNormInference_492_0;
float* resnet50_Relu_493_0;
float* resnet50_Reshape_494_0;
float* resnet50_Convolution_495_0;
float* resnet50_BatchNormInference_496_0;
float* resnet50_Relu_498_0;
float* resnet50_Sum_499_0;
float* resnet50_Divide_501_0;
float* resnet50_Dot_502_0;
float* resnet50_Broadcast_503_0;
float* resnet50_Add_504_0;
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
int resnet50_num_SMs;
cublasHandle_t resnet50_cublas_handle_0;
cudnnHandle_t resnet50_cudnn_handle_0;

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {64, 224, 224, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {64, 1001}
#endif

// Node name:	Reshape_376
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_123_0	type: float	shape: Shape{1, 1, 512, 1024}
// Output:
//	- name: resnet50_Reshape_376_0	type: float	shape: Shape{1024, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_376(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_376_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_376<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_393
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_391_0	type: float	shape: Shape{64, 1024, 14, 14}
//	- name: resnet50_Reshape_392_0	type: float	shape: Shape{256, 1024, 1, 1}
// Output:
//	- name: resnet50_Convolution_393_0	type: float	shape: Shape{64, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_393(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_332
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_73_0	type: float	shape: Shape{1, 1, 128, 512}
// Output:
//	- name: resnet50_Reshape_332_0	type: float	shape: Shape{512, 128, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_332(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_332_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_332<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_465
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_463_0	type: float	shape: Shape{64, 512, 7, 7}
//	- name: resnet50_Reshape_464_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: resnet50_Convolution_465_0	type: float	shape: Shape{64, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_465(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 512, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_40_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_235
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_235_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_235(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_235_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_235_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_203
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_203_0	type: float	shape: Shape{1, 1, 1024, 256}
void resnet50_Constant_float_cuda_Constant_203(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_203_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_203_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_51_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_36_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_36_0 failed.\n");
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
//	- name: resnet50_Constant_10_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_11_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_27_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_27_0 failed.\n");
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
//	- name: resnet50_Constant_6_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_6_0 failed.\n");
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
//	- name: resnet50_Constant_15_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_21_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_21_0 failed.\n");
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
//	- name: resnet50_Constant_165_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_165(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_165_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_165_0 failed.\n");
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
//	- name: resnet50_Constant_178_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_178(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_178_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_178_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_12_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_169_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_169_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_20
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_20_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_20_0 failed.\n");
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
//	- name: resnet50_Constant_38_0	type: float	shape: Shape{1, 1, 64, 256}
void resnet50_Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_64
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_64_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_64(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_64_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_64_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_123
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_123_0	type: float	shape: Shape{1, 1, 512, 1024}
void resnet50_Constant_float_cuda_Constant_123(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_123_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_123_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_76_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_213
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_213_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_213(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_213_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_213_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_91_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_17
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_17_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_162
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_162_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_162_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_162_0 failed.\n");
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
//	- name: resnet50_Constant_4_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_44_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_259
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_259_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_259(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_259_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_259_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_70
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_70_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_70_0 failed.\n");
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
//	- name: resnet50_Constant_50_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_50_0 failed.\n");
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
//	- name: resnet50_Constant_196_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_196(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_196_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_196_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_34
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_34_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_34(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_34_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_14_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_192
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_192_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_192(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_192_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_192_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_246
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_246_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_246(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_246_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_246_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_129
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_129_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_129(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_129_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_129_0 failed.\n");
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
//	- name: resnet50_Constant_153_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_153(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_153_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_153_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_19_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_73
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_73_0	type: float	shape: Shape{1, 1, 128, 512}
void resnet50_Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_73_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_42_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_42_0 failed.\n");
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
//	- name: resnet50_Constant_125_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_125_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_82
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_82_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_82(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_82_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_82_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_135
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_135_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_135(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_135_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_135_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_39_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_60
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_60_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_60_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_94
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_94_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_168
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_168_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_168(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_168_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_168_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_13
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_13_0	type: float	shape: Shape{1, 1, 64, 64}
void resnet50_Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_13_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_236
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_236_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_236(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_236_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_236_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_61_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_43_0	type: float	shape: Shape{1, 1, 256, 64}
void resnet50_Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_120
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_120_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_31_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_66
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_66_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_66_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_190
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_190_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_190_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_25_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_158
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_158_0	type: float	shape: Shape{1, 1, 1024, 256}
void resnet50_Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_77
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_77_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_77(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_77_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_77_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_95
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_95_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_95_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_47_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_46_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_45_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_198
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_198_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_198(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_198_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_198_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_48_0	type: float	shape: Shape{3, 3, 64, 64}
void resnet50_Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_48_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_230
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_230_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_230(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_230_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_230_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_224
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_224_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_224(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_224_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_224_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_176
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_176_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_176_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_68_0	type: float	shape: Shape{3, 3, 128, 128}
void resnet50_Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_68_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_226
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_226_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_226(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_226_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_226_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_33_0	type: float	shape: Shape{3, 3, 64, 64}
void resnet50_Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_33_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_219
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_219_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_219(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_219_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_219_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_152
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_152_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_152(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_152_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_152_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_53_0	type: float	shape: Shape{1, 1, 64, 256}
void resnet50_Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_185
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_185_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_185_0 failed.\n");
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
//	- name: resnet50_Constant_99_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_99_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_134
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_134_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_134(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_134_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_134_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_57_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_197
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_197_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_197_0 failed.\n");
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
//	- name: resnet50_Constant_225_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_225(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_225_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_225_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_233
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_233_0	type: float	shape: Shape{1, 1, 512, 2048}
void resnet50_Constant_float_cuda_Constant_233(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_233_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_233_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_223
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_223_0	type: float	shape: Shape{1, 1, 1024, 512}
void resnet50_Constant_float_cuda_Constant_223(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_223_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_223_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_234
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_234_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_234(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_234_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_234_0 failed.\n");
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
//	- name: resnet50_Constant_186_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_186(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_186_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_186_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_188
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_188_0	type: float	shape: Shape{1, 1, 1024, 256}
void resnet50_Constant_float_cuda_Constant_188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_188_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_214
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_214_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_214(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_214_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_214_0 failed.\n");
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
//	- name: resnet50_Constant_41_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_41(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_41_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_41_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_222
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_222_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_222(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_222_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_222_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_189
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_189_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_189(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_189_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_189_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_183
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_183_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_183(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_183_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_183_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_242
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_242_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_242(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_242_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_242_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_239
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_239_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_239(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_239_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_239_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_133
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_133_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_133(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_133_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_133_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_23_0	type: float	shape: Shape{1, 1, 64, 256}
void resnet50_Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_267
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_267_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_267(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_267_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_267_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_207
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_207_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_207(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_207_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_207_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_78_0	type: float	shape: Shape{1, 1, 512, 128}
void resnet50_Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_78_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_180
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_180_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_180(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_180_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_180_0 failed.\n");
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
//	- name: resnet50_Constant_247_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_247(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_247_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_247_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_215
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_215_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_215(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_215_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_215_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_202
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_202_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_202(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_202_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_202_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_229
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_229_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_229(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_229_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_229_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_241
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_241_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_241(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_241_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_241_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_244
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_244_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_244(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_244_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_244_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_126_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_137
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_137_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_107
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_107_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_107_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_245
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_245_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_245(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_245_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_245_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_248
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_248_0	type: float	shape: Shape{1, 1, 512, 2048}
void resnet50_Constant_float_cuda_Constant_248(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_248_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_248_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_266
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_266_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_266(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_266_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_266_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_55_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_55_0 failed.\n");
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
//	- name: resnet50_Constant_170_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_59_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_59_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_270
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_270_0	type: float	shape: Shape{1001}
void resnet50_Constant_float_cuda_Constant_270(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_270_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_270_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_237
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_237_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_237(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_237_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_237_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_71
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_71_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_71_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_177
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_177_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_177_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_22
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_22_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_22_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_262
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_262_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_262(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_262_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_262_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_254
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_254_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_254_0 failed.\n");
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
//	- name: resnet50_Constant_83_0	type: float	shape: Shape{3, 3, 128, 128}
void resnet50_Constant_float_cuda_Constant_83(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_83_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_83_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_218
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_218_0	type: float	shape: Shape{1, 1, 1024, 2048}
void resnet50_Constant_float_cuda_Constant_218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_218_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8388608];
    bin_file.read(tmp_mem, 8388608);
    cudaMemcpyAsync(output0, tmp_mem, 8388608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_258
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_258_0	type: float	shape: Shape{3, 3, 512, 512}
void resnet50_Constant_float_cuda_Constant_258(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_258_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_258_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_249
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_249_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_249(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_249_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_249_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_253
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_253_0	type: float	shape: Shape{1, 1, 2048, 512}
void resnet50_Constant_float_cuda_Constant_253(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_253_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_253_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_260
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_260_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_260(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_260_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_260_0 failed.\n");
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
//	- name: resnet50_Constant_269_0	type: float	shape: Shape{2048, 1001}
void resnet50_Constant_float_cuda_Constant_269(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_269_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_269_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8200192];
    bin_file.read(tmp_mem, 8200192);
    cudaMemcpyAsync(output0, tmp_mem, 8200192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_216
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_216_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_216(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_216_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_216_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_227
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_227_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_227(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_227_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_227_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_174
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_174_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_174_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_217
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_217_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_217(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_217_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_217_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_238
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_238_0	type: float	shape: Shape{1, 1, 2048, 512}
void resnet50_Constant_float_cuda_Constant_238(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_238_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_238_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_210
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_210_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_210(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_210_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_210_0 failed.\n");
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
//	- name: resnet50_Constant_232_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_232(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_232_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_232_0 failed.\n");
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
//	- name: resnet50_Constant_261_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_261(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_261_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_261_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_187
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_187_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_187(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_187_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_187_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_206
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_206_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_206(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_206_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_206_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_200
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_200_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_200(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_200_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_200_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_199
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_199_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_199(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_199_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_199_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_201
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_201_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_201(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_201_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_201_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_58_0	type: float	shape: Shape{1, 1, 256, 512}
void resnet50_Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[524288];
    bin_file.read(tmp_mem, 524288);
    cudaMemcpyAsync(output0, tmp_mem, 524288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_211
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_211_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_211(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_211_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_211_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_195
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_195_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_195(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_195_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_195_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_243
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_243_0	type: float	shape: Shape{3, 3, 512, 512}
void resnet50_Constant_float_cuda_Constant_243(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_243_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_243_0 failed.\n");
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
//	- name: resnet50_Constant_179_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_179_0 failed.\n");
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
//	- name: resnet50_Constant_184_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_184(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_184_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_184_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_220
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_220_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_220(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_220_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_220_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_194
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_194_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_194_0 failed.\n");
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
//	- name: resnet50_Constant_160_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_160(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_160_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_160_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_257
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_257_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_257(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_257_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_257_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_191
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_191_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_204
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_204_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_204(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_204_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_204_0 failed.\n");
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
//	- name: resnet50_Constant_62_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_62(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_62_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_62_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_252
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_252_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_252(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_252_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_252_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_205
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_205_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_205(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_205_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_205_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_29_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_164
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_164_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_181
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_181_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_181_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_182
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_182_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_182_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_231
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_231_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_231(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_231_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_231_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_175
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_175_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_175(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_175_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_175_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_24_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_256
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_256_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_256(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_256_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_256_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_172
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_172_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_172(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_172_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_172_0 failed.\n");
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
//	- name: resnet50_Constant_75_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_75(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_75_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_75_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_37_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_9_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_9_0 failed.\n");
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
//	- name: resnet50_Constant_250_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_250(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_250_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_250_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_65
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_65_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_65_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_49
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_49_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_49_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_221
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_221_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_221(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_221_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_221_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_8_0	type: float	shape: Shape{1, 1, 64, 256}
void resnet50_Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_89
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_89_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_89(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_89_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_89_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_156
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_156_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_156_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_35_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_3_0	type: float	shape: Shape{7, 7, 3, 64}
void resnet50_Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[37632];
    bin_file.read(tmp_mem, 37632);
    cudaMemcpyAsync(output0, tmp_mem, 37632, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_54_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_54_0 failed.\n");
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
//	- name: resnet50_Constant_30_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_143
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_143_0	type: float	shape: Shape{1, 1, 1024, 256}
void resnet50_Constant_float_cuda_Constant_143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_143_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_212
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_212_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_212(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_212_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_212_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_147
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_147_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_147(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_147_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_147_0 failed.\n");
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
//	- name: resnet50_Constant_74_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_74(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_74_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_74_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_141
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_141_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_141(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_141_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_141_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_150
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_150_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_150_0 failed.\n");
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
//	- name: resnet50_Constant_81_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_81(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_81_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_81_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_69
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_69_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_69_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_85
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_85_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_85_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_5_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_272
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_272_0	type: float	shape: Shape{}
void resnet50_Constant_float_cuda_Constant_272(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_272_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_272_0 failed.\n");
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
//	- name: resnet50_Constant_80_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_80(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_80_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_80_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_140
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_140_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_140_0 failed.\n");
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
//	- name: resnet50_Constant_86_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_86_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_171
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_171_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_171_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_84_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_88_0	type: float	shape: Shape{1, 1, 128, 512}
void resnet50_Constant_float_cuda_Constant_88(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_88_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_92_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_92_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_265
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_265_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_265(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_265_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_265_0 failed.\n");
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
//	- name: resnet50_Constant_90_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_500
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_500_0	type: float	shape: Shape{64, 2048}
void resnet50_Constant_float_cuda_Constant_500(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_500_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_500_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[524288];
    bin_file.read(tmp_mem, 524288);
    cudaMemcpyAsync(output0, tmp_mem, 524288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_263
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_263_0	type: float	shape: Shape{1, 1, 512, 2048}
void resnet50_Constant_float_cuda_Constant_263(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_263_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_263_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_93_0	type: float	shape: Shape{1, 1, 512, 128}
void resnet50_Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_97
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_97_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_97_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_163
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_163_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_163(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_163_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_163_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_96_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_96_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_166
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_166_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_166(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_166_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_166_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_102
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_102_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_102(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_102_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_102_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_87_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_87_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_136
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_136_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_136(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_136_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_136_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_167_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_167_0 failed.\n");
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
//	- name: resnet50_Constant_101_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_101_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_264
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_264_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_264(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_264_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_264_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_67_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_67_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_100_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_100_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_103
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_103_0	type: float	shape: Shape{1, 1, 128, 512}
void resnet50_Constant_float_cuda_Constant_103(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_103_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_103_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_208
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_208_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_208(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_208_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_208_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_104_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_228
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_228_0	type: float	shape: Shape{3, 3, 512, 512}
void resnet50_Constant_float_cuda_Constant_228(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_228_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_228_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_28_0	type: float	shape: Shape{1, 1, 256, 64}
void resnet50_Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_105
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_105_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_105(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_105_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_105_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_108
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_108_0	type: float	shape: Shape{1, 1, 512, 128}
void resnet50_Constant_float_cuda_Constant_108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_108_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_240
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_240_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_240(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_240_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_240_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_113
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_113_0	type: float	shape: Shape{3, 3, 128, 128}
void resnet50_Constant_float_cuda_Constant_113(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_113_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_113_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_26_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_26_0 failed.\n");
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
//	- name: resnet50_Constant_118_0	type: float	shape: Shape{1, 1, 128, 512}
void resnet50_Constant_float_cuda_Constant_118(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_118_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_118_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_112_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_106
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_106_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_106(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_106_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_106_0 failed.\n");
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
//	- name: resnet50_Constant_121_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_121(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_121_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_121_0 failed.\n");
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
//	- name: resnet50_Constant_111_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_111_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_109_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_109_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_154
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_154_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_154(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_154_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_154_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_117_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_117(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_117_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_117_0 failed.\n");
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
//	- name: resnet50_Constant_116_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_114
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_114_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_114_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_52_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_119
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_119_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_119_0 failed.\n");
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
//	- name: resnet50_Constant_115_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_115(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_115_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_115_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_142
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_142_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_142(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_142_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_142_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_122
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_122_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_122(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_122_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_122_0 failed.\n");
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
//	- name: resnet50_Constant_32_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_128
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_128_0	type: float	shape: Shape{1, 1, 512, 256}
void resnet50_Constant_float_cuda_Constant_128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_128_0 failed.\n");
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
//	- name: resnet50_Constant_209_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_209(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_209_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_209_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_159
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_159_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_159(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_159_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_159_0 failed.\n");
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
//	- name: resnet50_Constant_131_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_131(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_131_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_131_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_255
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_255_0	type: float	shape: Shape{512}
void resnet50_Constant_float_cuda_Constant_255(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_255_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_255_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_173
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_173_0	type: float	shape: Shape{1, 1, 1024, 256}
void resnet50_Constant_float_cuda_Constant_173(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_173_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_173_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_130
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_130_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_130(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_130_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_130_0 failed.\n");
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
//	- name: resnet50_Constant_138_0	type: float	shape: Shape{1, 1, 256, 1024}
void resnet50_Constant_float_cuda_Constant_138(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_138_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_138_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_193_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_193(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_193_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_193_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_139
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_139_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_139(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_139_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_139_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_251
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_251_0	type: float	shape: Shape{2048}
void resnet50_Constant_float_cuda_Constant_251(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_251_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_251_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8192];
    bin_file.read(tmp_mem, 8192);
    cudaMemcpyAsync(output0, tmp_mem, 8192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_79
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_79_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_79_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_127
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_127_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_127(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_127_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_127_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_124
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_124_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_124(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_124_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_124_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_63
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_63_0	type: float	shape: Shape{1, 1, 256, 128}
void resnet50_Constant_float_cuda_Constant_63(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_63_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_63_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_98
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_98_0	type: float	shape: Shape{3, 3, 128, 128}
void resnet50_Constant_float_cuda_Constant_98(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_98_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_98_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_110
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_110_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_110_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_146
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_146_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_56
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_56_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_56_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_144_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_144_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_145
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_145_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_145(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_145_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_145_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_132
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_132_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_132(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_132_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_132_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_148
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_148_0	type: float	shape: Shape{3, 3, 256, 256}
void resnet50_Constant_float_cuda_Constant_148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_148_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_151
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_151_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_151(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_151_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_151_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_149
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_149_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_149_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_18_0	type: float	shape: Shape{3, 3, 64, 64}
void resnet50_Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_157
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_157_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_157(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_157_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_157_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_155
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_155_0	type: float	shape: Shape{1024}
void resnet50_Constant_float_cuda_Constant_155(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_155_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_155_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_72_0	type: float	shape: Shape{128}
void resnet50_Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_72_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_16
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_16_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_16_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_7_0	type: float	shape: Shape{64}
void resnet50_Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_161
// Description:	Constant
// Input:
// Output:
//	- name: resnet50_Constant_161_0	type: float	shape: Shape{256}
void resnet50_Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/resnet50/Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load resnet50_Constant_161_0 failed.\n");
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
//	- name: resnet50_Constant_233_0	type: float	shape: Shape{1, 1, 512, 2048}
// Output:
//	- name: resnet50_Reshape_468_0	type: float	shape: Shape{2048, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_468(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_468_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_468<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_276
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_4_0	type: float	shape: Shape{64}
//	- name: resnet50_Constant_5_0	type: float	shape: Shape{64}
//	- name: resnet50_Convolution_275_0	type: float	shape: Shape{64, 64, 112, 112}
//	- name: resnet50_Constant_6_0	type: float	shape: Shape{64}
//	- name: resnet50_Constant_7_0	type: float	shape: Shape{64}
// Output:
//	- name: resnet50_BatchNormInference_276_0	type: float	shape: Shape{64, 64, 112, 112}
extern "C" __launch_bounds__(512) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 112 * 112;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 112 * 112; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Divide_501
// Description:	Divide
// Input:
//	- name: resnet50_Sum_499_0	type: float	shape: Shape{64, 2048}
//	- name: resnet50_Constant_500_0	type: float	shape: Shape{64, 2048}
// Output:
//	- name: resnet50_Divide_501_0	type: float	shape: Shape{64, 2048}
extern "C" __launch_bounds__(512) __global__ void resnet50_Divide_float_float_float_cuda_Divide_501(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = fdividef(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void resnet50_Divide_float_float_float_cuda_Divide_501_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Divide_float_float_float_cuda_Divide_501<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_291
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_289_0	type: float	shape: Shape{64, 64, 56, 56}
//	- name: resnet50_Reshape_290_0	type: float	shape: Shape{256, 64, 1, 1}
// Output:
//	- name: resnet50_Convolution_291_0	type: float	shape: Shape{64, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_291(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_324
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_320_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: resnet50_Reshape_323_0	type: float	shape: Shape{128, 256, 1, 1}
// Output:
//	- name: resnet50_Convolution_324_0	type: float	shape: Shape{64, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_324(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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

extern "C" void resnet50_cuda_init()
{
// total memory:822361856

CUDA_SAFE_CALL(cudaMalloc((void**)&resnet50_group_persist_CUDA_GPU0_allocator_memory_pool,102973184));
CUDA_SAFE_CALL(cudaMemset((void*)resnet50_group_persist_CUDA_GPU0_allocator_memory_pool, 0, 102973184));
resnet50_Constant_272_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Constant_3_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+64);
resnet50_Constant_7_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+37696);
resnet50_Constant_6_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+37952);
resnet50_Constant_4_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+38208);
resnet50_Constant_5_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+38464);
resnet50_Constant_13_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+38720);
resnet50_Constant_17_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+55104);
resnet50_Constant_16_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+55360);
resnet50_Constant_14_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+55616);
resnet50_Constant_15_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+55872);
resnet50_Constant_18_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+56128);
resnet50_Constant_22_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+203584);
resnet50_Constant_21_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+203840);
resnet50_Constant_19_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+204096);
resnet50_Constant_20_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+204352);
resnet50_Constant_23_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+204608);
resnet50_Constant_27_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+270144);
resnet50_Constant_26_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+271168);
resnet50_Constant_24_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+272192);
resnet50_Constant_25_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+273216);
resnet50_Constant_8_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+274240);
resnet50_Constant_12_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+339776);
resnet50_Constant_11_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+340800);
resnet50_Constant_9_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+341824);
resnet50_Constant_10_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+342848);
resnet50_Constant_28_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+343872);
resnet50_Constant_32_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+409408);
resnet50_Constant_31_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+409664);
resnet50_Constant_29_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+409920);
resnet50_Constant_30_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+410176);
resnet50_Constant_33_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+410432);
resnet50_Constant_37_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+557888);
resnet50_Constant_36_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+558144);
resnet50_Constant_34_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+558400);
resnet50_Constant_35_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+558656);
resnet50_Constant_38_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+558912);
resnet50_Constant_42_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+624448);
resnet50_Constant_41_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+625472);
resnet50_Constant_39_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+626496);
resnet50_Constant_40_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+627520);
resnet50_Constant_43_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+628544);
resnet50_Constant_47_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+694080);
resnet50_Constant_46_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+694336);
resnet50_Constant_44_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+694592);
resnet50_Constant_45_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+694848);
resnet50_Constant_48_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+695104);
resnet50_Constant_52_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+842560);
resnet50_Constant_51_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+842816);
resnet50_Constant_49_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+843072);
resnet50_Constant_50_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+843328);
resnet50_Constant_53_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+843584);
resnet50_Constant_57_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+909120);
resnet50_Constant_56_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+910144);
resnet50_Constant_54_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+911168);
resnet50_Constant_55_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+912192);
resnet50_Constant_63_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+913216);
resnet50_Constant_67_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1044288);
resnet50_Constant_66_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1044800);
resnet50_Constant_64_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1045312);
resnet50_Constant_65_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1045824);
resnet50_Constant_68_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1046336);
resnet50_Constant_72_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1636160);
resnet50_Constant_71_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1636672);
resnet50_Constant_69_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1637184);
resnet50_Constant_70_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1637696);
resnet50_Constant_73_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1638208);
resnet50_Constant_77_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1900352);
resnet50_Constant_76_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1902400);
resnet50_Constant_74_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1904448);
resnet50_Constant_75_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1906496);
resnet50_Constant_58_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+1908544);
resnet50_Constant_62_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2432832);
resnet50_Constant_61_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2434880);
resnet50_Constant_59_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2436928);
resnet50_Constant_60_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2438976);
resnet50_Constant_78_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2441024);
resnet50_Constant_82_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2703168);
resnet50_Constant_81_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2703680);
resnet50_Constant_79_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2704192);
resnet50_Constant_80_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2704704);
resnet50_Constant_83_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+2705216);
resnet50_Constant_87_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3295040);
resnet50_Constant_86_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3295552);
resnet50_Constant_84_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3296064);
resnet50_Constant_85_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3296576);
resnet50_Constant_88_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3297088);
resnet50_Constant_92_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3559232);
resnet50_Constant_91_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3561280);
resnet50_Constant_89_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3563328);
resnet50_Constant_90_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3565376);
resnet50_Constant_93_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3567424);
resnet50_Constant_97_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3829568);
resnet50_Constant_96_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3830080);
resnet50_Constant_94_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3830592);
resnet50_Constant_95_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3831104);
resnet50_Constant_98_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+3831616);
resnet50_Constant_102_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4421440);
resnet50_Constant_101_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4421952);
resnet50_Constant_99_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4422464);
resnet50_Constant_100_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4422976);
resnet50_Constant_103_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4423488);
resnet50_Constant_107_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4685632);
resnet50_Constant_106_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4687680);
resnet50_Constant_104_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4689728);
resnet50_Constant_105_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4691776);
resnet50_Constant_108_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4693824);
resnet50_Constant_112_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4955968);
resnet50_Constant_111_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4956480);
resnet50_Constant_109_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4956992);
resnet50_Constant_110_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4957504);
resnet50_Constant_113_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+4958016);
resnet50_Constant_117_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5547840);
resnet50_Constant_116_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5548352);
resnet50_Constant_114_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5548864);
resnet50_Constant_115_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5549376);
resnet50_Constant_118_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5549888);
resnet50_Constant_122_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5812032);
resnet50_Constant_121_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5814080);
resnet50_Constant_119_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5816128);
resnet50_Constant_120_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5818176);
resnet50_Constant_128_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+5820224);
resnet50_Constant_132_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+6344512);
resnet50_Constant_131_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+6345536);
resnet50_Constant_129_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+6346560);
resnet50_Constant_130_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+6347584);
resnet50_Constant_133_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+6348608);
resnet50_Constant_137_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+8707904);
resnet50_Constant_136_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+8708928);
resnet50_Constant_134_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+8709952);
resnet50_Constant_135_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+8710976);
resnet50_Constant_138_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+8712000);
resnet50_Constant_142_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+9760576);
resnet50_Constant_141_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+9764672);
resnet50_Constant_139_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+9768768);
resnet50_Constant_140_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+9772864);
resnet50_Constant_123_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+9776960);
resnet50_Constant_127_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+11874112);
resnet50_Constant_126_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+11878208);
resnet50_Constant_124_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+11882304);
resnet50_Constant_125_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+11886400);
resnet50_Constant_143_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+11890496);
resnet50_Constant_147_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+12939072);
resnet50_Constant_146_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+12940096);
resnet50_Constant_144_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+12941120);
resnet50_Constant_145_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+12942144);
resnet50_Constant_148_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+12943168);
resnet50_Constant_152_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+15302464);
resnet50_Constant_151_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+15303488);
resnet50_Constant_149_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+15304512);
resnet50_Constant_150_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+15305536);
resnet50_Constant_153_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+15306560);
resnet50_Constant_157_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+16355136);
resnet50_Constant_156_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+16359232);
resnet50_Constant_154_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+16363328);
resnet50_Constant_155_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+16367424);
resnet50_Constant_158_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+16371520);
resnet50_Constant_162_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+17420096);
resnet50_Constant_161_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+17421120);
resnet50_Constant_159_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+17422144);
resnet50_Constant_160_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+17423168);
resnet50_Constant_163_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+17424192);
resnet50_Constant_167_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+19783488);
resnet50_Constant_166_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+19784512);
resnet50_Constant_164_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+19785536);
resnet50_Constant_165_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+19786560);
resnet50_Constant_168_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+19787584);
resnet50_Constant_172_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+20836160);
resnet50_Constant_171_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+20840256);
resnet50_Constant_169_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+20844352);
resnet50_Constant_170_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+20848448);
resnet50_Constant_173_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+20852544);
resnet50_Constant_177_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+21901120);
resnet50_Constant_176_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+21902144);
resnet50_Constant_174_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+21903168);
resnet50_Constant_175_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+21904192);
resnet50_Constant_178_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+21905216);
resnet50_Constant_182_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+24264512);
resnet50_Constant_181_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+24265536);
resnet50_Constant_179_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+24266560);
resnet50_Constant_180_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+24267584);
resnet50_Constant_183_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+24268608);
resnet50_Constant_187_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+25317184);
resnet50_Constant_186_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+25321280);
resnet50_Constant_184_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+25325376);
resnet50_Constant_185_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+25329472);
resnet50_Constant_188_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+25333568);
resnet50_Constant_192_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+26382144);
resnet50_Constant_191_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+26383168);
resnet50_Constant_189_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+26384192);
resnet50_Constant_190_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+26385216);
resnet50_Constant_193_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+26386240);
resnet50_Constant_197_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+28745536);
resnet50_Constant_196_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+28746560);
resnet50_Constant_194_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+28747584);
resnet50_Constant_195_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+28748608);
resnet50_Constant_198_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+28749632);
resnet50_Constant_202_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+29798208);
resnet50_Constant_201_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+29802304);
resnet50_Constant_199_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+29806400);
resnet50_Constant_200_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+29810496);
resnet50_Constant_203_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+29814592);
resnet50_Constant_207_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+30863168);
resnet50_Constant_206_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+30864192);
resnet50_Constant_204_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+30865216);
resnet50_Constant_205_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+30866240);
resnet50_Constant_208_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+30867264);
resnet50_Constant_212_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+33226560);
resnet50_Constant_211_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+33227584);
resnet50_Constant_209_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+33228608);
resnet50_Constant_210_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+33229632);
resnet50_Constant_213_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+33230656);
resnet50_Constant_217_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+34279232);
resnet50_Constant_216_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+34283328);
resnet50_Constant_214_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+34287424);
resnet50_Constant_215_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+34291520);
resnet50_Constant_223_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+34295616);
resnet50_Constant_227_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+36392768);
resnet50_Constant_226_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+36394816);
resnet50_Constant_224_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+36396864);
resnet50_Constant_225_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+36398912);
resnet50_Constant_228_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+36400960);
resnet50_Constant_232_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+45838144);
resnet50_Constant_231_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+45840192);
resnet50_Constant_229_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+45842240);
resnet50_Constant_230_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+45844288);
resnet50_Constant_233_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+45846336);
resnet50_Constant_237_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+50040640);
resnet50_Constant_236_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+50048832);
resnet50_Constant_234_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+50057024);
resnet50_Constant_235_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+50065216);
resnet50_Constant_218_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+50073408);
resnet50_Constant_222_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+58462016);
resnet50_Constant_221_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+58470208);
resnet50_Constant_219_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+58478400);
resnet50_Constant_220_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+58486592);
resnet50_Constant_238_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+58494784);
resnet50_Constant_242_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+62689088);
resnet50_Constant_241_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+62691136);
resnet50_Constant_239_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+62693184);
resnet50_Constant_240_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+62695232);
resnet50_Constant_243_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+62697280);
resnet50_Constant_247_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+72134464);
resnet50_Constant_246_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+72136512);
resnet50_Constant_244_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+72138560);
resnet50_Constant_245_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+72140608);
resnet50_Constant_248_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+72142656);
resnet50_Constant_252_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+76336960);
resnet50_Constant_251_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+76345152);
resnet50_Constant_249_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+76353344);
resnet50_Constant_250_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+76361536);
resnet50_Constant_253_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+76369728);
resnet50_Constant_257_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+80564032);
resnet50_Constant_256_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+80566080);
resnet50_Constant_254_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+80568128);
resnet50_Constant_255_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+80570176);
resnet50_Constant_258_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+80572224);
resnet50_Constant_262_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+90009408);
resnet50_Constant_261_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+90011456);
resnet50_Constant_259_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+90013504);
resnet50_Constant_260_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+90015552);
resnet50_Constant_263_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+90017600);
resnet50_Constant_267_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94211904);
resnet50_Constant_266_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94220096);
resnet50_Constant_264_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94228288);
resnet50_Constant_265_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94236480);
resnet50_Constant_500_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94244672);
resnet50_Constant_269_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+94768960);
resnet50_Constant_270_0 = (float*)(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool+102969152);

CUDA_SAFE_CALL(cudaMalloc((void**)&resnet50_group_0_CUDA_GPU0_allocator_memory_pool,719388672));
CUDA_SAFE_CALL(cudaMemset((void*)resnet50_group_0_CUDA_GPU0_allocator_memory_pool, 0, 719388672));
resnet50_Reshape_271_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Pad_273_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+38535168);
resnet50_Reshape_274_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_275_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+79162368);
resnet50_BatchNormInference_276_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+284683264);
resnet50_Relu_277_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+284683264);
resnet50_MaxPool_278_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_281_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_282_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51396608);
resnet50_BatchNormInference_284_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102776832);
resnet50_Relu_285_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102776832);
resnet50_Reshape_286_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_287_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+154157056);
resnet50_BatchNormInference_288_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Relu_289_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Reshape_290_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_291_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102825984);
resnet50_BatchNormInference_292_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+308346880);
resnet50_Reshape_279_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_280_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51445760);
resnet50_BatchNormInference_283_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+513867776);
resnet50_Relu_294_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_295_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Convolution_296_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205586432);
resnet50_BatchNormInference_297_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+256966656);
resnet50_Relu_298_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+256966656);
resnet50_Reshape_299_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Convolution_300_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+308346880);
resnet50_BatchNormInference_301_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Relu_302_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Reshape_303_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+256901120);
resnet50_Convolution_304_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+256966656);
resnet50_BatchNormInference_305_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+462487552);
resnet50_Relu_307_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Reshape_308_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_309_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+65536);
resnet50_BatchNormInference_310_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51445760);
resnet50_Relu_311_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51445760);
resnet50_Reshape_312_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_313_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102825984);
resnet50_BatchNormInference_314_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_315_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_316_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_317_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+411041792);
resnet50_BatchNormInference_318_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_320_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+411041792);
resnet50_Reshape_323_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_324_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+131072);
resnet50_BatchNormInference_326_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25821184);
resnet50_Relu_327_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25821184);
resnet50_Reshape_328_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_329_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51511296);
resnet50_BatchNormInference_330_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_331_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_332_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Convolution_333_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_BatchNormInference_334_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128712704);
resnet50_Reshape_321_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_322_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+524288);
resnet50_BatchNormInference_325_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+231473152);
resnet50_Relu_336_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_337_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_338_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+103022592);
resnet50_BatchNormInference_339_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128712704);
resnet50_Relu_340_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128712704);
resnet50_Reshape_341_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_342_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+154402816);
resnet50_BatchNormInference_343_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Relu_344_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Reshape_345_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128450560);
resnet50_Convolution_346_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128712704);
resnet50_BatchNormInference_347_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+231473152);
resnet50_Relu_349_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Reshape_350_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_351_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+262144);
resnet50_BatchNormInference_352_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_Relu_353_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_Reshape_354_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_355_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51642368);
resnet50_BatchNormInference_356_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_357_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_358_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Convolution_359_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_BatchNormInference_360_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_362_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+205520896);
resnet50_Reshape_363_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_364_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+262144);
resnet50_BatchNormInference_365_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_Relu_366_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_Reshape_367_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_368_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51642368);
resnet50_BatchNormInference_369_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_370_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_371_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Convolution_372_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25952256);
resnet50_BatchNormInference_373_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+308281344);
resnet50_Relu_375_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_378_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_379_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+103284736);
resnet50_BatchNormInference_381_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+116129792);
resnet50_Relu_382_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+116129792);
resnet50_Reshape_383_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_384_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+128974848);
resnet50_BatchNormInference_385_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Relu_386_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Reshape_387_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+115605504);
resnet50_Convolution_388_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+116654080);
resnet50_BatchNormInference_389_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+168034304);
resnet50_Reshape_376_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Convolution_377_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+104857600);
resnet50_BatchNormInference_380_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_391_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Reshape_392_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_393_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+1048576);
resnet50_BatchNormInference_394_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Relu_395_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Reshape_396_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_397_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+26738688);
resnet50_BatchNormInference_398_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_399_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_400_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+12845056);
resnet50_Convolution_401_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_BatchNormInference_402_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_404_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Reshape_405_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_406_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+1048576);
resnet50_BatchNormInference_407_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Relu_408_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Reshape_409_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_410_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+26738688);
resnet50_BatchNormInference_411_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_412_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_413_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+12845056);
resnet50_Convolution_414_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_BatchNormInference_415_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+154140672);
resnet50_Relu_417_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_418_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_419_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+52428800);
resnet50_BatchNormInference_420_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+65273856);
resnet50_Relu_421_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+65273856);
resnet50_Reshape_422_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_423_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+78118912);
resnet50_BatchNormInference_424_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Relu_425_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Reshape_426_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+64225280);
resnet50_Convolution_427_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+65273856);
resnet50_BatchNormInference_428_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+116654080);
resnet50_Relu_430_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Reshape_431_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_432_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+1048576);
resnet50_BatchNormInference_433_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Relu_434_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Reshape_435_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_436_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+26738688);
resnet50_BatchNormInference_437_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_438_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_439_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+12845056);
resnet50_Convolution_440_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_BatchNormInference_441_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_443_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
resnet50_Reshape_444_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_445_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+1048576);
resnet50_BatchNormInference_446_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Relu_447_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_Reshape_448_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_449_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+26738688);
resnet50_BatchNormInference_450_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_451_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_452_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+12845056);
resnet50_Convolution_453_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+13893632);
resnet50_BatchNormInference_454_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+154140672);
resnet50_Relu_456_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_459_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_460_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+53477376);
resnet50_BatchNormInference_462_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+59899904);
resnet50_Relu_463_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+59899904);
resnet50_Reshape_464_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+66322432);
resnet50_Convolution_465_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_BatchNormInference_466_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+57802752);
resnet50_Relu_467_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+57802752);
resnet50_Reshape_468_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_469_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+64225280);
resnet50_BatchNormInference_470_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+89915392);
resnet50_Reshape_457_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Convolution_458_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+59768832);
resnet50_BatchNormInference_461_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_472_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Reshape_473_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_474_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+4194304);
resnet50_BatchNormInference_475_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+10616832);
resnet50_Relu_476_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+10616832);
resnet50_Reshape_477_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_478_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+17039360);
resnet50_BatchNormInference_479_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_480_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_481_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+6422528);
resnet50_Convolution_482_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_BatchNormInference_483_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_485_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
resnet50_Reshape_486_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_487_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+4194304);
resnet50_BatchNormInference_488_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+10616832);
resnet50_Relu_489_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+10616832);
resnet50_Reshape_490_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Convolution_491_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+17039360);
resnet50_BatchNormInference_492_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Relu_493_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Reshape_494_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+6422528);
resnet50_Convolution_495_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+10616832);
resnet50_BatchNormInference_496_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+77070336);
resnet50_Relu_498_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Sum_499_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Divide_501_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
resnet50_Dot_502_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
resnet50_Broadcast_503_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+256256);
resnet50_Add_504_0 = (float*)(resnet50_group_0_CUDA_GPU0_allocator_memory_pool+0);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&resnet50_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&resnet50_cudnn_handle_0));
 // name=Constant_272
resnet50_Constant_float_cuda_Constant_272(0, resnet50_Constant_272_0);
 // name=cg/conv0/conv2d/kernel
resnet50_Constant_float_cuda_Constant_3(0, resnet50_Constant_3_0);
 // name=cg/conv0/batchnorm0/moving_variance
resnet50_Constant_float_cuda_Constant_7(0, resnet50_Constant_7_0);
 // name=cg/conv0/batchnorm0/moving_mean
resnet50_Constant_float_cuda_Constant_6(0, resnet50_Constant_6_0);
 // name=cg/conv0/batchnorm0/gamma
resnet50_Constant_float_cuda_Constant_4(0, resnet50_Constant_4_0);
 // name=cg/conv0/batchnorm0/beta
resnet50_Constant_float_cuda_Constant_5(0, resnet50_Constant_5_0);
 // name=cg/resnet_v10/conv2/conv2d/kernel
resnet50_Constant_float_cuda_Constant_13(0, resnet50_Constant_13_0);
 // name=cg/resnet_v10/conv2/batchnorm2/moving_variance
resnet50_Constant_float_cuda_Constant_17(0, resnet50_Constant_17_0);
 // name=cg/resnet_v10/conv2/batchnorm2/moving_mean
resnet50_Constant_float_cuda_Constant_16(0, resnet50_Constant_16_0);
 // name=cg/resnet_v10/conv2/batchnorm2/gamma
resnet50_Constant_float_cuda_Constant_14(0, resnet50_Constant_14_0);
 // name=cg/resnet_v10/conv2/batchnorm2/beta
resnet50_Constant_float_cuda_Constant_15(0, resnet50_Constant_15_0);
 // name=cg/resnet_v10/conv3/conv2d/kernel
resnet50_Constant_float_cuda_Constant_18(0, resnet50_Constant_18_0);
 // name=cg/resnet_v10/conv3/batchnorm3/moving_variance
resnet50_Constant_float_cuda_Constant_22(0, resnet50_Constant_22_0);
 // name=cg/resnet_v10/conv3/batchnorm3/moving_mean
resnet50_Constant_float_cuda_Constant_21(0, resnet50_Constant_21_0);
 // name=cg/resnet_v10/conv3/batchnorm3/gamma
resnet50_Constant_float_cuda_Constant_19(0, resnet50_Constant_19_0);
 // name=cg/resnet_v10/conv3/batchnorm3/beta
resnet50_Constant_float_cuda_Constant_20(0, resnet50_Constant_20_0);
 // name=cg/resnet_v10/conv4/conv2d/kernel
resnet50_Constant_float_cuda_Constant_23(0, resnet50_Constant_23_0);
 // name=cg/resnet_v10/conv4/batchnorm4/moving_variance
resnet50_Constant_float_cuda_Constant_27(0, resnet50_Constant_27_0);
 // name=cg/resnet_v10/conv4/batchnorm4/moving_mean
resnet50_Constant_float_cuda_Constant_26(0, resnet50_Constant_26_0);
 // name=cg/resnet_v10/conv4/batchnorm4/gamma
resnet50_Constant_float_cuda_Constant_24(0, resnet50_Constant_24_0);
 // name=cg/resnet_v10/conv4/batchnorm4/beta
resnet50_Constant_float_cuda_Constant_25(0, resnet50_Constant_25_0);
 // name=cg/resnet_v10/conv1/conv2d/kernel
resnet50_Constant_float_cuda_Constant_8(0, resnet50_Constant_8_0);
 // name=cg/resnet_v10/conv1/batchnorm1/moving_variance
resnet50_Constant_float_cuda_Constant_12(0, resnet50_Constant_12_0);
 // name=cg/resnet_v10/conv1/batchnorm1/moving_mean
resnet50_Constant_float_cuda_Constant_11(0, resnet50_Constant_11_0);
 // name=cg/resnet_v10/conv1/batchnorm1/gamma
resnet50_Constant_float_cuda_Constant_9(0, resnet50_Constant_9_0);
 // name=cg/resnet_v10/conv1/batchnorm1/beta
resnet50_Constant_float_cuda_Constant_10(0, resnet50_Constant_10_0);
 // name=cg/resnet_v11/conv5/conv2d/kernel
resnet50_Constant_float_cuda_Constant_28(0, resnet50_Constant_28_0);
 // name=cg/resnet_v11/conv5/batchnorm5/moving_variance
resnet50_Constant_float_cuda_Constant_32(0, resnet50_Constant_32_0);
 // name=cg/resnet_v11/conv5/batchnorm5/moving_mean
resnet50_Constant_float_cuda_Constant_31(0, resnet50_Constant_31_0);
 // name=cg/resnet_v11/conv5/batchnorm5/gamma
resnet50_Constant_float_cuda_Constant_29(0, resnet50_Constant_29_0);
 // name=cg/resnet_v11/conv5/batchnorm5/beta
resnet50_Constant_float_cuda_Constant_30(0, resnet50_Constant_30_0);
 // name=cg/resnet_v11/conv6/conv2d/kernel
resnet50_Constant_float_cuda_Constant_33(0, resnet50_Constant_33_0);
 // name=cg/resnet_v11/conv6/batchnorm6/moving_variance
resnet50_Constant_float_cuda_Constant_37(0, resnet50_Constant_37_0);
 // name=cg/resnet_v11/conv6/batchnorm6/moving_mean
resnet50_Constant_float_cuda_Constant_36(0, resnet50_Constant_36_0);
 // name=cg/resnet_v11/conv6/batchnorm6/gamma
resnet50_Constant_float_cuda_Constant_34(0, resnet50_Constant_34_0);
 // name=cg/resnet_v11/conv6/batchnorm6/beta
resnet50_Constant_float_cuda_Constant_35(0, resnet50_Constant_35_0);
 // name=cg/resnet_v11/conv7/conv2d/kernel
resnet50_Constant_float_cuda_Constant_38(0, resnet50_Constant_38_0);
 // name=cg/resnet_v11/conv7/batchnorm7/moving_variance
resnet50_Constant_float_cuda_Constant_42(0, resnet50_Constant_42_0);
 // name=cg/resnet_v11/conv7/batchnorm7/moving_mean
resnet50_Constant_float_cuda_Constant_41(0, resnet50_Constant_41_0);
 // name=cg/resnet_v11/conv7/batchnorm7/gamma
resnet50_Constant_float_cuda_Constant_39(0, resnet50_Constant_39_0);
 // name=cg/resnet_v11/conv7/batchnorm7/beta
resnet50_Constant_float_cuda_Constant_40(0, resnet50_Constant_40_0);
 // name=cg/resnet_v12/conv8/conv2d/kernel
resnet50_Constant_float_cuda_Constant_43(0, resnet50_Constant_43_0);
 // name=cg/resnet_v12/conv8/batchnorm8/moving_variance
resnet50_Constant_float_cuda_Constant_47(0, resnet50_Constant_47_0);
 // name=cg/resnet_v12/conv8/batchnorm8/moving_mean
resnet50_Constant_float_cuda_Constant_46(0, resnet50_Constant_46_0);
 // name=cg/resnet_v12/conv8/batchnorm8/gamma
resnet50_Constant_float_cuda_Constant_44(0, resnet50_Constant_44_0);
 // name=cg/resnet_v12/conv8/batchnorm8/beta
resnet50_Constant_float_cuda_Constant_45(0, resnet50_Constant_45_0);
 // name=cg/resnet_v12/conv9/conv2d/kernel
resnet50_Constant_float_cuda_Constant_48(0, resnet50_Constant_48_0);
 // name=cg/resnet_v12/conv9/batchnorm9/moving_variance
resnet50_Constant_float_cuda_Constant_52(0, resnet50_Constant_52_0);
 // name=cg/resnet_v12/conv9/batchnorm9/moving_mean
resnet50_Constant_float_cuda_Constant_51(0, resnet50_Constant_51_0);
 // name=cg/resnet_v12/conv9/batchnorm9/gamma
resnet50_Constant_float_cuda_Constant_49(0, resnet50_Constant_49_0);
 // name=cg/resnet_v12/conv9/batchnorm9/beta
resnet50_Constant_float_cuda_Constant_50(0, resnet50_Constant_50_0);
 // name=cg/resnet_v12/conv10/conv2d/kernel
resnet50_Constant_float_cuda_Constant_53(0, resnet50_Constant_53_0);
 // name=cg/resnet_v12/conv10/batchnorm10/moving_variance
resnet50_Constant_float_cuda_Constant_57(0, resnet50_Constant_57_0);
 // name=cg/resnet_v12/conv10/batchnorm10/moving_mean
resnet50_Constant_float_cuda_Constant_56(0, resnet50_Constant_56_0);
 // name=cg/resnet_v12/conv10/batchnorm10/gamma
resnet50_Constant_float_cuda_Constant_54(0, resnet50_Constant_54_0);
 // name=cg/resnet_v12/conv10/batchnorm10/beta
resnet50_Constant_float_cuda_Constant_55(0, resnet50_Constant_55_0);
 // name=cg/resnet_v13/conv12/conv2d/kernel
resnet50_Constant_float_cuda_Constant_63(0, resnet50_Constant_63_0);
 // name=cg/resnet_v13/conv12/batchnorm12/moving_variance
resnet50_Constant_float_cuda_Constant_67(0, resnet50_Constant_67_0);
 // name=cg/resnet_v13/conv12/batchnorm12/moving_mean
resnet50_Constant_float_cuda_Constant_66(0, resnet50_Constant_66_0);
 // name=cg/resnet_v13/conv12/batchnorm12/gamma
resnet50_Constant_float_cuda_Constant_64(0, resnet50_Constant_64_0);
 // name=cg/resnet_v13/conv12/batchnorm12/beta
resnet50_Constant_float_cuda_Constant_65(0, resnet50_Constant_65_0);
 // name=cg/resnet_v13/conv13/conv2d/kernel
resnet50_Constant_float_cuda_Constant_68(0, resnet50_Constant_68_0);
 // name=cg/resnet_v13/conv13/batchnorm13/moving_variance
resnet50_Constant_float_cuda_Constant_72(0, resnet50_Constant_72_0);
 // name=cg/resnet_v13/conv13/batchnorm13/moving_mean
resnet50_Constant_float_cuda_Constant_71(0, resnet50_Constant_71_0);
 // name=cg/resnet_v13/conv13/batchnorm13/gamma
resnet50_Constant_float_cuda_Constant_69(0, resnet50_Constant_69_0);
 // name=cg/resnet_v13/conv13/batchnorm13/beta
resnet50_Constant_float_cuda_Constant_70(0, resnet50_Constant_70_0);
 // name=cg/resnet_v13/conv14/conv2d/kernel
resnet50_Constant_float_cuda_Constant_73(0, resnet50_Constant_73_0);
 // name=cg/resnet_v13/conv14/batchnorm14/moving_variance
resnet50_Constant_float_cuda_Constant_77(0, resnet50_Constant_77_0);
 // name=cg/resnet_v13/conv14/batchnorm14/moving_mean
resnet50_Constant_float_cuda_Constant_76(0, resnet50_Constant_76_0);
 // name=cg/resnet_v13/conv14/batchnorm14/gamma
resnet50_Constant_float_cuda_Constant_74(0, resnet50_Constant_74_0);
 // name=cg/resnet_v13/conv14/batchnorm14/beta
resnet50_Constant_float_cuda_Constant_75(0, resnet50_Constant_75_0);
 // name=cg/resnet_v13/conv11/conv2d/kernel
resnet50_Constant_float_cuda_Constant_58(0, resnet50_Constant_58_0);
 // name=cg/resnet_v13/conv11/batchnorm11/moving_variance
resnet50_Constant_float_cuda_Constant_62(0, resnet50_Constant_62_0);
 // name=cg/resnet_v13/conv11/batchnorm11/moving_mean
resnet50_Constant_float_cuda_Constant_61(0, resnet50_Constant_61_0);
 // name=cg/resnet_v13/conv11/batchnorm11/gamma
resnet50_Constant_float_cuda_Constant_59(0, resnet50_Constant_59_0);
 // name=cg/resnet_v13/conv11/batchnorm11/beta
resnet50_Constant_float_cuda_Constant_60(0, resnet50_Constant_60_0);
 // name=cg/resnet_v14/conv15/conv2d/kernel
resnet50_Constant_float_cuda_Constant_78(0, resnet50_Constant_78_0);
 // name=cg/resnet_v14/conv15/batchnorm15/moving_variance
resnet50_Constant_float_cuda_Constant_82(0, resnet50_Constant_82_0);
 // name=cg/resnet_v14/conv15/batchnorm15/moving_mean
resnet50_Constant_float_cuda_Constant_81(0, resnet50_Constant_81_0);
 // name=cg/resnet_v14/conv15/batchnorm15/gamma
resnet50_Constant_float_cuda_Constant_79(0, resnet50_Constant_79_0);
 // name=cg/resnet_v14/conv15/batchnorm15/beta
resnet50_Constant_float_cuda_Constant_80(0, resnet50_Constant_80_0);
 // name=cg/resnet_v14/conv16/conv2d/kernel
resnet50_Constant_float_cuda_Constant_83(0, resnet50_Constant_83_0);
 // name=cg/resnet_v14/conv16/batchnorm16/moving_variance
resnet50_Constant_float_cuda_Constant_87(0, resnet50_Constant_87_0);
 // name=cg/resnet_v14/conv16/batchnorm16/moving_mean
resnet50_Constant_float_cuda_Constant_86(0, resnet50_Constant_86_0);
 // name=cg/resnet_v14/conv16/batchnorm16/gamma
resnet50_Constant_float_cuda_Constant_84(0, resnet50_Constant_84_0);
 // name=cg/resnet_v14/conv16/batchnorm16/beta
resnet50_Constant_float_cuda_Constant_85(0, resnet50_Constant_85_0);
 // name=cg/resnet_v14/conv17/conv2d/kernel
resnet50_Constant_float_cuda_Constant_88(0, resnet50_Constant_88_0);
 // name=cg/resnet_v14/conv17/batchnorm17/moving_variance
resnet50_Constant_float_cuda_Constant_92(0, resnet50_Constant_92_0);
 // name=cg/resnet_v14/conv17/batchnorm17/moving_mean
resnet50_Constant_float_cuda_Constant_91(0, resnet50_Constant_91_0);
 // name=cg/resnet_v14/conv17/batchnorm17/gamma
resnet50_Constant_float_cuda_Constant_89(0, resnet50_Constant_89_0);
 // name=cg/resnet_v14/conv17/batchnorm17/beta
resnet50_Constant_float_cuda_Constant_90(0, resnet50_Constant_90_0);
 // name=cg/resnet_v15/conv18/conv2d/kernel
resnet50_Constant_float_cuda_Constant_93(0, resnet50_Constant_93_0);
 // name=cg/resnet_v15/conv18/batchnorm18/moving_variance
resnet50_Constant_float_cuda_Constant_97(0, resnet50_Constant_97_0);
 // name=cg/resnet_v15/conv18/batchnorm18/moving_mean
resnet50_Constant_float_cuda_Constant_96(0, resnet50_Constant_96_0);
 // name=cg/resnet_v15/conv18/batchnorm18/gamma
resnet50_Constant_float_cuda_Constant_94(0, resnet50_Constant_94_0);
 // name=cg/resnet_v15/conv18/batchnorm18/beta
resnet50_Constant_float_cuda_Constant_95(0, resnet50_Constant_95_0);
 // name=cg/resnet_v15/conv19/conv2d/kernel
resnet50_Constant_float_cuda_Constant_98(0, resnet50_Constant_98_0);
 // name=cg/resnet_v15/conv19/batchnorm19/moving_variance
resnet50_Constant_float_cuda_Constant_102(0, resnet50_Constant_102_0);
 // name=cg/resnet_v15/conv19/batchnorm19/moving_mean
resnet50_Constant_float_cuda_Constant_101(0, resnet50_Constant_101_0);
 // name=cg/resnet_v15/conv19/batchnorm19/gamma
resnet50_Constant_float_cuda_Constant_99(0, resnet50_Constant_99_0);
 // name=cg/resnet_v15/conv19/batchnorm19/beta
resnet50_Constant_float_cuda_Constant_100(0, resnet50_Constant_100_0);
 // name=cg/resnet_v15/conv20/conv2d/kernel
resnet50_Constant_float_cuda_Constant_103(0, resnet50_Constant_103_0);
 // name=cg/resnet_v15/conv20/batchnorm20/moving_variance
resnet50_Constant_float_cuda_Constant_107(0, resnet50_Constant_107_0);
 // name=cg/resnet_v15/conv20/batchnorm20/moving_mean
resnet50_Constant_float_cuda_Constant_106(0, resnet50_Constant_106_0);
 // name=cg/resnet_v15/conv20/batchnorm20/gamma
resnet50_Constant_float_cuda_Constant_104(0, resnet50_Constant_104_0);
 // name=cg/resnet_v15/conv20/batchnorm20/beta
resnet50_Constant_float_cuda_Constant_105(0, resnet50_Constant_105_0);
 // name=cg/resnet_v16/conv21/conv2d/kernel
resnet50_Constant_float_cuda_Constant_108(0, resnet50_Constant_108_0);
 // name=cg/resnet_v16/conv21/batchnorm21/moving_variance
resnet50_Constant_float_cuda_Constant_112(0, resnet50_Constant_112_0);
 // name=cg/resnet_v16/conv21/batchnorm21/moving_mean
resnet50_Constant_float_cuda_Constant_111(0, resnet50_Constant_111_0);
 // name=cg/resnet_v16/conv21/batchnorm21/gamma
resnet50_Constant_float_cuda_Constant_109(0, resnet50_Constant_109_0);
 // name=cg/resnet_v16/conv21/batchnorm21/beta
resnet50_Constant_float_cuda_Constant_110(0, resnet50_Constant_110_0);
 // name=cg/resnet_v16/conv22/conv2d/kernel
resnet50_Constant_float_cuda_Constant_113(0, resnet50_Constant_113_0);
 // name=cg/resnet_v16/conv22/batchnorm22/moving_variance
resnet50_Constant_float_cuda_Constant_117(0, resnet50_Constant_117_0);
 // name=cg/resnet_v16/conv22/batchnorm22/moving_mean
resnet50_Constant_float_cuda_Constant_116(0, resnet50_Constant_116_0);
 // name=cg/resnet_v16/conv22/batchnorm22/gamma
resnet50_Constant_float_cuda_Constant_114(0, resnet50_Constant_114_0);
 // name=cg/resnet_v16/conv22/batchnorm22/beta
resnet50_Constant_float_cuda_Constant_115(0, resnet50_Constant_115_0);
 // name=cg/resnet_v16/conv23/conv2d/kernel
resnet50_Constant_float_cuda_Constant_118(0, resnet50_Constant_118_0);
 // name=cg/resnet_v16/conv23/batchnorm23/moving_variance
resnet50_Constant_float_cuda_Constant_122(0, resnet50_Constant_122_0);
 // name=cg/resnet_v16/conv23/batchnorm23/moving_mean
resnet50_Constant_float_cuda_Constant_121(0, resnet50_Constant_121_0);
 // name=cg/resnet_v16/conv23/batchnorm23/gamma
resnet50_Constant_float_cuda_Constant_119(0, resnet50_Constant_119_0);
 // name=cg/resnet_v16/conv23/batchnorm23/beta
resnet50_Constant_float_cuda_Constant_120(0, resnet50_Constant_120_0);
 // name=cg/resnet_v17/conv25/conv2d/kernel
resnet50_Constant_float_cuda_Constant_128(0, resnet50_Constant_128_0);
 // name=cg/resnet_v17/conv25/batchnorm25/moving_variance
resnet50_Constant_float_cuda_Constant_132(0, resnet50_Constant_132_0);
 // name=cg/resnet_v17/conv25/batchnorm25/moving_mean
resnet50_Constant_float_cuda_Constant_131(0, resnet50_Constant_131_0);
 // name=cg/resnet_v17/conv25/batchnorm25/gamma
resnet50_Constant_float_cuda_Constant_129(0, resnet50_Constant_129_0);
 // name=cg/resnet_v17/conv25/batchnorm25/beta
resnet50_Constant_float_cuda_Constant_130(0, resnet50_Constant_130_0);
 // name=cg/resnet_v17/conv26/conv2d/kernel
resnet50_Constant_float_cuda_Constant_133(0, resnet50_Constant_133_0);
 // name=cg/resnet_v17/conv26/batchnorm26/moving_variance
resnet50_Constant_float_cuda_Constant_137(0, resnet50_Constant_137_0);
 // name=cg/resnet_v17/conv26/batchnorm26/moving_mean
resnet50_Constant_float_cuda_Constant_136(0, resnet50_Constant_136_0);
 // name=cg/resnet_v17/conv26/batchnorm26/gamma
resnet50_Constant_float_cuda_Constant_134(0, resnet50_Constant_134_0);
 // name=cg/resnet_v17/conv26/batchnorm26/beta
resnet50_Constant_float_cuda_Constant_135(0, resnet50_Constant_135_0);
 // name=cg/resnet_v17/conv27/conv2d/kernel
resnet50_Constant_float_cuda_Constant_138(0, resnet50_Constant_138_0);
 // name=cg/resnet_v17/conv27/batchnorm27/moving_variance
resnet50_Constant_float_cuda_Constant_142(0, resnet50_Constant_142_0);
 // name=cg/resnet_v17/conv27/batchnorm27/moving_mean
resnet50_Constant_float_cuda_Constant_141(0, resnet50_Constant_141_0);
 // name=cg/resnet_v17/conv27/batchnorm27/gamma
resnet50_Constant_float_cuda_Constant_139(0, resnet50_Constant_139_0);
 // name=cg/resnet_v17/conv27/batchnorm27/beta
resnet50_Constant_float_cuda_Constant_140(0, resnet50_Constant_140_0);
 // name=cg/resnet_v17/conv24/conv2d/kernel
resnet50_Constant_float_cuda_Constant_123(0, resnet50_Constant_123_0);
 // name=cg/resnet_v17/conv24/batchnorm24/moving_variance
resnet50_Constant_float_cuda_Constant_127(0, resnet50_Constant_127_0);
 // name=cg/resnet_v17/conv24/batchnorm24/moving_mean
resnet50_Constant_float_cuda_Constant_126(0, resnet50_Constant_126_0);
 // name=cg/resnet_v17/conv24/batchnorm24/gamma
resnet50_Constant_float_cuda_Constant_124(0, resnet50_Constant_124_0);
 // name=cg/resnet_v17/conv24/batchnorm24/beta
resnet50_Constant_float_cuda_Constant_125(0, resnet50_Constant_125_0);
 // name=cg/resnet_v18/conv28/conv2d/kernel
resnet50_Constant_float_cuda_Constant_143(0, resnet50_Constant_143_0);
 // name=cg/resnet_v18/conv28/batchnorm28/moving_variance
resnet50_Constant_float_cuda_Constant_147(0, resnet50_Constant_147_0);
 // name=cg/resnet_v18/conv28/batchnorm28/moving_mean
resnet50_Constant_float_cuda_Constant_146(0, resnet50_Constant_146_0);
 // name=cg/resnet_v18/conv28/batchnorm28/gamma
resnet50_Constant_float_cuda_Constant_144(0, resnet50_Constant_144_0);
 // name=cg/resnet_v18/conv28/batchnorm28/beta
resnet50_Constant_float_cuda_Constant_145(0, resnet50_Constant_145_0);
 // name=cg/resnet_v18/conv29/conv2d/kernel
resnet50_Constant_float_cuda_Constant_148(0, resnet50_Constant_148_0);
 // name=cg/resnet_v18/conv29/batchnorm29/moving_variance
resnet50_Constant_float_cuda_Constant_152(0, resnet50_Constant_152_0);
 // name=cg/resnet_v18/conv29/batchnorm29/moving_mean
resnet50_Constant_float_cuda_Constant_151(0, resnet50_Constant_151_0);
 // name=cg/resnet_v18/conv29/batchnorm29/gamma
resnet50_Constant_float_cuda_Constant_149(0, resnet50_Constant_149_0);
 // name=cg/resnet_v18/conv29/batchnorm29/beta
resnet50_Constant_float_cuda_Constant_150(0, resnet50_Constant_150_0);
 // name=cg/resnet_v18/conv30/conv2d/kernel
resnet50_Constant_float_cuda_Constant_153(0, resnet50_Constant_153_0);
 // name=cg/resnet_v18/conv30/batchnorm30/moving_variance
resnet50_Constant_float_cuda_Constant_157(0, resnet50_Constant_157_0);
 // name=cg/resnet_v18/conv30/batchnorm30/moving_mean
resnet50_Constant_float_cuda_Constant_156(0, resnet50_Constant_156_0);
 // name=cg/resnet_v18/conv30/batchnorm30/gamma
resnet50_Constant_float_cuda_Constant_154(0, resnet50_Constant_154_0);
 // name=cg/resnet_v18/conv30/batchnorm30/beta
resnet50_Constant_float_cuda_Constant_155(0, resnet50_Constant_155_0);
 // name=cg/resnet_v19/conv31/conv2d/kernel
resnet50_Constant_float_cuda_Constant_158(0, resnet50_Constant_158_0);
 // name=cg/resnet_v19/conv31/batchnorm31/moving_variance
resnet50_Constant_float_cuda_Constant_162(0, resnet50_Constant_162_0);
 // name=cg/resnet_v19/conv31/batchnorm31/moving_mean
resnet50_Constant_float_cuda_Constant_161(0, resnet50_Constant_161_0);
 // name=cg/resnet_v19/conv31/batchnorm31/gamma
resnet50_Constant_float_cuda_Constant_159(0, resnet50_Constant_159_0);
 // name=cg/resnet_v19/conv31/batchnorm31/beta
resnet50_Constant_float_cuda_Constant_160(0, resnet50_Constant_160_0);
 // name=cg/resnet_v19/conv32/conv2d/kernel
resnet50_Constant_float_cuda_Constant_163(0, resnet50_Constant_163_0);
 // name=cg/resnet_v19/conv32/batchnorm32/moving_variance
resnet50_Constant_float_cuda_Constant_167(0, resnet50_Constant_167_0);
 // name=cg/resnet_v19/conv32/batchnorm32/moving_mean
resnet50_Constant_float_cuda_Constant_166(0, resnet50_Constant_166_0);
 // name=cg/resnet_v19/conv32/batchnorm32/gamma
resnet50_Constant_float_cuda_Constant_164(0, resnet50_Constant_164_0);
 // name=cg/resnet_v19/conv32/batchnorm32/beta
resnet50_Constant_float_cuda_Constant_165(0, resnet50_Constant_165_0);
 // name=cg/resnet_v19/conv33/conv2d/kernel
resnet50_Constant_float_cuda_Constant_168(0, resnet50_Constant_168_0);
 // name=cg/resnet_v19/conv33/batchnorm33/moving_variance
resnet50_Constant_float_cuda_Constant_172(0, resnet50_Constant_172_0);
 // name=cg/resnet_v19/conv33/batchnorm33/moving_mean
resnet50_Constant_float_cuda_Constant_171(0, resnet50_Constant_171_0);
 // name=cg/resnet_v19/conv33/batchnorm33/gamma
resnet50_Constant_float_cuda_Constant_169(0, resnet50_Constant_169_0);
 // name=cg/resnet_v19/conv33/batchnorm33/beta
resnet50_Constant_float_cuda_Constant_170(0, resnet50_Constant_170_0);
 // name=cg/resnet_v110/conv34/conv2d/kernel
resnet50_Constant_float_cuda_Constant_173(0, resnet50_Constant_173_0);
 // name=cg/resnet_v110/conv34/batchnorm34/moving_variance
resnet50_Constant_float_cuda_Constant_177(0, resnet50_Constant_177_0);
 // name=cg/resnet_v110/conv34/batchnorm34/moving_mean
resnet50_Constant_float_cuda_Constant_176(0, resnet50_Constant_176_0);
 // name=cg/resnet_v110/conv34/batchnorm34/gamma
resnet50_Constant_float_cuda_Constant_174(0, resnet50_Constant_174_0);
 // name=cg/resnet_v110/conv34/batchnorm34/beta
resnet50_Constant_float_cuda_Constant_175(0, resnet50_Constant_175_0);
 // name=cg/resnet_v110/conv35/conv2d/kernel
resnet50_Constant_float_cuda_Constant_178(0, resnet50_Constant_178_0);
 // name=cg/resnet_v110/conv35/batchnorm35/moving_variance
resnet50_Constant_float_cuda_Constant_182(0, resnet50_Constant_182_0);
 // name=cg/resnet_v110/conv35/batchnorm35/moving_mean
resnet50_Constant_float_cuda_Constant_181(0, resnet50_Constant_181_0);
 // name=cg/resnet_v110/conv35/batchnorm35/gamma
resnet50_Constant_float_cuda_Constant_179(0, resnet50_Constant_179_0);
 // name=cg/resnet_v110/conv35/batchnorm35/beta
resnet50_Constant_float_cuda_Constant_180(0, resnet50_Constant_180_0);
 // name=cg/resnet_v110/conv36/conv2d/kernel
resnet50_Constant_float_cuda_Constant_183(0, resnet50_Constant_183_0);
 // name=cg/resnet_v110/conv36/batchnorm36/moving_variance
resnet50_Constant_float_cuda_Constant_187(0, resnet50_Constant_187_0);
 // name=cg/resnet_v110/conv36/batchnorm36/moving_mean
resnet50_Constant_float_cuda_Constant_186(0, resnet50_Constant_186_0);
 // name=cg/resnet_v110/conv36/batchnorm36/gamma
resnet50_Constant_float_cuda_Constant_184(0, resnet50_Constant_184_0);
 // name=cg/resnet_v110/conv36/batchnorm36/beta
resnet50_Constant_float_cuda_Constant_185(0, resnet50_Constant_185_0);
 // name=cg/resnet_v111/conv37/conv2d/kernel
resnet50_Constant_float_cuda_Constant_188(0, resnet50_Constant_188_0);
 // name=cg/resnet_v111/conv37/batchnorm37/moving_variance
resnet50_Constant_float_cuda_Constant_192(0, resnet50_Constant_192_0);
 // name=cg/resnet_v111/conv37/batchnorm37/moving_mean
resnet50_Constant_float_cuda_Constant_191(0, resnet50_Constant_191_0);
 // name=cg/resnet_v111/conv37/batchnorm37/gamma
resnet50_Constant_float_cuda_Constant_189(0, resnet50_Constant_189_0);
 // name=cg/resnet_v111/conv37/batchnorm37/beta
resnet50_Constant_float_cuda_Constant_190(0, resnet50_Constant_190_0);
 // name=cg/resnet_v111/conv38/conv2d/kernel
resnet50_Constant_float_cuda_Constant_193(0, resnet50_Constant_193_0);
 // name=cg/resnet_v111/conv38/batchnorm38/moving_variance
resnet50_Constant_float_cuda_Constant_197(0, resnet50_Constant_197_0);
 // name=cg/resnet_v111/conv38/batchnorm38/moving_mean
resnet50_Constant_float_cuda_Constant_196(0, resnet50_Constant_196_0);
 // name=cg/resnet_v111/conv38/batchnorm38/gamma
resnet50_Constant_float_cuda_Constant_194(0, resnet50_Constant_194_0);
 // name=cg/resnet_v111/conv38/batchnorm38/beta
resnet50_Constant_float_cuda_Constant_195(0, resnet50_Constant_195_0);
 // name=cg/resnet_v111/conv39/conv2d/kernel
resnet50_Constant_float_cuda_Constant_198(0, resnet50_Constant_198_0);
 // name=cg/resnet_v111/conv39/batchnorm39/moving_variance
resnet50_Constant_float_cuda_Constant_202(0, resnet50_Constant_202_0);
 // name=cg/resnet_v111/conv39/batchnorm39/moving_mean
resnet50_Constant_float_cuda_Constant_201(0, resnet50_Constant_201_0);
 // name=cg/resnet_v111/conv39/batchnorm39/gamma
resnet50_Constant_float_cuda_Constant_199(0, resnet50_Constant_199_0);
 // name=cg/resnet_v111/conv39/batchnorm39/beta
resnet50_Constant_float_cuda_Constant_200(0, resnet50_Constant_200_0);
 // name=cg/resnet_v112/conv40/conv2d/kernel
resnet50_Constant_float_cuda_Constant_203(0, resnet50_Constant_203_0);
 // name=cg/resnet_v112/conv40/batchnorm40/moving_variance
resnet50_Constant_float_cuda_Constant_207(0, resnet50_Constant_207_0);
 // name=cg/resnet_v112/conv40/batchnorm40/moving_mean
resnet50_Constant_float_cuda_Constant_206(0, resnet50_Constant_206_0);
 // name=cg/resnet_v112/conv40/batchnorm40/gamma
resnet50_Constant_float_cuda_Constant_204(0, resnet50_Constant_204_0);
 // name=cg/resnet_v112/conv40/batchnorm40/beta
resnet50_Constant_float_cuda_Constant_205(0, resnet50_Constant_205_0);
 // name=cg/resnet_v112/conv41/conv2d/kernel
resnet50_Constant_float_cuda_Constant_208(0, resnet50_Constant_208_0);
 // name=cg/resnet_v112/conv41/batchnorm41/moving_variance
resnet50_Constant_float_cuda_Constant_212(0, resnet50_Constant_212_0);
 // name=cg/resnet_v112/conv41/batchnorm41/moving_mean
resnet50_Constant_float_cuda_Constant_211(0, resnet50_Constant_211_0);
 // name=cg/resnet_v112/conv41/batchnorm41/gamma
resnet50_Constant_float_cuda_Constant_209(0, resnet50_Constant_209_0);
 // name=cg/resnet_v112/conv41/batchnorm41/beta
resnet50_Constant_float_cuda_Constant_210(0, resnet50_Constant_210_0);
 // name=cg/resnet_v112/conv42/conv2d/kernel
resnet50_Constant_float_cuda_Constant_213(0, resnet50_Constant_213_0);
 // name=cg/resnet_v112/conv42/batchnorm42/moving_variance
resnet50_Constant_float_cuda_Constant_217(0, resnet50_Constant_217_0);
 // name=cg/resnet_v112/conv42/batchnorm42/moving_mean
resnet50_Constant_float_cuda_Constant_216(0, resnet50_Constant_216_0);
 // name=cg/resnet_v112/conv42/batchnorm42/gamma
resnet50_Constant_float_cuda_Constant_214(0, resnet50_Constant_214_0);
 // name=cg/resnet_v112/conv42/batchnorm42/beta
resnet50_Constant_float_cuda_Constant_215(0, resnet50_Constant_215_0);
 // name=cg/resnet_v113/conv44/conv2d/kernel
resnet50_Constant_float_cuda_Constant_223(0, resnet50_Constant_223_0);
 // name=cg/resnet_v113/conv44/batchnorm44/moving_variance
resnet50_Constant_float_cuda_Constant_227(0, resnet50_Constant_227_0);
 // name=cg/resnet_v113/conv44/batchnorm44/moving_mean
resnet50_Constant_float_cuda_Constant_226(0, resnet50_Constant_226_0);
 // name=cg/resnet_v113/conv44/batchnorm44/gamma
resnet50_Constant_float_cuda_Constant_224(0, resnet50_Constant_224_0);
 // name=cg/resnet_v113/conv44/batchnorm44/beta
resnet50_Constant_float_cuda_Constant_225(0, resnet50_Constant_225_0);
 // name=cg/resnet_v113/conv45/conv2d/kernel
resnet50_Constant_float_cuda_Constant_228(0, resnet50_Constant_228_0);
 // name=cg/resnet_v113/conv45/batchnorm45/moving_variance
resnet50_Constant_float_cuda_Constant_232(0, resnet50_Constant_232_0);
 // name=cg/resnet_v113/conv45/batchnorm45/moving_mean
resnet50_Constant_float_cuda_Constant_231(0, resnet50_Constant_231_0);
 // name=cg/resnet_v113/conv45/batchnorm45/gamma
resnet50_Constant_float_cuda_Constant_229(0, resnet50_Constant_229_0);
 // name=cg/resnet_v113/conv45/batchnorm45/beta
resnet50_Constant_float_cuda_Constant_230(0, resnet50_Constant_230_0);
 // name=cg/resnet_v113/conv46/conv2d/kernel
resnet50_Constant_float_cuda_Constant_233(0, resnet50_Constant_233_0);
 // name=cg/resnet_v113/conv46/batchnorm46/moving_variance
resnet50_Constant_float_cuda_Constant_237(0, resnet50_Constant_237_0);
 // name=cg/resnet_v113/conv46/batchnorm46/moving_mean
resnet50_Constant_float_cuda_Constant_236(0, resnet50_Constant_236_0);
 // name=cg/resnet_v113/conv46/batchnorm46/gamma
resnet50_Constant_float_cuda_Constant_234(0, resnet50_Constant_234_0);
 // name=cg/resnet_v113/conv46/batchnorm46/beta
resnet50_Constant_float_cuda_Constant_235(0, resnet50_Constant_235_0);
 // name=cg/resnet_v113/conv43/conv2d/kernel
resnet50_Constant_float_cuda_Constant_218(0, resnet50_Constant_218_0);
 // name=cg/resnet_v113/conv43/batchnorm43/moving_variance
resnet50_Constant_float_cuda_Constant_222(0, resnet50_Constant_222_0);
 // name=cg/resnet_v113/conv43/batchnorm43/moving_mean
resnet50_Constant_float_cuda_Constant_221(0, resnet50_Constant_221_0);
 // name=cg/resnet_v113/conv43/batchnorm43/gamma
resnet50_Constant_float_cuda_Constant_219(0, resnet50_Constant_219_0);
 // name=cg/resnet_v113/conv43/batchnorm43/beta
resnet50_Constant_float_cuda_Constant_220(0, resnet50_Constant_220_0);
 // name=cg/resnet_v114/conv47/conv2d/kernel
resnet50_Constant_float_cuda_Constant_238(0, resnet50_Constant_238_0);
 // name=cg/resnet_v114/conv47/batchnorm47/moving_variance
resnet50_Constant_float_cuda_Constant_242(0, resnet50_Constant_242_0);
 // name=cg/resnet_v114/conv47/batchnorm47/moving_mean
resnet50_Constant_float_cuda_Constant_241(0, resnet50_Constant_241_0);
 // name=cg/resnet_v114/conv47/batchnorm47/gamma
resnet50_Constant_float_cuda_Constant_239(0, resnet50_Constant_239_0);
 // name=cg/resnet_v114/conv47/batchnorm47/beta
resnet50_Constant_float_cuda_Constant_240(0, resnet50_Constant_240_0);
 // name=cg/resnet_v114/conv48/conv2d/kernel
resnet50_Constant_float_cuda_Constant_243(0, resnet50_Constant_243_0);
 // name=cg/resnet_v114/conv48/batchnorm48/moving_variance
resnet50_Constant_float_cuda_Constant_247(0, resnet50_Constant_247_0);
 // name=cg/resnet_v114/conv48/batchnorm48/moving_mean
resnet50_Constant_float_cuda_Constant_246(0, resnet50_Constant_246_0);
 // name=cg/resnet_v114/conv48/batchnorm48/gamma
resnet50_Constant_float_cuda_Constant_244(0, resnet50_Constant_244_0);
 // name=cg/resnet_v114/conv48/batchnorm48/beta
resnet50_Constant_float_cuda_Constant_245(0, resnet50_Constant_245_0);
 // name=cg/resnet_v114/conv49/conv2d/kernel
resnet50_Constant_float_cuda_Constant_248(0, resnet50_Constant_248_0);
 // name=cg/resnet_v114/conv49/batchnorm49/moving_variance
resnet50_Constant_float_cuda_Constant_252(0, resnet50_Constant_252_0);
 // name=cg/resnet_v114/conv49/batchnorm49/moving_mean
resnet50_Constant_float_cuda_Constant_251(0, resnet50_Constant_251_0);
 // name=cg/resnet_v114/conv49/batchnorm49/gamma
resnet50_Constant_float_cuda_Constant_249(0, resnet50_Constant_249_0);
 // name=cg/resnet_v114/conv49/batchnorm49/beta
resnet50_Constant_float_cuda_Constant_250(0, resnet50_Constant_250_0);
 // name=cg/resnet_v115/conv50/conv2d/kernel
resnet50_Constant_float_cuda_Constant_253(0, resnet50_Constant_253_0);
 // name=cg/resnet_v115/conv50/batchnorm50/moving_variance
resnet50_Constant_float_cuda_Constant_257(0, resnet50_Constant_257_0);
 // name=cg/resnet_v115/conv50/batchnorm50/moving_mean
resnet50_Constant_float_cuda_Constant_256(0, resnet50_Constant_256_0);
 // name=cg/resnet_v115/conv50/batchnorm50/gamma
resnet50_Constant_float_cuda_Constant_254(0, resnet50_Constant_254_0);
 // name=cg/resnet_v115/conv50/batchnorm50/beta
resnet50_Constant_float_cuda_Constant_255(0, resnet50_Constant_255_0);
 // name=cg/resnet_v115/conv51/conv2d/kernel
resnet50_Constant_float_cuda_Constant_258(0, resnet50_Constant_258_0);
 // name=cg/resnet_v115/conv51/batchnorm51/moving_variance
resnet50_Constant_float_cuda_Constant_262(0, resnet50_Constant_262_0);
 // name=cg/resnet_v115/conv51/batchnorm51/moving_mean
resnet50_Constant_float_cuda_Constant_261(0, resnet50_Constant_261_0);
 // name=cg/resnet_v115/conv51/batchnorm51/gamma
resnet50_Constant_float_cuda_Constant_259(0, resnet50_Constant_259_0);
 // name=cg/resnet_v115/conv51/batchnorm51/beta
resnet50_Constant_float_cuda_Constant_260(0, resnet50_Constant_260_0);
 // name=cg/resnet_v115/conv52/conv2d/kernel
resnet50_Constant_float_cuda_Constant_263(0, resnet50_Constant_263_0);
 // name=cg/resnet_v115/conv52/batchnorm52/moving_variance
resnet50_Constant_float_cuda_Constant_267(0, resnet50_Constant_267_0);
 // name=cg/resnet_v115/conv52/batchnorm52/moving_mean
resnet50_Constant_float_cuda_Constant_266(0, resnet50_Constant_266_0);
 // name=cg/resnet_v115/conv52/batchnorm52/gamma
resnet50_Constant_float_cuda_Constant_264(0, resnet50_Constant_264_0);
 // name=cg/resnet_v115/conv52/batchnorm52/beta
resnet50_Constant_float_cuda_Constant_265(0, resnet50_Constant_265_0);
 // name=Constant_500
resnet50_Constant_float_cuda_Constant_500(0, resnet50_Constant_500_0);
 // name=cg/affine0/weights
resnet50_Constant_float_cuda_Constant_269(0, resnet50_Constant_269_0);
 // name=cg/affine0/biases
resnet50_Constant_float_cuda_Constant_270(0, resnet50_Constant_270_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&resnet50_num_SMs, cudaDevAttrMultiProcessorCount, 0));
}

// Node name:	Convolution_296
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_294_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: resnet50_Reshape_295_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: resnet50_Convolution_296_0	type: float	shape: Shape{64, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_296(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_284
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_14_0	type: float	shape: Shape{64}
//	- name: resnet50_Constant_15_0	type: float	shape: Shape{64}
//	- name: resnet50_Convolution_282_0	type: float	shape: Shape{64, 64, 56, 56}
//	- name: resnet50_Constant_16_0	type: float	shape: Shape{64}
//	- name: resnet50_Constant_17_0	type: float	shape: Shape{64}
// Output:
//	- name: resnet50_BatchNormInference_284_0	type: float	shape: Shape{64, 64, 56, 56}
extern "C" __launch_bounds__(512) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 56 * 56;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 56 * 56; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_274
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_3_0	type: float	shape: Shape{7, 7, 3, 64}
// Output:
//	- name: resnet50_Reshape_274_0	type: float	shape: Shape{64, 3, 7, 7}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_274(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_274_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_274<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_334
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_74_0	type: float	shape: Shape{512}
//	- name: resnet50_Constant_75_0	type: float	shape: Shape{512}
//	- name: resnet50_Convolution_333_0	type: float	shape: Shape{64, 512, 28, 28}
//	- name: resnet50_Constant_76_0	type: float	shape: Shape{512}
//	- name: resnet50_Constant_77_0	type: float	shape: Shape{512}
// Output:
//	- name: resnet50_BatchNormInference_334_0	type: float	shape: Shape{64, 512, 28, 28}
extern "C" __launch_bounds__(512) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 28 * 28;
    const int c_id = blockIdx.x % 512;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 28 * 28; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Pad_273
// Description:	Pad
// Input:
//	- name: resnet50_Reshape_271_0	type: float	shape: Shape{64, 3, 224, 224}
//	- name: resnet50_Constant_272_0	type: float	shape: Shape{}
// Output:
//	- name: resnet50_Pad_273_0	type: float	shape: Shape{64, 3, 230, 230}
extern "C" __launch_bounds__(64) __global__ void resnet50_Pad_float_float_float_cuda_Pad_273(float* input0, float* input1, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* in = input0;
    float* pad = input1;
    float* out = output0;
    if (tid < 10156800)
    {
        size_t input_shape0 = 64;
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
extern void resnet50_Pad_float_float_float_cuda_Pad_273_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Pad_float_float_float_cuda_Pad_273<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_275
// Description:	Convolution
// Input:
//	- name: resnet50_Pad_273_0	type: float	shape: Shape{64, 3, 230, 230}
//	- name: resnet50_Reshape_274_0	type: float	shape: Shape{64, 3, 7, 7}
// Output:
//	- name: resnet50_Convolution_275_0	type: float	shape: Shape{64, 64, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_275(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 3, 230, 230));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 7, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_378
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_128_0	type: float	shape: Shape{1, 1, 512, 256}
// Output:
//	- name: resnet50_Reshape_378_0	type: float	shape: Shape{256, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_378(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_378_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_378<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_328
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_68_0	type: float	shape: Shape{3, 3, 128, 128}
// Output:
//	- name: resnet50_Reshape_328_0	type: float	shape: Shape{128, 128, 3, 3}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_328(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_328_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_328<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_392
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_143_0	type: float	shape: Shape{1, 1, 1024, 256}
// Output:
//	- name: resnet50_Reshape_392_0	type: float	shape: Shape{256, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_392(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_392<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_271
// Description:	Reshape
// Input:
//	- name: Parameter_0_0	type: float	shape: Shape{64, 224, 224, 3}
// Output:
//	- name: resnet50_Reshape_271_0	type: float	shape: Shape{64, 3, 224, 224}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_271(float* input0, float* output0)
{
    uint32_t input_strides0 = 150528;
    uint32_t input_strides1 = 3;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 150528;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 50176;
    size_t nx = 3;
    size_t ny = 50176;
    size_t nz = 64;
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
extern void resnet50_Reshape_float_float_cuda_Reshape_271_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_271<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_287
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_285_0	type: float	shape: Shape{64, 64, 56, 56}
//	- name: resnet50_Reshape_286_0	type: float	shape: Shape{64, 64, 3, 3}
// Output:
//	- name: resnet50_Convolution_287_0	type: float	shape: Shape{64, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_287(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	MaxPool_278
// Description:	MaxPool
// Input:
//	- name: resnet50_Relu_277_0	type: float	shape: Shape{64, 64, 112, 112}
// Output:
//	- name: resnet50_MaxPool_278_0	type: float	shape: Shape{64, 64, 56, 56}
void MaxPool_float_float_cuda_lib_MaxPool_278(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
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
// Node name:	Reshape_286
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_18_0	type: float	shape: Shape{3, 3, 64, 64}
// Output:
//	- name: resnet50_Reshape_286_0	type: float	shape: Shape{64, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_286(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_286_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_286<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: resnet50_BatchNormInference_283_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: resnet50_BatchNormInference_292_0	type: float	shape: Shape{64, 256, 56, 56}
// Output:
//	- name: resnet50_Relu_294_0	type: float	shape: Shape{64, 256, 56, 56}
// Fused functions:
// Add_float_float_float_cuda_Add_293<<<dim3(100352, 1, 1), dim3(512, 1, 1), 0, 0>>>(resnet50_BatchNormInference_283_0, resnet50_BatchNormInference_292_0, Add_293_0);
// Relu_float_float_cuda_Relu_294<<<dim3(100352, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_293_0, resnet50_Relu_294_0);
extern "C" __launch_bounds__(512) __global__ void resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output0[tid] = temp1;

}
extern void resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_473
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_238_0	type: float	shape: Shape{1, 1, 2048, 512}
// Output:
//	- name: resnet50_Reshape_473_0	type: float	shape: Shape{512, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_473(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_473_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_473<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_329
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_327_0	type: float	shape: Shape{64, 128, 28, 28}
//	- name: resnet50_Reshape_328_0	type: float	shape: Shape{128, 128, 3, 3}
// Output:
//	- name: resnet50_Convolution_329_0	type: float	shape: Shape{64, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_329(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_389
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_139_0	type: float	shape: Shape{1024}
//	- name: resnet50_Constant_140_0	type: float	shape: Shape{1024}
//	- name: resnet50_Convolution_388_0	type: float	shape: Shape{64, 1024, 14, 14}
//	- name: resnet50_Constant_141_0	type: float	shape: Shape{1024}
//	- name: resnet50_Constant_142_0	type: float	shape: Shape{1024}
// Output:
//	- name: resnet50_BatchNormInference_389_0	type: float	shape: Shape{64, 1024, 14, 14}
extern "C" __launch_bounds__(196) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 14 * 14;
    const int c_id = blockIdx.x % 1024;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 14 * 14; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Result_505
// Description:	Result
// Input:
//	- name: resnet50_Add_504_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Result_505_0	type: float	shape: Shape{64, 1001}
void Result_float_float_cuda_lib_Result_505(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Convolution_282
// Description:	Convolution
// Input:
//	- name: resnet50_MaxPool_278_0	type: float	shape: Shape{64, 64, 56, 56}
//	- name: resnet50_Reshape_281_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: resnet50_Convolution_282_0	type: float	shape: Shape{64, 64, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_282(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_457
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_218_0	type: float	shape: Shape{1, 1, 1024, 2048}
// Output:
//	- name: resnet50_Reshape_457_0	type: float	shape: Shape{2048, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_457(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_457_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_457<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_460
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_456_0	type: float	shape: Shape{64, 1024, 14, 14}
//	- name: resnet50_Reshape_459_0	type: float	shape: Shape{512, 1024, 1, 1}
// Output:
//	- name: resnet50_Convolution_460_0	type: float	shape: Shape{64, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_460(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_458
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_456_0	type: float	shape: Shape{64, 1024, 14, 14}
//	- name: resnet50_Reshape_457_0	type: float	shape: Shape{2048, 1024, 1, 1}
// Output:
//	- name: resnet50_Convolution_458_0	type: float	shape: Shape{64, 2048, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_458(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_459
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_223_0	type: float	shape: Shape{1, 1, 1024, 512}
// Output:
//	- name: resnet50_Reshape_459_0	type: float	shape: Shape{512, 1024, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_459(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_459_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_459<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_377
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_375_0	type: float	shape: Shape{64, 512, 28, 28}
//	- name: resnet50_Reshape_376_0	type: float	shape: Shape{1024, 512, 1, 1}
// Output:
//	- name: resnet50_Convolution_377_0	type: float	shape: Shape{64, 1024, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_377(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_469
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_467_0	type: float	shape: Shape{64, 512, 7, 7}
//	- name: resnet50_Reshape_468_0	type: float	shape: Shape{2048, 512, 1, 1}
// Output:
//	- name: resnet50_Convolution_469_0	type: float	shape: Shape{64, 2048, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_469(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_290
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_23_0	type: float	shape: Shape{1, 1, 64, 256}
// Output:
//	- name: resnet50_Reshape_290_0	type: float	shape: Shape{256, 64, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_290(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_290_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_290<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_322
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_320_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: resnet50_Reshape_321_0	type: float	shape: Shape{512, 256, 1, 1}
// Output:
//	- name: resnet50_Convolution_322_0	type: float	shape: Shape{64, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_322(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_281
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_13_0	type: float	shape: Shape{1, 1, 64, 64}
// Output:
//	- name: resnet50_Reshape_281_0	type: float	shape: Shape{64, 64, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_281(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_281_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_281<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_292
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_24_0	type: float	shape: Shape{256}
//	- name: resnet50_Constant_25_0	type: float	shape: Shape{256}
//	- name: resnet50_Convolution_291_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: resnet50_Constant_26_0	type: float	shape: Shape{256}
//	- name: resnet50_Constant_27_0	type: float	shape: Shape{256}
// Output:
//	- name: resnet50_BatchNormInference_292_0	type: float	shape: Shape{64, 256, 56, 56}
extern "C" __launch_bounds__(512) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 56 * 56;
    const int c_id = blockIdx.x % 256;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 56 * 56; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_323
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_63_0	type: float	shape: Shape{1, 1, 256, 128}
// Output:
//	- name: resnet50_Reshape_323_0	type: float	shape: Shape{128, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_323(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_323_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_323<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_502
// Description:	Dot
// Input:
//	- name: resnet50_Divide_501_0	type: float	shape: Shape{64, 2048}
//	- name: resnet50_Constant_269_0	type: float	shape: Shape{2048, 1001}
// Output:
//	- name: resnet50_Dot_502_0	type: float	shape: Shape{64, 1001}
void Dot_float_float_float_cuda_lib_Dot_502(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 64, 2048, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 2048, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Convolution_388
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_386_0	type: float	shape: Shape{64, 256, 14, 14}
//	- name: resnet50_Reshape_387_0	type: float	shape: Shape{1024, 256, 1, 1}
// Output:
//	- name: resnet50_Convolution_388_0	type: float	shape: Shape{64, 1024, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_388(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_326
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_64_0	type: float	shape: Shape{128}
//	- name: resnet50_Constant_65_0	type: float	shape: Shape{128}
//	- name: resnet50_Convolution_324_0	type: float	shape: Shape{64, 128, 28, 28}
//	- name: resnet50_Constant_66_0	type: float	shape: Shape{128}
//	- name: resnet50_Constant_67_0	type: float	shape: Shape{128}
// Output:
//	- name: resnet50_BatchNormInference_326_0	type: float	shape: Shape{64, 128, 28, 28}
extern "C" __launch_bounds__(512) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 28 * 28;
    const int c_id = blockIdx.x % 128;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 28 * 28; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_333
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_331_0	type: float	shape: Shape{64, 128, 28, 28}
//	- name: resnet50_Reshape_332_0	type: float	shape: Shape{512, 128, 1, 1}
// Output:
//	- name: resnet50_Convolution_333_0	type: float	shape: Shape{64, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_333(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 128, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Add_504
// Description:	Add
// Input:
//	- name: resnet50_Dot_502_0	type: float	shape: Shape{64, 1001}
//	- name: resnet50_Broadcast_503_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: resnet50_Add_504_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void resnet50_Add_float_float_float_cuda_Add_504(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 64 + threadIdx.x] = add(input0[blockIdx.x * 64 + threadIdx.x], input1[blockIdx.x * 64 + threadIdx.x]);

}
extern void resnet50_Add_float_float_float_cuda_Add_504_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Add_float_float_float_cuda_Add_504<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_464
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_228_0	type: float	shape: Shape{3, 3, 512, 512}
// Output:
//	- name: resnet50_Reshape_464_0	type: float	shape: Shape{512, 512, 3, 3}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_464(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_464_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_464<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_462
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_224_0	type: float	shape: Shape{512}
//	- name: resnet50_Constant_225_0	type: float	shape: Shape{512}
//	- name: resnet50_Convolution_460_0	type: float	shape: Shape{64, 512, 7, 7}
//	- name: resnet50_Constant_226_0	type: float	shape: Shape{512}
//	- name: resnet50_Constant_227_0	type: float	shape: Shape{512}
// Output:
//	- name: resnet50_BatchNormInference_462_0	type: float	shape: Shape{64, 512, 7, 7}
extern "C" __launch_bounds__(49) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 7 * 7;
    const int c_id = blockIdx.x % 512;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 7 * 7; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_321
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_58_0	type: float	shape: Shape{1, 1, 256, 512}
// Output:
//	- name: resnet50_Reshape_321_0	type: float	shape: Shape{512, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_321(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_321_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_321<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_474
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_472_0	type: float	shape: Shape{64, 2048, 7, 7}
//	- name: resnet50_Reshape_473_0	type: float	shape: Shape{512, 2048, 1, 1}
// Output:
//	- name: resnet50_Convolution_474_0	type: float	shape: Shape{64, 512, 7, 7}
void Convolution_float_float_float_cuda_lib_Convolution_474(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 2048, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_337
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_78_0	type: float	shape: Shape{1, 1, 512, 128}
// Output:
//	- name: resnet50_Reshape_337_0	type: float	shape: Shape{128, 512, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_337(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_337_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_337<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_338
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_336_0	type: float	shape: Shape{64, 512, 28, 28}
//	- name: resnet50_Reshape_337_0	type: float	shape: Shape{128, 512, 1, 1}
// Output:
//	- name: resnet50_Convolution_338_0	type: float	shape: Shape{64, 128, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_338(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Relu_277
// Description:	Relu
// Input:
//	- name: resnet50_BatchNormInference_276_0	type: float	shape: Shape{64, 64, 112, 112}
// Output:
//	- name: resnet50_Relu_277_0	type: float	shape: Shape{64, 64, 112, 112}
extern "C" __launch_bounds__(512) __global__ void resnet50_Relu_float_float_cuda_Relu_277(float* input0, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern void resnet50_Relu_float_float_cuda_Relu_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Relu_float_float_cuda_Relu_277<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_503
// Description:	Broadcast
// Input:
//	- name: resnet50_Constant_270_0	type: float	shape: Shape{1001}
// Output:
//	- name: resnet50_Broadcast_503_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void resnet50_Broadcast_float_float_cuda_Broadcast_503(float* input0, float* output0)
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
extern void resnet50_Broadcast_float_float_cuda_Broadcast_503_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Broadcast_float_float_cuda_Broadcast_503<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_379
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_375_0	type: float	shape: Shape{64, 512, 28, 28}
//	- name: resnet50_Reshape_378_0	type: float	shape: Shape{256, 512, 1, 1}
// Output:
//	- name: resnet50_Convolution_379_0	type: float	shape: Shape{64, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_379(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Sum_499
// Description:	Sum
// Input:
//	- name: resnet50_Relu_498_0	type: float	shape: Shape{64, 2048, 7, 7}
// Output:
//	- name: resnet50_Sum_499_0	type: float	shape: Shape{64, 2048}
extern "C" __launch_bounds__(32) __global__ void resnet50_Sum_float_float_cuda_Sum_499(float* input0, float* output0)
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
extern void resnet50_Sum_float_float_cuda_Sum_499_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Sum_float_float_cuda_Sum_499<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_381
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_129_0	type: float	shape: Shape{256}
//	- name: resnet50_Constant_130_0	type: float	shape: Shape{256}
//	- name: resnet50_Convolution_379_0	type: float	shape: Shape{64, 256, 14, 14}
//	- name: resnet50_Constant_131_0	type: float	shape: Shape{256}
//	- name: resnet50_Constant_132_0	type: float	shape: Shape{256}
// Output:
//	- name: resnet50_BatchNormInference_381_0	type: float	shape: Shape{64, 256, 14, 14}
extern "C" __launch_bounds__(196) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 14 * 14;
    const int c_id = blockIdx.x % 256;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 14 * 14; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_383
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_133_0	type: float	shape: Shape{3, 3, 256, 256}
// Output:
//	- name: resnet50_Reshape_383_0	type: float	shape: Shape{256, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_383(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_383_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_383<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_470
// Description:	BatchNormInference
// Input:
//	- name: resnet50_Constant_234_0	type: float	shape: Shape{2048}
//	- name: resnet50_Constant_235_0	type: float	shape: Shape{2048}
//	- name: resnet50_Convolution_469_0	type: float	shape: Shape{64, 2048, 7, 7}
//	- name: resnet50_Constant_236_0	type: float	shape: Shape{2048}
//	- name: resnet50_Constant_237_0	type: float	shape: Shape{2048}
// Output:
//	- name: resnet50_BatchNormInference_470_0	type: float	shape: Shape{64, 2048, 7, 7}
extern "C" __launch_bounds__(49) __global__ void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 7 * 7;
    const int c_id = blockIdx.x % 2048;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 7 * 7; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(1.001e-05 + input4[c_id])));
    }

}
extern void resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_295
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_28_0	type: float	shape: Shape{1, 1, 256, 64}
// Output:
//	- name: resnet50_Reshape_295_0	type: float	shape: Shape{64, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_295(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_295_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_295<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_384
// Description:	Convolution
// Input:
//	- name: resnet50_Relu_382_0	type: float	shape: Shape{64, 256, 14, 14}
//	- name: resnet50_Reshape_383_0	type: float	shape: Shape{256, 256, 3, 3}
// Output:
//	- name: resnet50_Convolution_384_0	type: float	shape: Shape{64, 256, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_384(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 256, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_387
// Description:	Reshape
// Input:
//	- name: resnet50_Constant_138_0	type: float	shape: Shape{1, 1, 256, 1024}
// Output:
//	- name: resnet50_Reshape_387_0	type: float	shape: Shape{1024, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void resnet50_Reshape_float_float_cuda_Reshape_387(float* input0, float* output0)
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
extern void resnet50_Reshape_float_float_cuda_Reshape_387_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_387<<<grids, blocks, mem, stream>>>(input0, output0);
}



extern "C" void resnet50_cuda_free()
{

CUDA_SAFE_CALL(cudaFree(resnet50_group_persist_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(resnet50_group_0_CUDA_GPU0_allocator_memory_pool));
CUBLAS_SAFE_CALL(cublasDestroy(resnet50_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(resnet50_cudnn_handle_0));
}

#include "./include/dnn.h"

class resnet50_Reshape_float_float_cuda_Reshape_271_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_271_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_271_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_271_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_271<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_271_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Pad_float_float_float_cuda_Pad_273_CallKernel : public Kernel {
public:
    resnet50_Pad_float_float_float_cuda_Pad_273_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Pad_float_float_float_cuda_Pad_273_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Pad_float_float_float_cuda_Pad_273_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Pad_float_float_float_cuda_Pad_273<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Pad_float_float_float_cuda_Pad_273_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_274_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_274_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_274_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_274_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_274<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_274_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_275Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_275Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_275";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 3, 230, 230, 64, 3, 7, 7, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_275(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 3, 230, 230));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 7, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_275(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Relu_float_float_cuda_Relu_277_CallKernel : public Kernel {
public:
    resnet50_Relu_float_float_cuda_Relu_277_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Relu_float_float_cuda_Relu_277_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Relu_float_float_cuda_Relu_277_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Relu_float_float_cuda_Relu_277<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Relu_float_float_cuda_Relu_277_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_MaxPool_float_float_cuda_lib_MaxPool_278Kernel : public Kernel {
public:
    resnet50_MaxPool_float_float_cuda_lib_MaxPool_278Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_MaxPool_float_float_cuda_lib_MaxPool_278";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_278(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    // CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_278(cudnn_handle, input0, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_281_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_281_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_281_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_281_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_281<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_281_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_282Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_282Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_282";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 64, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_282(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_282(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_286_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_286_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_286_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_286_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_286<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_286_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_287Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_287Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_287";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_287(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 64, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_287(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_290_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_290_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_290<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_290_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_291";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 64, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_291(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 64, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_291(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel : public Kernel {
public:
    resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Add_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Add_Relu_0_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_295_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_295_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_295_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_295_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_295<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_295_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_296Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_296Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_296";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 256, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_296(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_296(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_323_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_323_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_323_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_323_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_323<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_323_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_324Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_324Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_324";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 256, 56, 56, 128, 256, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_324(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_324(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_328_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_328_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_328<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_328_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_329";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_329(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_329(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_332_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_332_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_332<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_332_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_333";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 128, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_333(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 128, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_333(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_321_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_321_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_321_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_321_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_321<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_321_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_322Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_322Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_322";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 256, 56, 56, 512, 256, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_322(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_322(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_337_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_337_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_337_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_337_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_337<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_337_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_338Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_338Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_338";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 512, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_338(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_338(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_378_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_378_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_378_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_378_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_378<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_378_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_379Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_379Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_379";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 512, 28, 28, 256, 512, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_379(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_379(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_383_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_383_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_383<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_383_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_384";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_384(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 256, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_384(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_387_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_387_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_387<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_387_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_388";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 256, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_388(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 256, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_388(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_376_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_376_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_376_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_376_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_376<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_376_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_377Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_377Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_377";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 512, 28, 28, 1024, 512, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_377(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1024, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_377(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_392_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_392_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_392<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_392_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_393";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 1024, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_393(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 14, 14));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_393(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_459_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_459_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_459_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_459_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_459<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_459_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_460Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_460Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_460";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 1024, 14, 14, 512, 1024, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_460(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_460(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_464_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_464_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_464_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_464_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_464<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_464_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_465Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_465Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_465";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_465(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 512, 3, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_465(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_468_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_468_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_468_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_468_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_468<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_468_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_469Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_469Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_469";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 512, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_469(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 512, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_469(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel : public Kernel {
public:
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  input2, float*  input3, float*  input4, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->input2 = input2, this->input3 = input3, this->input4 = input4, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_Call(grids, blocks, mem, stream, input0, input1, input2, input3, input4, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_457_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_457_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_457_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_457_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_457<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_457_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_458Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_458Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_458";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 1024, 14, 14, 2048, 1024, 1, 1, 0, 0, 2, 2, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_458(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 1024, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2048, 1024, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_458(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Reshape_float_float_cuda_Reshape_473_CallKernel : public Kernel {
public:
    resnet50_Reshape_float_float_cuda_Reshape_473_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Reshape_float_float_cuda_Reshape_473_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_473_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Reshape_float_float_cuda_Reshape_473<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_473_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Convolution_float_float_float_cuda_lib_Convolution_474Kernel : public Kernel {
public:
    resnet50_Convolution_float_float_float_cuda_lib_Convolution_474Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Convolution_float_float_float_cuda_lib_Convolution_474";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({64, 2048, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_474(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 2048, 7, 7));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 2048, 1, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        this->Convolution_float_float_float_cuda_lib_Convolution_474(cudnn_handle, input0, input1, output0);
    }
};


class resnet50_Sum_float_float_cuda_Sum_499_CallKernel : public Kernel {
public:
    resnet50_Sum_float_float_cuda_Sum_499_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Sum_float_float_cuda_Sum_499_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Sum_float_float_cuda_Sum_499_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Sum_float_float_cuda_Sum_499<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Sum_float_float_cuda_Sum_499_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Divide_float_float_float_cuda_Divide_501_CallKernel : public Kernel {
public:
    resnet50_Divide_float_float_float_cuda_Divide_501_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Divide_float_float_float_cuda_Divide_501_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Divide_float_float_float_cuda_Divide_501_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Divide_float_float_float_cuda_Divide_501<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Divide_float_float_float_cuda_Divide_501_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class resnet50_Dot_float_float_float_cuda_lib_Dot_502Kernel : public Kernel {
public:
    resnet50_Dot_float_float_float_cuda_lib_Dot_502Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Dot_float_float_float_cuda_lib_Dot_502";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1001;
    ret[1] = 64;
    ret[2] = 2048;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_502(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 64, 2048, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 2048, &beta, static_cast<float*>(output0), 1001));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_502(cublas_handle, input0, input1, output0);
    }
};


class resnet50_Broadcast_float_float_cuda_Broadcast_503_CallKernel : public Kernel {
public:
    resnet50_Broadcast_float_float_cuda_Broadcast_503_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Broadcast_float_float_cuda_Broadcast_503_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_503_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    resnet50_Broadcast_float_float_cuda_Broadcast_503<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_503_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class resnet50_Add_float_float_float_cuda_Add_504_CallKernel : public Kernel {
public:
    resnet50_Add_float_float_float_cuda_Add_504_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Add_float_float_float_cuda_Add_504_Call";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_504_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    resnet50_Add_float_float_float_cuda_Add_504<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Add_float_float_float_cuda_Add_504_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class resnet50_Result_float_float_cuda_lib_Result_505Kernel : public Kernel {
public:
    resnet50_Result_float_float_cuda_lib_Result_505Kernel(float*  input0, float**  output0, float*  Parameter_0_0, float**  Result_505_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_505_0 = Result_505_0;
        this->kernelName = "resnet50_Result_float_float_cuda_lib_Result_505";
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
    float*  Parameter_0_0; float**  Result_505_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_505(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        this->Result_float_float_cuda_lib_Result_505(input0, output0);
    }
};
void Resnet50::gen_vector(float*  Parameter_0_0, float**  Result_505_0) {
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_271_CallKernel(dim3(1, 3136, 64), dim3(16, 16, 1), 0, nullptr, std::move(Parameter_0_0), std::move(resnet50_Reshape_271_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Pad_float_float_float_cuda_Pad_273_CallKernel(dim3(158700, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(resnet50_Reshape_271_0), std::move(resnet50_Constant_272_0), std::move(resnet50_Pad_273_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_274_CallKernel(dim3(4, 3, 4), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_3_0), std::move(resnet50_Reshape_274_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_275Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Pad_273_0), std::move(resnet50_Reshape_274_0), std::move(resnet50_Convolution_275_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_276_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_4_0), std::move(resnet50_Constant_5_0), std::move(resnet50_Convolution_275_0), std::move(resnet50_Constant_6_0), std::move(resnet50_Constant_7_0), std::move(resnet50_BatchNormInference_276_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_276_0), std::move(resnet50_Relu_277_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_MaxPool_float_float_cuda_lib_MaxPool_278Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_277_0), std::move(resnet50_MaxPool_278_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_281_CallKernel(dim3(4, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_13_0), std::move(resnet50_Reshape_281_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_282Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_MaxPool_278_0), std::move(resnet50_Reshape_281_0), std::move(resnet50_Convolution_282_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_14_0), std::move(resnet50_Constant_15_0), std::move(resnet50_Convolution_282_0), std::move(resnet50_Constant_16_0), std::move(resnet50_Constant_17_0), std::move(resnet50_BatchNormInference_284_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_284_0), std::move(resnet50_Relu_285_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_18_0), std::move(resnet50_Reshape_286_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_285_0), std::move(resnet50_Reshape_286_0), std::move(resnet50_Convolution_287_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_19_0), std::move(resnet50_Constant_20_0), std::move(resnet50_Convolution_287_0), std::move(resnet50_Constant_21_0), std::move(resnet50_Constant_22_0), std::move(resnet50_BatchNormInference_288_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_288_0), std::move(resnet50_Relu_289_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_23_0), std::move(resnet50_Reshape_290_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_289_0), std::move(resnet50_Reshape_290_0), std::move(resnet50_Convolution_291_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(16384, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_24_0), std::move(resnet50_Constant_25_0), std::move(resnet50_Convolution_291_0), std::move(resnet50_Constant_26_0), std::move(resnet50_Constant_27_0), std::move(resnet50_BatchNormInference_292_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_8_0), std::move(resnet50_Reshape_279_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_MaxPool_278_0), std::move(resnet50_Reshape_279_0), std::move(resnet50_Convolution_280_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(16384, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_9_0), std::move(resnet50_Constant_10_0), std::move(resnet50_Convolution_280_0), std::move(resnet50_Constant_11_0), std::move(resnet50_Constant_12_0), std::move(resnet50_BatchNormInference_283_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_283_0), std::move(resnet50_BatchNormInference_292_0), std::move(resnet50_Relu_294_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_295_CallKernel(dim3(4, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_28_0), std::move(resnet50_Reshape_295_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_296Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_294_0), std::move(resnet50_Reshape_295_0), std::move(resnet50_Convolution_296_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_29_0), std::move(resnet50_Constant_30_0), std::move(resnet50_Convolution_296_0), std::move(resnet50_Constant_31_0), std::move(resnet50_Constant_32_0), std::move(resnet50_BatchNormInference_297_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_297_0), std::move(resnet50_Relu_298_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_33_0), std::move(resnet50_Reshape_299_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_298_0), std::move(resnet50_Reshape_299_0), std::move(resnet50_Convolution_300_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_34_0), std::move(resnet50_Constant_35_0), std::move(resnet50_Convolution_300_0), std::move(resnet50_Constant_36_0), std::move(resnet50_Constant_37_0), std::move(resnet50_BatchNormInference_301_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_301_0), std::move(resnet50_Relu_302_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_38_0), std::move(resnet50_Reshape_303_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_302_0), std::move(resnet50_Reshape_303_0), std::move(resnet50_Convolution_304_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(16384, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_39_0), std::move(resnet50_Constant_40_0), std::move(resnet50_Convolution_304_0), std::move(resnet50_Constant_41_0), std::move(resnet50_Constant_42_0), std::move(resnet50_BatchNormInference_305_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_294_0), std::move(resnet50_BatchNormInference_305_0), std::move(resnet50_Relu_307_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_295_CallKernel(dim3(4, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_43_0), std::move(resnet50_Reshape_308_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_296Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_307_0), std::move(resnet50_Reshape_308_0), std::move(resnet50_Convolution_309_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_44_0), std::move(resnet50_Constant_45_0), std::move(resnet50_Convolution_309_0), std::move(resnet50_Constant_46_0), std::move(resnet50_Constant_47_0), std::move(resnet50_BatchNormInference_310_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_310_0), std::move(resnet50_Relu_311_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_286_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_48_0), std::move(resnet50_Reshape_312_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_287Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_311_0), std::move(resnet50_Reshape_312_0), std::move(resnet50_Convolution_313_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_284_CallKernel(dim3(4096, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_49_0), std::move(resnet50_Constant_50_0), std::move(resnet50_Convolution_313_0), std::move(resnet50_Constant_51_0), std::move(resnet50_Constant_52_0), std::move(resnet50_BatchNormInference_314_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_314_0), std::move(resnet50_Relu_315_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_290_CallKernel(dim3(16, 4, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_53_0), std::move(resnet50_Reshape_316_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_291Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_315_0), std::move(resnet50_Reshape_316_0), std::move(resnet50_Convolution_317_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_292_CallKernel(dim3(16384, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_54_0), std::move(resnet50_Constant_55_0), std::move(resnet50_Convolution_317_0), std::move(resnet50_Constant_56_0), std::move(resnet50_Constant_57_0), std::move(resnet50_BatchNormInference_318_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_307_0), std::move(resnet50_BatchNormInference_318_0), std::move(resnet50_Relu_320_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_323_CallKernel(dim3(8, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_63_0), std::move(resnet50_Reshape_323_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_324Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_320_0), std::move(resnet50_Reshape_323_0), std::move(resnet50_Convolution_324_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_64_0), std::move(resnet50_Constant_65_0), std::move(resnet50_Convolution_324_0), std::move(resnet50_Constant_66_0), std::move(resnet50_Constant_67_0), std::move(resnet50_BatchNormInference_326_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_326_0), std::move(resnet50_Relu_327_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_68_0), std::move(resnet50_Reshape_328_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_327_0), std::move(resnet50_Reshape_328_0), std::move(resnet50_Convolution_329_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_69_0), std::move(resnet50_Constant_70_0), std::move(resnet50_Convolution_329_0), std::move(resnet50_Constant_71_0), std::move(resnet50_Constant_72_0), std::move(resnet50_BatchNormInference_330_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_330_0), std::move(resnet50_Relu_331_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_73_0), std::move(resnet50_Reshape_332_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_331_0), std::move(resnet50_Reshape_332_0), std::move(resnet50_Convolution_333_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_74_0), std::move(resnet50_Constant_75_0), std::move(resnet50_Convolution_333_0), std::move(resnet50_Constant_76_0), std::move(resnet50_Constant_77_0), std::move(resnet50_BatchNormInference_334_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_321_CallKernel(dim3(32, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_58_0), std::move(resnet50_Reshape_321_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_322Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_320_0), std::move(resnet50_Reshape_321_0), std::move(resnet50_Convolution_322_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_59_0), std::move(resnet50_Constant_60_0), std::move(resnet50_Convolution_322_0), std::move(resnet50_Constant_61_0), std::move(resnet50_Constant_62_0), std::move(resnet50_BatchNormInference_325_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_325_0), std::move(resnet50_BatchNormInference_334_0), std::move(resnet50_Relu_336_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_78_0), std::move(resnet50_Reshape_337_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_336_0), std::move(resnet50_Reshape_337_0), std::move(resnet50_Convolution_338_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_79_0), std::move(resnet50_Constant_80_0), std::move(resnet50_Convolution_338_0), std::move(resnet50_Constant_81_0), std::move(resnet50_Constant_82_0), std::move(resnet50_BatchNormInference_339_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_339_0), std::move(resnet50_Relu_340_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_83_0), std::move(resnet50_Reshape_341_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_340_0), std::move(resnet50_Reshape_341_0), std::move(resnet50_Convolution_342_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_84_0), std::move(resnet50_Constant_85_0), std::move(resnet50_Convolution_342_0), std::move(resnet50_Constant_86_0), std::move(resnet50_Constant_87_0), std::move(resnet50_BatchNormInference_343_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_343_0), std::move(resnet50_Relu_344_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_88_0), std::move(resnet50_Reshape_345_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_344_0), std::move(resnet50_Reshape_345_0), std::move(resnet50_Convolution_346_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_89_0), std::move(resnet50_Constant_90_0), std::move(resnet50_Convolution_346_0), std::move(resnet50_Constant_91_0), std::move(resnet50_Constant_92_0), std::move(resnet50_BatchNormInference_347_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_336_0), std::move(resnet50_BatchNormInference_347_0), std::move(resnet50_Relu_349_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_93_0), std::move(resnet50_Reshape_350_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_349_0), std::move(resnet50_Reshape_350_0), std::move(resnet50_Convolution_351_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_94_0), std::move(resnet50_Constant_95_0), std::move(resnet50_Convolution_351_0), std::move(resnet50_Constant_96_0), std::move(resnet50_Constant_97_0), std::move(resnet50_BatchNormInference_352_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_352_0), std::move(resnet50_Relu_353_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_98_0), std::move(resnet50_Reshape_354_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_353_0), std::move(resnet50_Reshape_354_0), std::move(resnet50_Convolution_355_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_99_0), std::move(resnet50_Constant_100_0), std::move(resnet50_Convolution_355_0), std::move(resnet50_Constant_101_0), std::move(resnet50_Constant_102_0), std::move(resnet50_BatchNormInference_356_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_356_0), std::move(resnet50_Relu_357_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_103_0), std::move(resnet50_Reshape_358_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_357_0), std::move(resnet50_Reshape_358_0), std::move(resnet50_Convolution_359_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_104_0), std::move(resnet50_Constant_105_0), std::move(resnet50_Convolution_359_0), std::move(resnet50_Constant_106_0), std::move(resnet50_Constant_107_0), std::move(resnet50_BatchNormInference_360_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_349_0), std::move(resnet50_BatchNormInference_360_0), std::move(resnet50_Relu_362_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_337_CallKernel(dim3(8, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_108_0), std::move(resnet50_Reshape_363_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_338Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_362_0), std::move(resnet50_Reshape_363_0), std::move(resnet50_Convolution_364_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_109_0), std::move(resnet50_Constant_110_0), std::move(resnet50_Convolution_364_0), std::move(resnet50_Constant_111_0), std::move(resnet50_Constant_112_0), std::move(resnet50_BatchNormInference_365_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_365_0), std::move(resnet50_Relu_366_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_328_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_113_0), std::move(resnet50_Reshape_367_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_329Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_366_0), std::move(resnet50_Reshape_367_0), std::move(resnet50_Convolution_368_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_326_CallKernel(dim3(8192, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_114_0), std::move(resnet50_Constant_115_0), std::move(resnet50_Convolution_368_0), std::move(resnet50_Constant_116_0), std::move(resnet50_Constant_117_0), std::move(resnet50_BatchNormInference_369_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_369_0), std::move(resnet50_Relu_370_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_332_CallKernel(dim3(32, 8, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_118_0), std::move(resnet50_Reshape_371_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_333Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_370_0), std::move(resnet50_Reshape_371_0), std::move(resnet50_Convolution_372_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_334_CallKernel(dim3(32768, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Constant_119_0), std::move(resnet50_Constant_120_0), std::move(resnet50_Convolution_372_0), std::move(resnet50_Constant_121_0), std::move(resnet50_Constant_122_0), std::move(resnet50_BatchNormInference_373_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_362_0), std::move(resnet50_BatchNormInference_373_0), std::move(resnet50_Relu_375_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_378_CallKernel(dim3(16, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_128_0), std::move(resnet50_Reshape_378_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_379Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_375_0), std::move(resnet50_Reshape_378_0), std::move(resnet50_Convolution_379_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_129_0), std::move(resnet50_Constant_130_0), std::move(resnet50_Convolution_379_0), std::move(resnet50_Constant_131_0), std::move(resnet50_Constant_132_0), std::move(resnet50_BatchNormInference_381_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_381_0), std::move(resnet50_Relu_382_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_133_0), std::move(resnet50_Reshape_383_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_382_0), std::move(resnet50_Reshape_383_0), std::move(resnet50_Convolution_384_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_134_0), std::move(resnet50_Constant_135_0), std::move(resnet50_Convolution_384_0), std::move(resnet50_Constant_136_0), std::move(resnet50_Constant_137_0), std::move(resnet50_BatchNormInference_385_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_385_0), std::move(resnet50_Relu_386_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_138_0), std::move(resnet50_Reshape_387_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_386_0), std::move(resnet50_Reshape_387_0), std::move(resnet50_Convolution_388_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_139_0), std::move(resnet50_Constant_140_0), std::move(resnet50_Convolution_388_0), std::move(resnet50_Constant_141_0), std::move(resnet50_Constant_142_0), std::move(resnet50_BatchNormInference_389_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_376_CallKernel(dim3(64, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_123_0), std::move(resnet50_Reshape_376_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_377Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_375_0), std::move(resnet50_Reshape_376_0), std::move(resnet50_Convolution_377_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_124_0), std::move(resnet50_Constant_125_0), std::move(resnet50_Convolution_377_0), std::move(resnet50_Constant_126_0), std::move(resnet50_Constant_127_0), std::move(resnet50_BatchNormInference_380_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_380_0), std::move(resnet50_BatchNormInference_389_0), std::move(resnet50_Relu_391_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_143_0), std::move(resnet50_Reshape_392_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_391_0), std::move(resnet50_Reshape_392_0), std::move(resnet50_Convolution_393_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_144_0), std::move(resnet50_Constant_145_0), std::move(resnet50_Convolution_393_0), std::move(resnet50_Constant_146_0), std::move(resnet50_Constant_147_0), std::move(resnet50_BatchNormInference_394_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_394_0), std::move(resnet50_Relu_395_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_148_0), std::move(resnet50_Reshape_396_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_395_0), std::move(resnet50_Reshape_396_0), std::move(resnet50_Convolution_397_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_149_0), std::move(resnet50_Constant_150_0), std::move(resnet50_Convolution_397_0), std::move(resnet50_Constant_151_0), std::move(resnet50_Constant_152_0), std::move(resnet50_BatchNormInference_398_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_398_0), std::move(resnet50_Relu_399_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_153_0), std::move(resnet50_Reshape_400_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_399_0), std::move(resnet50_Reshape_400_0), std::move(resnet50_Convolution_401_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_154_0), std::move(resnet50_Constant_155_0), std::move(resnet50_Convolution_401_0), std::move(resnet50_Constant_156_0), std::move(resnet50_Constant_157_0), std::move(resnet50_BatchNormInference_402_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_391_0), std::move(resnet50_BatchNormInference_402_0), std::move(resnet50_Relu_404_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_158_0), std::move(resnet50_Reshape_405_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_404_0), std::move(resnet50_Reshape_405_0), std::move(resnet50_Convolution_406_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_159_0), std::move(resnet50_Constant_160_0), std::move(resnet50_Convolution_406_0), std::move(resnet50_Constant_161_0), std::move(resnet50_Constant_162_0), std::move(resnet50_BatchNormInference_407_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_407_0), std::move(resnet50_Relu_408_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_163_0), std::move(resnet50_Reshape_409_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_408_0), std::move(resnet50_Reshape_409_0), std::move(resnet50_Convolution_410_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_164_0), std::move(resnet50_Constant_165_0), std::move(resnet50_Convolution_410_0), std::move(resnet50_Constant_166_0), std::move(resnet50_Constant_167_0), std::move(resnet50_BatchNormInference_411_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_411_0), std::move(resnet50_Relu_412_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_168_0), std::move(resnet50_Reshape_413_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_412_0), std::move(resnet50_Reshape_413_0), std::move(resnet50_Convolution_414_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_169_0), std::move(resnet50_Constant_170_0), std::move(resnet50_Convolution_414_0), std::move(resnet50_Constant_171_0), std::move(resnet50_Constant_172_0), std::move(resnet50_BatchNormInference_415_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_404_0), std::move(resnet50_BatchNormInference_415_0), std::move(resnet50_Relu_417_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_173_0), std::move(resnet50_Reshape_418_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_417_0), std::move(resnet50_Reshape_418_0), std::move(resnet50_Convolution_419_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_174_0), std::move(resnet50_Constant_175_0), std::move(resnet50_Convolution_419_0), std::move(resnet50_Constant_176_0), std::move(resnet50_Constant_177_0), std::move(resnet50_BatchNormInference_420_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_420_0), std::move(resnet50_Relu_421_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_178_0), std::move(resnet50_Reshape_422_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_421_0), std::move(resnet50_Reshape_422_0), std::move(resnet50_Convolution_423_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_179_0), std::move(resnet50_Constant_180_0), std::move(resnet50_Convolution_423_0), std::move(resnet50_Constant_181_0), std::move(resnet50_Constant_182_0), std::move(resnet50_BatchNormInference_424_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_424_0), std::move(resnet50_Relu_425_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_183_0), std::move(resnet50_Reshape_426_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_425_0), std::move(resnet50_Reshape_426_0), std::move(resnet50_Convolution_427_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_184_0), std::move(resnet50_Constant_185_0), std::move(resnet50_Convolution_427_0), std::move(resnet50_Constant_186_0), std::move(resnet50_Constant_187_0), std::move(resnet50_BatchNormInference_428_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_417_0), std::move(resnet50_BatchNormInference_428_0), std::move(resnet50_Relu_430_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_188_0), std::move(resnet50_Reshape_431_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_430_0), std::move(resnet50_Reshape_431_0), std::move(resnet50_Convolution_432_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_189_0), std::move(resnet50_Constant_190_0), std::move(resnet50_Convolution_432_0), std::move(resnet50_Constant_191_0), std::move(resnet50_Constant_192_0), std::move(resnet50_BatchNormInference_433_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_433_0), std::move(resnet50_Relu_434_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_193_0), std::move(resnet50_Reshape_435_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_434_0), std::move(resnet50_Reshape_435_0), std::move(resnet50_Convolution_436_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_194_0), std::move(resnet50_Constant_195_0), std::move(resnet50_Convolution_436_0), std::move(resnet50_Constant_196_0), std::move(resnet50_Constant_197_0), std::move(resnet50_BatchNormInference_437_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_437_0), std::move(resnet50_Relu_438_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_198_0), std::move(resnet50_Reshape_439_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_438_0), std::move(resnet50_Reshape_439_0), std::move(resnet50_Convolution_440_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_199_0), std::move(resnet50_Constant_200_0), std::move(resnet50_Convolution_440_0), std::move(resnet50_Constant_201_0), std::move(resnet50_Constant_202_0), std::move(resnet50_BatchNormInference_441_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_430_0), std::move(resnet50_BatchNormInference_441_0), std::move(resnet50_Relu_443_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_392_CallKernel(dim3(16, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_203_0), std::move(resnet50_Reshape_444_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_393Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_443_0), std::move(resnet50_Reshape_444_0), std::move(resnet50_Convolution_445_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_204_0), std::move(resnet50_Constant_205_0), std::move(resnet50_Convolution_445_0), std::move(resnet50_Constant_206_0), std::move(resnet50_Constant_207_0), std::move(resnet50_BatchNormInference_446_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_446_0), std::move(resnet50_Relu_447_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_383_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_208_0), std::move(resnet50_Reshape_448_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_384Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_447_0), std::move(resnet50_Reshape_448_0), std::move(resnet50_Convolution_449_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_381_CallKernel(dim3(16384, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_209_0), std::move(resnet50_Constant_210_0), std::move(resnet50_Convolution_449_0), std::move(resnet50_Constant_211_0), std::move(resnet50_Constant_212_0), std::move(resnet50_BatchNormInference_450_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_450_0), std::move(resnet50_Relu_451_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_387_CallKernel(dim3(64, 16, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_213_0), std::move(resnet50_Reshape_452_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_388Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_451_0), std::move(resnet50_Reshape_452_0), std::move(resnet50_Convolution_453_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_389_CallKernel(dim3(65536, 1, 1), dim3(196, 1, 1), 0, nullptr, std::move(resnet50_Constant_214_0), std::move(resnet50_Constant_215_0), std::move(resnet50_Convolution_453_0), std::move(resnet50_Constant_216_0), std::move(resnet50_Constant_217_0), std::move(resnet50_BatchNormInference_454_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_443_0), std::move(resnet50_BatchNormInference_454_0), std::move(resnet50_Relu_456_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_459_CallKernel(dim3(32, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_223_0), std::move(resnet50_Reshape_459_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_460Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_456_0), std::move(resnet50_Reshape_459_0), std::move(resnet50_Convolution_460_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_224_0), std::move(resnet50_Constant_225_0), std::move(resnet50_Convolution_460_0), std::move(resnet50_Constant_226_0), std::move(resnet50_Constant_227_0), std::move(resnet50_BatchNormInference_462_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_462_0), std::move(resnet50_Relu_463_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_228_0), std::move(resnet50_Reshape_464_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_463_0), std::move(resnet50_Reshape_464_0), std::move(resnet50_Convolution_465_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_229_0), std::move(resnet50_Constant_230_0), std::move(resnet50_Convolution_465_0), std::move(resnet50_Constant_231_0), std::move(resnet50_Constant_232_0), std::move(resnet50_BatchNormInference_466_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_466_0), std::move(resnet50_Relu_467_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_233_0), std::move(resnet50_Reshape_468_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_467_0), std::move(resnet50_Reshape_468_0), std::move(resnet50_Convolution_469_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(131072, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_234_0), std::move(resnet50_Constant_235_0), std::move(resnet50_Convolution_469_0), std::move(resnet50_Constant_236_0), std::move(resnet50_Constant_237_0), std::move(resnet50_BatchNormInference_470_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_457_CallKernel(dim3(128, 64, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_218_0), std::move(resnet50_Reshape_457_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_458Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_456_0), std::move(resnet50_Reshape_457_0), std::move(resnet50_Convolution_458_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(131072, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_219_0), std::move(resnet50_Constant_220_0), std::move(resnet50_Convolution_458_0), std::move(resnet50_Constant_221_0), std::move(resnet50_Constant_222_0), std::move(resnet50_BatchNormInference_461_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_461_0), std::move(resnet50_BatchNormInference_470_0), std::move(resnet50_Relu_472_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_473_CallKernel(dim3(32, 128, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_238_0), std::move(resnet50_Reshape_473_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_474Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_472_0), std::move(resnet50_Reshape_473_0), std::move(resnet50_Convolution_474_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_239_0), std::move(resnet50_Constant_240_0), std::move(resnet50_Convolution_474_0), std::move(resnet50_Constant_241_0), std::move(resnet50_Constant_242_0), std::move(resnet50_BatchNormInference_475_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_475_0), std::move(resnet50_Relu_476_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_243_0), std::move(resnet50_Reshape_477_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_476_0), std::move(resnet50_Reshape_477_0), std::move(resnet50_Convolution_478_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_244_0), std::move(resnet50_Constant_245_0), std::move(resnet50_Convolution_478_0), std::move(resnet50_Constant_246_0), std::move(resnet50_Constant_247_0), std::move(resnet50_BatchNormInference_479_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_479_0), std::move(resnet50_Relu_480_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_248_0), std::move(resnet50_Reshape_481_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_480_0), std::move(resnet50_Reshape_481_0), std::move(resnet50_Convolution_482_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(131072, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_249_0), std::move(resnet50_Constant_250_0), std::move(resnet50_Convolution_482_0), std::move(resnet50_Constant_251_0), std::move(resnet50_Constant_252_0), std::move(resnet50_BatchNormInference_483_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_472_0), std::move(resnet50_BatchNormInference_483_0), std::move(resnet50_Relu_485_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_473_CallKernel(dim3(32, 128, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_253_0), std::move(resnet50_Reshape_486_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_474Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_485_0), std::move(resnet50_Reshape_486_0), std::move(resnet50_Convolution_487_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_254_0), std::move(resnet50_Constant_255_0), std::move(resnet50_Convolution_487_0), std::move(resnet50_Constant_256_0), std::move(resnet50_Constant_257_0), std::move(resnet50_BatchNormInference_488_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_488_0), std::move(resnet50_Relu_489_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_464_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(resnet50_Constant_258_0), std::move(resnet50_Reshape_490_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_465Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_489_0), std::move(resnet50_Reshape_490_0), std::move(resnet50_Convolution_491_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_462_CallKernel(dim3(32768, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_259_0), std::move(resnet50_Constant_260_0), std::move(resnet50_Convolution_491_0), std::move(resnet50_Constant_261_0), std::move(resnet50_Constant_262_0), std::move(resnet50_BatchNormInference_492_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Relu_float_float_cuda_Relu_277_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_BatchNormInference_492_0), std::move(resnet50_Relu_493_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Reshape_float_float_cuda_Reshape_468_CallKernel(dim3(128, 32, 1), dim3(16, 16, 1), 0, nullptr, std::move(resnet50_Constant_263_0), std::move(resnet50_Reshape_494_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Convolution_float_float_float_cuda_lib_Convolution_469Kernel(std::move(resnet50_cudnn_handle_0), std::move(resnet50_Relu_493_0), std::move(resnet50_Reshape_494_0), std::move(resnet50_Convolution_495_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_470_CallKernel(dim3(131072, 1, 1), dim3(49, 1, 1), 0, nullptr, std::move(resnet50_Constant_264_0), std::move(resnet50_Constant_265_0), std::move(resnet50_Convolution_495_0), std::move(resnet50_Constant_266_0), std::move(resnet50_Constant_267_0), std::move(resnet50_BatchNormInference_496_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_FusedKernel_float_float_float_cuda_Add_Relu_0_CallKernel(dim3(12544, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Relu_485_0), std::move(resnet50_BatchNormInference_496_0), std::move(resnet50_Relu_498_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Sum_float_float_cuda_Sum_499_CallKernel(dim3(131072, 1, 1), dim3(32, 1, 1), 0, nullptr, std::move(resnet50_Relu_498_0), std::move(resnet50_Sum_499_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Divide_float_float_float_cuda_Divide_501_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(resnet50_Sum_499_0), std::move(resnet50_Constant_500_0), std::move(resnet50_Divide_501_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Dot_float_float_float_cuda_lib_Dot_502Kernel(std::move(resnet50_cublas_handle_0), std::move(resnet50_Divide_501_0), std::move(resnet50_Constant_269_0), std::move(resnet50_Dot_502_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Broadcast_float_float_cuda_Broadcast_503_CallKernel(dim3(1001, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(resnet50_Constant_270_0), std::move(resnet50_Broadcast_503_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Add_float_float_float_cuda_Add_504_CallKernel(dim3(1001, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(resnet50_Dot_502_0), std::move(resnet50_Broadcast_503_0), std::move(resnet50_Add_504_0), std::move(Parameter_0_0), std::move(Result_505_0)));
    kernels.emplace_back(new resnet50_Result_float_float_cuda_lib_Result_505Kernel(std::move(resnet50_Add_504_0), std::move(Result_505_0), std::move(Parameter_0_0), std::move(Result_505_0)));
}
