// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nnfusion_rt.h"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>
#include <vector>
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
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* Reshape_486_0;
float* Reshape_487_0;
float* Convolution_488_0;
float* BatchNormInference_489_0;
float* Relu_490_0;
float* Reshape_491_0;
float* Convolution_492_0;
float* BatchNormInference_493_0;
float* Relu_494_0;
float* Reshape_495_0;
float* Convolution_496_0;
float* BatchNormInference_497_0;
float* Relu_498_0;
float* MaxPool_499_0;
float* Reshape_500_0;
float* Convolution_501_0;
float* BatchNormInference_502_0;
float* Relu_503_0;
float* Reshape_504_0;
float* Convolution_505_0;
float* BatchNormInference_506_0;
float* Relu_507_0;
float* MaxPool_508_0;
float* Reshape_511_0;
float* Convolution_512_0;
float* BatchNormInference_517_0;
float* Relu_522_0;
float* Reshape_525_0;
float* Convolution_526_0;
float* BatchNormInference_530_0;
float* Relu_532_0;
float* Reshape_509_0;
float* Convolution_510_0;
float* BatchNormInference_516_0;
float* Relu_521_0;
float* AvgPool_515_0;
float* Reshape_519_0;
float* Convolution_520_0;
float* BatchNormInference_524_0;
float* Relu_529_0;
float* Reshape_513_0;
float* Convolution_514_0;
float* BatchNormInference_518_0;
float* Relu_523_0;
float* Reshape_527_0;
float* Convolution_528_0;
float* BatchNormInference_531_0;
float* Relu_533_0;
float* Reshape_534_0;
float* Convolution_535_0;
float* BatchNormInference_536_0;
float* Relu_537_0;
float* Concat_538_0;
float* AvgPool_545_0;
float* Reshape_549_0;
float* Convolution_550_0;
float* BatchNormInference_554_0;
float* Relu_559_0;
float* Reshape_543_0;
float* Convolution_544_0;
float* BatchNormInference_548_0;
float* Relu_553_0;
float* Reshape_557_0;
float* Convolution_558_0;
float* BatchNormInference_561_0;
float* Relu_563_0;
float* Reshape_564_0;
float* Convolution_565_0;
float* BatchNormInference_566_0;
float* Relu_567_0;
float* Reshape_541_0;
float* Convolution_542_0;
float* BatchNormInference_547_0;
float* Relu_552_0;
float* Reshape_555_0;
float* Convolution_556_0;
float* BatchNormInference_560_0;
float* Relu_562_0;
float* Reshape_539_0;
float* Convolution_540_0;
float* BatchNormInference_546_0;
float* Relu_551_0;
float* Concat_568_0;
float* AvgPool_575_0;
float* Reshape_579_0;
float* Convolution_580_0;
float* BatchNormInference_584_0;
float* Relu_589_0;
float* Reshape_573_0;
float* Convolution_574_0;
float* BatchNormInference_578_0;
float* Relu_583_0;
float* Reshape_587_0;
float* Convolution_588_0;
float* BatchNormInference_591_0;
float* Relu_593_0;
float* Reshape_594_0;
float* Convolution_595_0;
float* BatchNormInference_596_0;
float* Relu_597_0;
float* Reshape_571_0;
float* Convolution_572_0;
float* BatchNormInference_577_0;
float* Relu_582_0;
float* Reshape_585_0;
float* Convolution_586_0;
float* BatchNormInference_590_0;
float* Relu_592_0;
float* Reshape_569_0;
float* Convolution_570_0;
float* BatchNormInference_576_0;
float* Relu_581_0;
float* Concat_598_0;
float* MaxPool_603_0;
float* Reshape_601_0;
float* Convolution_602_0;
float* BatchNormInference_605_0;
float* Relu_607_0;
float* Reshape_608_0;
float* Convolution_609_0;
float* BatchNormInference_610_0;
float* Relu_611_0;
float* Reshape_612_0;
float* Convolution_613_0;
float* BatchNormInference_614_0;
float* Relu_615_0;
float* Reshape_599_0;
float* Convolution_600_0;
float* BatchNormInference_604_0;
float* Relu_606_0;
float* Concat_616_0;
float* AvgPool_623_0;
float* Reshape_627_0;
float* Convolution_628_0;
float* BatchNormInference_632_0;
float* Relu_637_0;
float* Reshape_621_0;
float* Convolution_622_0;
float* BatchNormInference_626_0;
float* Relu_631_0;
float* Reshape_635_0;
float* Convolution_636_0;
float* BatchNormInference_639_0;
float* Relu_641_0;
float* Reshape_644_0;
float* Convolution_645_0;
float* BatchNormInference_647_0;
float* Relu_649_0;
float* Reshape_650_0;
float* Convolution_651_0;
float* BatchNormInference_652_0;
float* Relu_653_0;
float* Reshape_654_0;
float* Convolution_655_0;
float* BatchNormInference_656_0;
float* Relu_657_0;
float* Reshape_619_0;
float* Convolution_620_0;
float* BatchNormInference_625_0;
float* Relu_630_0;
float* Reshape_633_0;
float* Convolution_634_0;
float* BatchNormInference_638_0;
float* Relu_640_0;
float* Reshape_642_0;
float* Convolution_643_0;
float* BatchNormInference_646_0;
float* Relu_648_0;
float* Reshape_617_0;
float* Convolution_618_0;
float* BatchNormInference_624_0;
float* Relu_629_0;
float* Concat_658_0;
float* AvgPool_665_0;
float* Reshape_669_0;
float* Convolution_670_0;
float* BatchNormInference_674_0;
float* Relu_679_0;
float* Reshape_663_0;
float* Convolution_664_0;
float* BatchNormInference_668_0;
float* Relu_673_0;
float* Reshape_677_0;
float* Convolution_678_0;
float* BatchNormInference_681_0;
float* Relu_683_0;
float* Reshape_686_0;
float* Convolution_687_0;
float* BatchNormInference_689_0;
float* Relu_691_0;
float* Reshape_692_0;
float* Convolution_693_0;
float* BatchNormInference_694_0;
float* Relu_695_0;
float* Reshape_696_0;
float* Convolution_697_0;
float* BatchNormInference_698_0;
float* Relu_699_0;
float* Reshape_661_0;
float* Convolution_662_0;
float* BatchNormInference_667_0;
float* Relu_672_0;
float* Reshape_675_0;
float* Convolution_676_0;
float* BatchNormInference_680_0;
float* Relu_682_0;
float* Reshape_684_0;
float* Convolution_685_0;
float* BatchNormInference_688_0;
float* Relu_690_0;
float* Reshape_659_0;
float* Convolution_660_0;
float* BatchNormInference_666_0;
float* Relu_671_0;
float* Concat_700_0;
float* AvgPool_707_0;
float* Reshape_711_0;
float* Convolution_712_0;
float* BatchNormInference_716_0;
float* Relu_721_0;
float* Reshape_705_0;
float* Convolution_706_0;
float* BatchNormInference_710_0;
float* Relu_715_0;
float* Reshape_719_0;
float* Convolution_720_0;
float* BatchNormInference_723_0;
float* Relu_725_0;
float* Reshape_728_0;
float* Convolution_729_0;
float* BatchNormInference_731_0;
float* Relu_733_0;
float* Reshape_734_0;
float* Convolution_735_0;
float* BatchNormInference_736_0;
float* Relu_737_0;
float* Reshape_738_0;
float* Convolution_739_0;
float* BatchNormInference_740_0;
float* Relu_741_0;
float* Reshape_703_0;
float* Convolution_704_0;
float* BatchNormInference_709_0;
float* Relu_714_0;
float* Reshape_717_0;
float* Convolution_718_0;
float* BatchNormInference_722_0;
float* Relu_724_0;
float* Reshape_726_0;
float* Convolution_727_0;
float* BatchNormInference_730_0;
float* Relu_732_0;
float* Reshape_701_0;
float* Convolution_702_0;
float* BatchNormInference_708_0;
float* Relu_713_0;
float* Concat_742_0;
float* AvgPool_749_0;
float* Reshape_753_0;
float* Convolution_754_0;
float* BatchNormInference_758_0;
float* Relu_763_0;
float* Reshape_747_0;
float* Convolution_748_0;
float* BatchNormInference_752_0;
float* Relu_757_0;
float* Reshape_761_0;
float* Convolution_762_0;
float* BatchNormInference_765_0;
float* Relu_767_0;
float* Reshape_770_0;
float* Convolution_771_0;
float* BatchNormInference_773_0;
float* Relu_775_0;
float* Reshape_776_0;
float* Convolution_777_0;
float* BatchNormInference_778_0;
float* Relu_779_0;
float* Reshape_780_0;
float* Convolution_781_0;
float* BatchNormInference_782_0;
float* Relu_783_0;
float* Reshape_745_0;
float* Convolution_746_0;
float* BatchNormInference_751_0;
float* Relu_756_0;
float* Reshape_759_0;
float* Convolution_760_0;
float* BatchNormInference_764_0;
float* Relu_766_0;
float* Reshape_768_0;
float* Convolution_769_0;
float* BatchNormInference_772_0;
float* Relu_774_0;
float* Reshape_743_0;
float* Convolution_744_0;
float* BatchNormInference_750_0;
float* Relu_755_0;
float* Concat_784_0;
float* MaxPool_789_0;
float* Reshape_787_0;
float* Convolution_788_0;
float* BatchNormInference_791_0;
float* Relu_793_0;
float* Reshape_796_0;
float* Convolution_797_0;
float* BatchNormInference_799_0;
float* Relu_801_0;
float* Reshape_802_0;
float* Convolution_803_0;
float* BatchNormInference_804_0;
float* Relu_805_0;
float* Reshape_806_0;
float* Convolution_807_0;
float* BatchNormInference_808_0;
float* Relu_809_0;
float* Reshape_785_0;
float* Convolution_786_0;
float* BatchNormInference_790_0;
float* Relu_792_0;
float* Reshape_794_0;
float* Convolution_795_0;
float* BatchNormInference_798_0;
float* Relu_800_0;
float* Concat_810_0;
float* AvgPool_817_0;
float* Reshape_821_0;
float* Convolution_822_0;
float* BatchNormInference_826_0;
float* Relu_833_0;
float* Reshape_815_0;
float* Convolution_816_0;
float* BatchNormInference_820_0;
float* Relu_825_0;
float* Reshape_831_0;
float* Convolution_832_0;
float* BatchNormInference_836_0;
float* Relu_839_0;
float* Reshape_842_0;
float* Convolution_843_0;
float* BatchNormInference_845_0;
float* Relu_847_0;
float* Reshape_840_0;
float* Convolution_841_0;
float* BatchNormInference_844_0;
float* Relu_846_0;
float* Reshape_813_0;
float* Convolution_814_0;
float* BatchNormInference_819_0;
float* Relu_824_0;
float* Reshape_829_0;
float* Convolution_830_0;
float* BatchNormInference_835_0;
float* Relu_838_0;
float* Reshape_827_0;
float* Convolution_828_0;
float* BatchNormInference_834_0;
float* Relu_837_0;
float* Reshape_811_0;
float* Convolution_812_0;
float* BatchNormInference_818_0;
float* Relu_823_0;
float* Concat_848_0;
float* MaxPool_855_0;
float* Reshape_859_0;
float* Convolution_860_0;
float* BatchNormInference_864_0;
float* Relu_871_0;
float* Reshape_853_0;
float* Convolution_854_0;
float* BatchNormInference_858_0;
float* Relu_863_0;
float* Reshape_869_0;
float* Convolution_870_0;
float* BatchNormInference_874_0;
float* Relu_877_0;
float* Reshape_880_0;
float* Convolution_881_0;
float* BatchNormInference_883_0;
float* Relu_885_0;
float* Reshape_878_0;
float* Convolution_879_0;
float* BatchNormInference_882_0;
float* Relu_884_0;
float* Reshape_851_0;
float* Convolution_852_0;
float* BatchNormInference_857_0;
float* Relu_862_0;
float* Reshape_867_0;
float* Convolution_868_0;
float* BatchNormInference_873_0;
float* Relu_876_0;
float* Reshape_865_0;
float* Convolution_866_0;
float* BatchNormInference_872_0;
float* Relu_875_0;
float* Reshape_849_0;
float* Convolution_850_0;
float* BatchNormInference_856_0;
float* Relu_861_0;
float* Concat_886_0;
float* AvgPool_887_0;
float* Reshape_888_0;
float* Dot_889_0;
float* Broadcast_890_0;
float* Add_891_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
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
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
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
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_484_0;
float* Constant_2_0;
float* Constant_6_0;
float* Constant_5_0;
float* Constant_4_0;
float* Constant_3_0;
float* Constant_7_0;
float* Constant_11_0;
float* Constant_10_0;
float* Constant_9_0;
float* Constant_8_0;
float* Constant_12_0;
float* Constant_16_0;
float* Constant_15_0;
float* Constant_14_0;
float* Constant_13_0;
float* Constant_17_0;
float* Constant_21_0;
float* Constant_20_0;
float* Constant_19_0;
float* Constant_18_0;
float* Constant_22_0;
float* Constant_26_0;
float* Constant_25_0;
float* Constant_24_0;
float* Constant_23_0;
float* Constant_32_0;
float* Constant_36_0;
float* Constant_35_0;
float* Constant_34_0;
float* Constant_33_0;
float* Constant_37_0;
float* Constant_41_0;
float* Constant_40_0;
float* Constant_39_0;
float* Constant_38_0;
float* Constant_27_0;
float* Constant_31_0;
float* Constant_30_0;
float* Constant_29_0;
float* Constant_28_0;
float* Constant_57_0;
float* Constant_61_0;
float* Constant_60_0;
float* Constant_59_0;
float* Constant_58_0;
float* Constant_42_0;
float* Constant_46_0;
float* Constant_45_0;
float* Constant_44_0;
float* Constant_43_0;
float* Constant_47_0;
float* Constant_51_0;
float* Constant_50_0;
float* Constant_49_0;
float* Constant_48_0;
float* Constant_52_0;
float* Constant_56_0;
float* Constant_55_0;
float* Constant_54_0;
float* Constant_53_0;
float* Constant_93_0;
float* Constant_94_0;
float* Constant_97_0;
float* Constant_96_0;
float* Constant_95_0;
float* Constant_78_0;
float* Constant_79_0;
float* Constant_82_0;
float* Constant_81_0;
float* Constant_80_0;
float* Constant_83_0;
float* Constant_84_0;
float* Constant_87_0;
float* Constant_86_0;
float* Constant_85_0;
float* Constant_88_0;
float* Constant_89_0;
float* Constant_92_0;
float* Constant_91_0;
float* Constant_90_0;
float* Constant_68_0;
float* Constant_69_0;
float* Constant_72_0;
float* Constant_71_0;
float* Constant_70_0;
float* Constant_73_0;
float* Constant_74_0;
float* Constant_77_0;
float* Constant_76_0;
float* Constant_75_0;
float* Constant_63_0;
float* Constant_64_0;
float* Constant_67_0;
float* Constant_66_0;
float* Constant_65_0;
float* Constant_129_0;
float* Constant_130_0;
float* Constant_133_0;
float* Constant_132_0;
float* Constant_131_0;
float* Constant_114_0;
float* Constant_115_0;
float* Constant_118_0;
float* Constant_117_0;
float* Constant_116_0;
float* Constant_119_0;
float* Constant_120_0;
float* Constant_123_0;
float* Constant_122_0;
float* Constant_121_0;
float* Constant_124_0;
float* Constant_125_0;
float* Constant_128_0;
float* Constant_127_0;
float* Constant_126_0;
float* Constant_104_0;
float* Constant_105_0;
float* Constant_108_0;
float* Constant_107_0;
float* Constant_106_0;
float* Constant_109_0;
float* Constant_110_0;
float* Constant_113_0;
float* Constant_112_0;
float* Constant_111_0;
float* Constant_99_0;
float* Constant_100_0;
float* Constant_103_0;
float* Constant_102_0;
float* Constant_101_0;
float* Constant_140_0;
float* Constant_144_0;
float* Constant_143_0;
float* Constant_142_0;
float* Constant_141_0;
float* Constant_145_0;
float* Constant_149_0;
float* Constant_148_0;
float* Constant_147_0;
float* Constant_146_0;
float* Constant_150_0;
float* Constant_154_0;
float* Constant_153_0;
float* Constant_152_0;
float* Constant_151_0;
float* Constant_135_0;
float* Constant_139_0;
float* Constant_138_0;
float* Constant_137_0;
float* Constant_136_0;
float* Constant_201_0;
float* Constant_205_0;
float* Constant_204_0;
float* Constant_203_0;
float* Constant_202_0;
float* Constant_176_0;
float* Constant_180_0;
float* Constant_179_0;
float* Constant_178_0;
float* Constant_177_0;
float* Constant_181_0;
float* Constant_185_0;
float* Constant_184_0;
float* Constant_183_0;
float* Constant_182_0;
float* Constant_186_0;
float* Constant_190_0;
float* Constant_189_0;
float* Constant_188_0;
float* Constant_187_0;
float* Constant_191_0;
float* Constant_195_0;
float* Constant_194_0;
float* Constant_193_0;
float* Constant_192_0;
float* Constant_196_0;
float* Constant_200_0;
float* Constant_199_0;
float* Constant_198_0;
float* Constant_197_0;
float* Constant_161_0;
float* Constant_165_0;
float* Constant_164_0;
float* Constant_163_0;
float* Constant_162_0;
float* Constant_166_0;
float* Constant_170_0;
float* Constant_169_0;
float* Constant_168_0;
float* Constant_167_0;
float* Constant_171_0;
float* Constant_175_0;
float* Constant_174_0;
float* Constant_173_0;
float* Constant_172_0;
float* Constant_156_0;
float* Constant_160_0;
float* Constant_159_0;
float* Constant_158_0;
float* Constant_157_0;
float* Constant_252_0;
float* Constant_253_0;
float* Constant_256_0;
float* Constant_255_0;
float* Constant_254_0;
float* Constant_227_0;
float* Constant_228_0;
float* Constant_231_0;
float* Constant_230_0;
float* Constant_229_0;
float* Constant_232_0;
float* Constant_233_0;
float* Constant_236_0;
float* Constant_235_0;
float* Constant_234_0;
float* Constant_237_0;
float* Constant_238_0;
float* Constant_241_0;
float* Constant_240_0;
float* Constant_239_0;
float* Constant_242_0;
float* Constant_243_0;
float* Constant_246_0;
float* Constant_245_0;
float* Constant_244_0;
float* Constant_247_0;
float* Constant_248_0;
float* Constant_251_0;
float* Constant_250_0;
float* Constant_249_0;
float* Constant_212_0;
float* Constant_213_0;
float* Constant_216_0;
float* Constant_215_0;
float* Constant_214_0;
float* Constant_217_0;
float* Constant_218_0;
float* Constant_221_0;
float* Constant_220_0;
float* Constant_219_0;
float* Constant_222_0;
float* Constant_223_0;
float* Constant_226_0;
float* Constant_225_0;
float* Constant_224_0;
float* Constant_207_0;
float* Constant_208_0;
float* Constant_211_0;
float* Constant_210_0;
float* Constant_209_0;
float* Constant_303_0;
float* Constant_304_0;
float* Constant_307_0;
float* Constant_306_0;
float* Constant_305_0;
float* Constant_278_0;
float* Constant_279_0;
float* Constant_282_0;
float* Constant_281_0;
float* Constant_280_0;
float* Constant_283_0;
float* Constant_284_0;
float* Constant_287_0;
float* Constant_286_0;
float* Constant_285_0;
float* Constant_288_0;
float* Constant_289_0;
float* Constant_292_0;
float* Constant_291_0;
float* Constant_290_0;
float* Constant_293_0;
float* Constant_294_0;
float* Constant_297_0;
float* Constant_296_0;
float* Constant_295_0;
float* Constant_298_0;
float* Constant_299_0;
float* Constant_302_0;
float* Constant_301_0;
float* Constant_300_0;
float* Constant_263_0;
float* Constant_264_0;
float* Constant_267_0;
float* Constant_266_0;
float* Constant_265_0;
float* Constant_268_0;
float* Constant_269_0;
float* Constant_272_0;
float* Constant_271_0;
float* Constant_270_0;
float* Constant_273_0;
float* Constant_274_0;
float* Constant_277_0;
float* Constant_276_0;
float* Constant_275_0;
float* Constant_258_0;
float* Constant_259_0;
float* Constant_262_0;
float* Constant_261_0;
float* Constant_260_0;
float* Constant_354_0;
float* Constant_355_0;
float* Constant_358_0;
float* Constant_357_0;
float* Constant_356_0;
float* Constant_329_0;
float* Constant_330_0;
float* Constant_333_0;
float* Constant_332_0;
float* Constant_331_0;
float* Constant_334_0;
float* Constant_335_0;
float* Constant_338_0;
float* Constant_337_0;
float* Constant_336_0;
float* Constant_339_0;
float* Constant_340_0;
float* Constant_343_0;
float* Constant_342_0;
float* Constant_341_0;
float* Constant_344_0;
float* Constant_345_0;
float* Constant_348_0;
float* Constant_347_0;
float* Constant_346_0;
float* Constant_349_0;
float* Constant_350_0;
float* Constant_353_0;
float* Constant_352_0;
float* Constant_351_0;
float* Constant_314_0;
float* Constant_315_0;
float* Constant_318_0;
float* Constant_317_0;
float* Constant_316_0;
float* Constant_319_0;
float* Constant_320_0;
float* Constant_323_0;
float* Constant_322_0;
float* Constant_321_0;
float* Constant_324_0;
float* Constant_325_0;
float* Constant_328_0;
float* Constant_327_0;
float* Constant_326_0;
float* Constant_309_0;
float* Constant_310_0;
float* Constant_313_0;
float* Constant_312_0;
float* Constant_311_0;
float* Constant_370_0;
float* Constant_374_0;
float* Constant_373_0;
float* Constant_372_0;
float* Constant_371_0;
float* Constant_375_0;
float* Constant_379_0;
float* Constant_378_0;
float* Constant_377_0;
float* Constant_376_0;
float* Constant_380_0;
float* Constant_384_0;
float* Constant_383_0;
float* Constant_382_0;
float* Constant_381_0;
float* Constant_385_0;
float* Constant_389_0;
float* Constant_388_0;
float* Constant_387_0;
float* Constant_386_0;
float* Constant_360_0;
float* Constant_364_0;
float* Constant_363_0;
float* Constant_362_0;
float* Constant_361_0;
float* Constant_365_0;
float* Constant_369_0;
float* Constant_368_0;
float* Constant_367_0;
float* Constant_366_0;
float* Constant_431_0;
float* Constant_435_0;
float* Constant_434_0;
float* Constant_433_0;
float* Constant_432_0;
float* Constant_411_0;
float* Constant_415_0;
float* Constant_414_0;
float* Constant_413_0;
float* Constant_412_0;
float* Constant_416_0;
float* Constant_420_0;
float* Constant_419_0;
float* Constant_418_0;
float* Constant_417_0;
float* Constant_426_0;
float* Constant_430_0;
float* Constant_429_0;
float* Constant_428_0;
float* Constant_427_0;
float* Constant_421_0;
float* Constant_425_0;
float* Constant_424_0;
float* Constant_423_0;
float* Constant_422_0;
float* Constant_396_0;
float* Constant_400_0;
float* Constant_399_0;
float* Constant_398_0;
float* Constant_397_0;
float* Constant_406_0;
float* Constant_410_0;
float* Constant_409_0;
float* Constant_408_0;
float* Constant_407_0;
float* Constant_401_0;
float* Constant_405_0;
float* Constant_404_0;
float* Constant_403_0;
float* Constant_402_0;
float* Constant_391_0;
float* Constant_395_0;
float* Constant_394_0;
float* Constant_393_0;
float* Constant_392_0;
float* Constant_477_0;
float* Constant_478_0;
float* Constant_481_0;
float* Constant_480_0;
float* Constant_479_0;
float* Constant_457_0;
float* Constant_458_0;
float* Constant_461_0;
float* Constant_460_0;
float* Constant_459_0;
float* Constant_462_0;
float* Constant_463_0;
float* Constant_466_0;
float* Constant_465_0;
float* Constant_464_0;
float* Constant_472_0;
float* Constant_473_0;
float* Constant_476_0;
float* Constant_475_0;
float* Constant_474_0;
float* Constant_467_0;
float* Constant_468_0;
float* Constant_471_0;
float* Constant_470_0;
float* Constant_469_0;
float* Constant_442_0;
float* Constant_443_0;
float* Constant_446_0;
float* Constant_445_0;
float* Constant_444_0;
float* Constant_452_0;
float* Constant_453_0;
float* Constant_456_0;
float* Constant_455_0;
float* Constant_454_0;
float* Constant_447_0;
float* Constant_448_0;
float* Constant_451_0;
float* Constant_450_0;
float* Constant_449_0;
float* Constant_437_0;
float* Constant_438_0;
float* Constant_441_0;
float* Constant_440_0;
float* Constant_439_0;
float* Constant_485_0;
// Node name:	Convolution_685
// Description:	Convolution
// Input:
//	- name: Relu_682_0	type: float	shape: Shape{32, 160, 17, 17}
//	- name: Reshape_684_0	type: float	shape: Shape{192, 160, 7, 1}
// Output:
//	- name: Convolution_685_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_685(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 160, 7, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 3, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_506
// Description:	BatchNormInference
// Input:
//	- name: Constant_23_0	type: float	shape: Shape{192}
//	- name: Constant_24_0	type: float	shape: Shape{192}
//	- name: Convolution_505_0	type: float	shape: Shape{32, 192, 71, 71}
//	- name: Constant_25_0	type: float	shape: Shape{192}
//	- name: Constant_26_0	type: float	shape: Shape{192}
// Output:
//	- name: BatchNormInference_506_0	type: float	shape: Shape{32, 192, 71, 71}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_506(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 71 * 71;
    const int c_id = blockIdx.x % 192;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 71 * 71; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_506_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_506<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_600
// Description:	Convolution
// Input:
//	- name: Concat_598_0	type: float	shape: Shape{32, 288, 35, 35}
//	- name: Reshape_599_0	type: float	shape: Shape{384, 288, 3, 3}
// Output:
//	- name: Convolution_600_0	type: float	shape: Shape{32, 384, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_600(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 288, 3, 3));
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
// Node name:	Concat_658
// Description:	Concat
// Input:
//	- name: Relu_629_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Relu_648_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Relu_657_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Relu_637_0	type: float	shape: Shape{32, 192, 17, 17}
// Output:
//	- name: Concat_658_0	type: float	shape: Shape{32, 768, 17, 17}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_float_cuda_Concat_658(float* input0, float* input1, float* input2, float* input3, float* output0)
{
    uint32_t inputs_strides[] = {55488, 55488, 55488, 55488};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 7102464)
    {
        uint32_t block_id = tid / 221952;
        uint32_t block_idx = tid % 221952;
        uint32_t output_idx = block_id * 221952 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
        if(block_idx < inputs_strides[3])
        {
            output0[output_idx] = input3[block_id * inputs_strides[3] + block_idx];
            return;
        }
        block_idx -= inputs_strides[3];
    }

}
extern void Concat_float_float_float_float_float_cuda_Concat_658_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    Concat_float_float_float_float_float_cuda_Concat_658<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}
// Node name:	Constant_350
// Description:	Constant
// Input:
// Output:
//	- name: Constant_350_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_350(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_350_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_350_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_163
// Description:	Constant
// Input:
// Output:
//	- name: Constant_163_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_163(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_163_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: Constant_7_0	type: float	shape: Shape{3, 3, 32, 32}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[36864];
    bin_file.read(tmp_mem, 36864);
    cudaMemcpyAsync(output0, tmp_mem, 36864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_186
// Description:	Constant
// Input:
// Output:
//	- name: Constant_186_0	type: float	shape: Shape{1, 7, 128, 128}
void Constant_float_cuda_Constant_186(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_186_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_186_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[458752];
    bin_file.read(tmp_mem, 458752);
    cudaMemcpyAsync(output0, tmp_mem, 458752, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_345
// Description:	Constant
// Input:
// Output:
//	- name: Constant_345_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_345(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_345_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_345_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_89
// Description:	Constant
// Input:
// Output:
//	- name: Constant_89_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_89(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_89_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_89_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_74
// Description:	Constant
// Input:
// Output:
//	- name: Constant_74_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_74(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_74_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_74_0 failed.\n");
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
//	- name: Constant_162_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_162_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: Constant_4_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_83
// Description:	Constant
// Input:
// Output:
//	- name: Constant_83_0	type: float	shape: Shape{3, 3, 64, 96}
void Constant_float_cuda_Constant_83(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_83_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_83_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[221184];
    bin_file.read(tmp_mem, 221184);
    cudaMemcpyAsync(output0, tmp_mem, 221184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_383
// Description:	Constant
// Input:
// Output:
//	- name: Constant_383_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_383(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_383_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_383_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_280
// Description:	Constant
// Input:
// Output:
//	- name: Constant_280_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_280(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_280_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_280_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_64
// Description:	Constant
// Input:
// Output:
//	- name: Constant_64_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_64(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_64_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_64_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_123
// Description:	Constant
// Input:
// Output:
//	- name: Constant_123_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_123(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_123_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_123_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: Constant_100_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_100(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_100_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_100_0 failed.\n");
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
//	- name: Constant_53_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_168
// Description:	Constant
// Input:
// Output:
//	- name: Constant_168_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_168(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_168_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_157
// Description:	Constant
// Input:
// Output:
//	- name: Constant_157_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_157(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_157_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_157_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_148
// Description:	Constant
// Input:
// Output:
//	- name: Constant_148_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_148_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_13
// Description:	Constant
// Input:
// Output:
//	- name: Constant_13_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_13_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_202
// Description:	Constant
// Input:
// Output:
//	- name: Constant_202_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_202(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_202_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_202_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: Constant_167_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: Constant_40_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_55
// Description:	Constant
// Input:
// Output:
//	- name: Constant_55_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_55(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_55_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_55_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: Constant_68_0	type: float	shape: Shape{1, 1, 256, 48}
void Constant_float_cuda_Constant_68(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_68_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_68_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[49152];
    bin_file.read(tmp_mem, 49152);
    cudaMemcpyAsync(output0, tmp_mem, 49152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_50
// Description:	Constant
// Input:
// Output:
//	- name: Constant_50_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_50_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_48
// Description:	Constant
// Input:
// Output:
//	- name: Constant_48_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_48(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_48_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_48_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_49
// Description:	Constant
// Input:
// Output:
//	- name: Constant_49_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_49_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_331
// Description:	Constant
// Input:
// Output:
//	- name: Constant_331_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_331(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_331_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_331_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_377
// Description:	Constant
// Input:
// Output:
//	- name: Constant_377_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_377(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_377_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_377_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: Constant_47_0	type: float	shape: Shape{3, 3, 64, 96}
void Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[221184];
    bin_file.read(tmp_mem, 221184);
    cudaMemcpyAsync(output0, tmp_mem, 221184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: Constant_96_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_96_0 failed.\n");
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
//	- name: Constant_272_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_272(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_272_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_272_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_80
// Description:	Constant
// Input:
// Output:
//	- name: Constant_80_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_80(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_80_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_80_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: Constant_5_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_221
// Description:	Constant
// Input:
// Output:
//	- name: Constant_221_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_221(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_221_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_221_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_122
// Description:	Constant
// Input:
// Output:
//	- name: Constant_122_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_122(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_122_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_122_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_190
// Description:	Constant
// Input:
// Output:
//	- name: Constant_190_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: Constant_25_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: Constant_43_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_395
// Description:	Constant
// Input:
// Output:
//	- name: Constant_395_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_395(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_395_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_395_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_236
// Description:	Constant
// Input:
// Output:
//	- name: Constant_236_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_236(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_236_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_236_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_320
// Description:	Constant
// Input:
// Output:
//	- name: Constant_320_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_320(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_320_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_320_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_286
// Description:	Constant
// Input:
// Output:
//	- name: Constant_286_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_286(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_286_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_286_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_82
// Description:	Constant
// Input:
// Output:
//	- name: Constant_82_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_82(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_82_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_82_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_125
// Description:	Constant
// Input:
// Output:
//	- name: Constant_125_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_125_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: Constant_42_0	type: float	shape: Shape{1, 1, 192, 64}
void Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[49152];
    bin_file.read(tmp_mem, 49152);
    cudaMemcpyAsync(output0, tmp_mem, 49152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_161
// Description:	Constant
// Input:
// Output:
//	- name: Constant_161_0	type: float	shape: Shape{1, 1, 768, 128}
void Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_161_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[393216];
    bin_file.read(tmp_mem, 393216);
    cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_351
// Description:	Constant
// Input:
// Output:
//	- name: Constant_351_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_351(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_351_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_351_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_60
// Description:	Constant
// Input:
// Output:
//	- name: Constant_60_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_60(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_60_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_60_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: Constant_33_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_33_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_140
// Description:	Constant
// Input:
// Output:
//	- name: Constant_140_0	type: float	shape: Shape{1, 1, 288, 64}
void Constant_float_cuda_Constant_140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_140_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[73728];
    bin_file.read(tmp_mem, 73728);
    cudaMemcpyAsync(output0, tmp_mem, 73728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_94
// Description:	Constant
// Input:
// Output:
//	- name: Constant_94_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_94(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_94_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_94_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: Constant_57_0	type: float	shape: Shape{1, 1, 192, 32}
void Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_340
// Description:	Constant
// Input:
// Output:
//	- name: Constant_340_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_340(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_340_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_340_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_197
// Description:	Constant
// Input:
// Output:
//	- name: Constant_197_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_197_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: Constant_37_0	type: float	shape: Shape{5, 5, 48, 64}
void Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[307200];
    bin_file.read(tmp_mem, 307200);
    cudaMemcpyAsync(output0, tmp_mem, 307200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: Constant_31_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_66
// Description:	Constant
// Input:
// Output:
//	- name: Constant_66_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_66(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_66_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_66_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_59
// Description:	Constant
// Input:
// Output:
//	- name: Constant_59_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_59(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_59_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_59_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_432
// Description:	Constant
// Input:
// Output:
//	- name: Constant_432_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_432(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_432_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_432_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_20
// Description:	Constant
// Input:
// Output:
//	- name: Constant_20_0	type: float	shape: Shape{80}
void Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_20_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[320];
    bin_file.read(tmp_mem, 320);
    cudaMemcpyAsync(output0, tmp_mem, 320, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_207
// Description:	Constant
// Input:
// Output:
//	- name: Constant_207_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_207(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_207_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_207_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_78
// Description:	Constant
// Input:
// Output:
//	- name: Constant_78_0	type: float	shape: Shape{1, 1, 256, 64}
void Constant_float_cuda_Constant_78(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_78_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_78_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: Constant_24_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_219
// Description:	Constant
// Input:
// Output:
//	- name: Constant_219_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_219(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_219_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_219_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_290
// Description:	Constant
// Input:
// Output:
//	- name: Constant_290_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_290(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_290_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_290_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_284
// Description:	Constant
// Input:
// Output:
//	- name: Constant_284_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_284(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_284_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_284_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_226
// Description:	Constant
// Input:
// Output:
//	- name: Constant_226_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_226(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_226_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_226_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_398
// Description:	Constant
// Input:
// Output:
//	- name: Constant_398_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_398(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_398_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_398_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2_0	type: float	shape: Shape{3, 3, 3, 32}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3456];
    bin_file.read(tmp_mem, 3456);
    cudaMemcpyAsync(output0, tmp_mem, 3456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_327
// Description:	Constant
// Input:
// Output:
//	- name: Constant_327_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_327(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_327_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_327_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_99
// Description:	Constant
// Input:
// Output:
//	- name: Constant_99_0	type: float	shape: Shape{1, 1, 288, 64}
void Constant_float_cuda_Constant_99(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_99_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_99_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[73728];
    bin_file.read(tmp_mem, 73728);
    cudaMemcpyAsync(output0, tmp_mem, 73728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_326
// Description:	Constant
// Input:
// Output:
//	- name: Constant_326_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_326(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_326_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_326_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_185
// Description:	Constant
// Input:
// Output:
//	- name: Constant_185_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_185_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_251
// Description:	Constant
// Input:
// Output:
//	- name: Constant_251_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_251(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_251_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_251_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_229
// Description:	Constant
// Input:
// Output:
//	- name: Constant_229_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_229(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_229_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_229_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_139
// Description:	Constant
// Input:
// Output:
//	- name: Constant_139_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_139(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_139_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_139_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{1, 1, 192, 64}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[49152];
    bin_file.read(tmp_mem, 49152);
    cudaMemcpyAsync(output0, tmp_mem, 49152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: Constant_15_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_310
// Description:	Constant
// Input:
// Output:
//	- name: Constant_310_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_310(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_310_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_310_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_379
// Description:	Constant
// Input:
// Output:
//	- name: Constant_379_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_379(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_379_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_379_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_228
// Description:	Constant
// Input:
// Output:
//	- name: Constant_228_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_228(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_228_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_228_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_201
// Description:	Constant
// Input:
// Output:
//	- name: Constant_201_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_201(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_201_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_201_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_16
// Description:	Constant
// Input:
// Output:
//	- name: Constant_16_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_261
// Description:	Constant
// Input:
// Output:
//	- name: Constant_261_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_261(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_261_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_261_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_416
// Description:	Constant
// Input:
// Output:
//	- name: Constant_416_0	type: float	shape: Shape{3, 3, 448, 384}
void Constant_float_cuda_Constant_416(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_416_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_416_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6193152];
    bin_file.read(tmp_mem, 6193152);
    cudaMemcpyAsync(output0, tmp_mem, 6193152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: Constant_72_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_72_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_198
// Description:	Constant
// Input:
// Output:
//	- name: Constant_198_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_198(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_198_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_198_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: Constant_45_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_56
// Description:	Constant
// Input:
// Output:
//	- name: Constant_56_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_56(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_56_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_56_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_192
// Description:	Constant
// Input:
// Output:
//	- name: Constant_192_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_192(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_192_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
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
//	- name: Constant_196_0	type: float	shape: Shape{1, 7, 128, 192}
void Constant_float_cuda_Constant_196(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_196_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_196_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[688128];
    bin_file.read(tmp_mem, 688128);
    cudaMemcpyAsync(output0, tmp_mem, 688128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_34
// Description:	Constant
// Input:
// Output:
//	- name: Constant_34_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_34(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_34_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_469
// Description:	Constant
// Input:
// Output:
//	- name: Constant_469_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_469(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_469_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_469_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_384
// Description:	Constant
// Input:
// Output:
//	- name: Constant_384_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_384(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_384_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_384_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: Constant_38_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_17
// Description:	Constant
// Input:
// Output:
//	- name: Constant_17_0	type: float	shape: Shape{1, 1, 64, 80}
void Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[20480];
    bin_file.read(tmp_mem, 20480);
    cudaMemcpyAsync(output0, tmp_mem, 20480, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{80}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[320];
    bin_file.read(tmp_mem, 320);
    cudaMemcpyAsync(output0, tmp_mem, 320, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_246
// Description:	Constant
// Input:
// Output:
//	- name: Constant_246_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_246(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_246_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_246_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_129
// Description:	Constant
// Input:
// Output:
//	- name: Constant_129_0	type: float	shape: Shape{1, 1, 288, 64}
void Constant_float_cuda_Constant_129(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_129_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_129_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[73728];
    bin_file.read(tmp_mem, 73728);
    cudaMemcpyAsync(output0, tmp_mem, 73728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_304
// Description:	Constant
// Input:
// Output:
//	- name: Constant_304_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_304(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_304_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_304_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: Constant_11_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_480
// Description:	Constant
// Input:
// Output:
//	- name: Constant_480_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_480(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_480_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_480_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_259
// Description:	Constant
// Input:
// Output:
//	- name: Constant_259_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_259(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_259_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_259_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: Constant_44_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_399
// Description:	Constant
// Input:
// Output:
//	- name: Constant_399_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_399(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_399_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_399_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_332
// Description:	Constant
// Input:
// Output:
//	- name: Constant_332_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_332(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_332_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_332_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: Constant_39_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
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
//	- name: Constant_22_0	type: float	shape: Shape{3, 3, 80, 192}
void Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_22_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[552960];
    bin_file.read(tmp_mem, 552960);
    cudaMemcpyAsync(output0, tmp_mem, 552960, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: Constant_58_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_58(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_165
// Description:	Constant
// Input:
// Output:
//	- name: Constant_165_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_165(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_165_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: Constant_21_0	type: float	shape: Shape{80}
void Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_21_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[320];
    bin_file.read(tmp_mem, 320);
    cudaMemcpyAsync(output0, tmp_mem, 320, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_382
// Description:	Constant
// Input:
// Output:
//	- name: Constant_382_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_382(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_382_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_382_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: Constant_169_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_169_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: Constant_29_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_205
// Description:	Constant
// Input:
// Output:
//	- name: Constant_205_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_205(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_205_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_205_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: Constant_12_0	type: float	shape: Shape{3, 3, 32, 64}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[73728];
    bin_file.read(tmp_mem, 73728);
    cudaMemcpyAsync(output0, tmp_mem, 73728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_178
// Description:	Constant
// Input:
// Output:
//	- name: Constant_178_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_178(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_178_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_423
// Description:	Constant
// Input:
// Output:
//	- name: Constant_423_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_423(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_423_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_423_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: Constant_88_0	type: float	shape: Shape{3, 3, 96, 96}
void Constant_float_cuda_Constant_88(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_88_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[331776];
    bin_file.read(tmp_mem, 331776);
    cudaMemcpyAsync(output0, tmp_mem, 331776, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_317
// Description:	Constant
// Input:
// Output:
//	- name: Constant_317_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_317(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_317_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_317_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_323
// Description:	Constant
// Input:
// Output:
//	- name: Constant_323_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_323(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_323_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_323_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_133
// Description:	Constant
// Input:
// Output:
//	- name: Constant_133_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_133(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_133_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_133_0 failed.\n");
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
//	- name: Constant_10_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: Constant_36_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_36_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_422
// Description:	Constant
// Input:
// Output:
//	- name: Constant_422_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_422(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_422_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_422_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_337
// Description:	Constant
// Input:
// Output:
//	- name: Constant_337_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_337(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_337_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_337_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: Constant_51_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: Constant_8_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_179
// Description:	Constant
// Input:
// Output:
//	- name: Constant_179_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_179_0 failed.\n");
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
//	- name: Constant_35_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_417
// Description:	Constant
// Input:
// Output:
//	- name: Constant_417_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_417(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_417_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_417_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_465
// Description:	Constant
// Input:
// Output:
//	- name: Constant_465_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_465(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_465_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_465_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_458
// Description:	Constant
// Input:
// Output:
//	- name: Constant_458_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_458(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_458_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_458_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: Constant_54_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_102
// Description:	Constant
// Input:
// Output:
//	- name: Constant_102_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_102(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_102_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_102_0 failed.\n");
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
//	- name: Constant_143_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_143_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_447
// Description:	Constant
// Input:
// Output:
//	- name: Constant_447_0	type: float	shape: Shape{1, 3, 384, 384}
void Constant_float_cuda_Constant_447(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_447_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_447_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_380
// Description:	Constant
// Input:
// Output:
//	- name: Constant_380_0	type: float	shape: Shape{7, 1, 192, 192}
void Constant_float_cuda_Constant_380(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_380_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_380_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_391
// Description:	Constant
// Input:
// Output:
//	- name: Constant_391_0	type: float	shape: Shape{1, 1, 1280, 320}
void Constant_float_cuda_Constant_391(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_391_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_391_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1638400];
    bin_file.read(tmp_mem, 1638400);
    cudaMemcpyAsync(output0, tmp_mem, 1638400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_450
// Description:	Constant
// Input:
// Output:
//	- name: Constant_450_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_450(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_450_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_450_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_368
// Description:	Constant
// Input:
// Output:
//	- name: Constant_368_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_368(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_368_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_368_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_387
// Description:	Constant
// Input:
// Output:
//	- name: Constant_387_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_387(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_387_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_387_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_441
// Description:	Constant
// Input:
// Output:
//	- name: Constant_441_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_441(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_441_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_441_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_405
// Description:	Constant
// Input:
// Output:
//	- name: Constant_405_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_405(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_405_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_405_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_176
// Description:	Constant
// Input:
// Output:
//	- name: Constant_176_0	type: float	shape: Shape{1, 1, 768, 128}
void Constant_float_cuda_Constant_176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_176_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[393216];
    bin_file.read(tmp_mem, 393216);
    cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_265
// Description:	Constant
// Input:
// Output:
//	- name: Constant_265_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_265(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_265_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_265_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_364
// Description:	Constant
// Input:
// Output:
//	- name: Constant_364_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_364(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_364_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_364_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: Constant_32_0	type: float	shape: Shape{1, 1, 192, 48}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[36864];
    bin_file.read(tmp_mem, 36864);
    cudaMemcpyAsync(output0, tmp_mem, 36864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_263
// Description:	Constant
// Input:
// Output:
//	- name: Constant_263_0	type: float	shape: Shape{1, 1, 768, 160}
void Constant_float_cuda_Constant_263(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_263_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_263_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[491520];
    bin_file.read(tmp_mem, 491520);
    cudaMemcpyAsync(output0, tmp_mem, 491520, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_128
// Description:	Constant
// Input:
// Output:
//	- name: Constant_128_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_128_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_314
// Description:	Constant
// Input:
// Output:
//	- name: Constant_314_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_314(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_314_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_314_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_135
// Description:	Constant
// Input:
// Output:
//	- name: Constant_135_0	type: float	shape: Shape{3, 3, 288, 384}
void Constant_float_cuda_Constant_135(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_135_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_135_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3981312];
    bin_file.read(tmp_mem, 3981312);
    cudaMemcpyAsync(output0, tmp_mem, 3981312, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_369
// Description:	Constant
// Input:
// Output:
//	- name: Constant_369_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_369(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_369_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_369_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_459
// Description:	Constant
// Input:
// Output:
//	- name: Constant_459_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_459(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_459_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_459_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_370
// Description:	Constant
// Input:
// Output:
//	- name: Constant_370_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_370(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_370_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_370_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_385
// Description:	Constant
// Input:
// Output:
//	- name: Constant_385_0	type: float	shape: Shape{3, 3, 192, 192}
void Constant_float_cuda_Constant_385(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_385_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_385_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1327104];
    bin_file.read(tmp_mem, 1327104);
    cudaMemcpyAsync(output0, tmp_mem, 1327104, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_460
// Description:	Constant
// Input:
// Output:
//	- name: Constant_460_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_460(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_460_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_460_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_367
// Description:	Constant
// Input:
// Output:
//	- name: Constant_367_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_367(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_367_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_367_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_366
// Description:	Constant
// Input:
// Output:
//	- name: Constant_366_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_366(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_366_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_366_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_374
// Description:	Constant
// Input:
// Output:
//	- name: Constant_374_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_374(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_374_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_374_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_421
// Description:	Constant
// Input:
// Output:
//	- name: Constant_421_0	type: float	shape: Shape{1, 3, 384, 384}
void Constant_float_cuda_Constant_421(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_421_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_421_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_484
// Description:	Constant
// Input:
// Output:
//	- name: Constant_484_0	type: float	shape: Shape{2048, 1001}
void Constant_float_cuda_Constant_484(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_484_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_484_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8200192];
    bin_file.read(tmp_mem, 8200192);
    cudaMemcpyAsync(output0, tmp_mem, 8200192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_376
// Description:	Constant
// Input:
// Output:
//	- name: Constant_376_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_376(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_376_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_376_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_330
// Description:	Constant
// Input:
// Output:
//	- name: Constant_330_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_330(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_330_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_330_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_410
// Description:	Constant
// Input:
// Output:
//	- name: Constant_410_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_410(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_410_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_410_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_420
// Description:	Constant
// Input:
// Output:
//	- name: Constant_420_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_420(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_420_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_420_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_347
// Description:	Constant
// Input:
// Output:
//	- name: Constant_347_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_347(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_347_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_347_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_118
// Description:	Constant
// Input:
// Output:
//	- name: Constant_118_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_118(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_118_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_118_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_408
// Description:	Constant
// Input:
// Output:
//	- name: Constant_408_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_408(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_408_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_408_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_132
// Description:	Constant
// Input:
// Output:
//	- name: Constant_132_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_132(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_132_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_132_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_414
// Description:	Constant
// Input:
// Output:
//	- name: Constant_414_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_414(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_414_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_414_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_404
// Description:	Constant
// Input:
// Output:
//	- name: Constant_404_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_404(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_404_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_404_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_403
// Description:	Constant
// Input:
// Output:
//	- name: Constant_403_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_403(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_403_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_403_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_454
// Description:	Constant
// Input:
// Output:
//	- name: Constant_454_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_454(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_454_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_454_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_444
// Description:	Constant
// Input:
// Output:
//	- name: Constant_444_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_444(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_444_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_444_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_438
// Description:	Constant
// Input:
// Output:
//	- name: Constant_438_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_438(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_438_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_438_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_358
// Description:	Constant
// Input:
// Output:
//	- name: Constant_358_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_358(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_358_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_358_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_107
// Description:	Constant
// Input:
// Output:
//	- name: Constant_107_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_107_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_434
// Description:	Constant
// Input:
// Output:
//	- name: Constant_434_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_434(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_434_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_434_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_397
// Description:	Constant
// Input:
// Output:
//	- name: Constant_397_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_397(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_397_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_397_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_449
// Description:	Constant
// Input:
// Output:
//	- name: Constant_449_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_449(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_449_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_449_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: Constant_109_0	type: float	shape: Shape{5, 5, 48, 64}
void Constant_float_cuda_Constant_109(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_109_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_109_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[307200];
    bin_file.read(tmp_mem, 307200);
    cudaMemcpyAsync(output0, tmp_mem, 307200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_363
// Description:	Constant
// Input:
// Output:
//	- name: Constant_363_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_363(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_363_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_363_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_418
// Description:	Constant
// Input:
// Output:
//	- name: Constant_418_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_418(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_418_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_418_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_401
// Description:	Constant
// Input:
// Output:
//	- name: Constant_401_0	type: float	shape: Shape{1, 3, 384, 384}
void Constant_float_cuda_Constant_401(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_401_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_401_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_411
// Description:	Constant
// Input:
// Output:
//	- name: Constant_411_0	type: float	shape: Shape{1, 1, 1280, 448}
void Constant_float_cuda_Constant_411(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_411_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_411_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2293760];
    bin_file.read(tmp_mem, 2293760);
    cudaMemcpyAsync(output0, tmp_mem, 2293760, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_142
// Description:	Constant
// Input:
// Output:
//	- name: Constant_142_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_142(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_142_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_142_0 failed.\n");
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
    std::ifstream bin_file("./Constant/Constant_180_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_443
// Description:	Constant
// Input:
// Output:
//	- name: Constant_443_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_443(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_443_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_443_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_116
// Description:	Constant
// Input:
// Output:
//	- name: Constant_116_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_462
// Description:	Constant
// Input:
// Output:
//	- name: Constant_462_0	type: float	shape: Shape{3, 3, 448, 384}
void Constant_float_cuda_Constant_462(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_462_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_462_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6193152];
    bin_file.read(tmp_mem, 6193152);
    cudaMemcpyAsync(output0, tmp_mem, 6193152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_171
// Description:	Constant
// Input:
// Output:
//	- name: Constant_171_0	type: float	shape: Shape{7, 1, 128, 192}
void Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_171_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[688128];
    bin_file.read(tmp_mem, 688128);
    cudaMemcpyAsync(output0, tmp_mem, 688128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: Constant_84_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_481
// Description:	Constant
// Input:
// Output:
//	- name: Constant_481_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_481(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_481_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_481_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_455
// Description:	Constant
// Input:
// Output:
//	- name: Constant_455_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_455(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_455_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_455_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_81
// Description:	Constant
// Input:
// Output:
//	- name: Constant_81_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_81(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_81_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_81_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_113
// Description:	Constant
// Input:
// Output:
//	- name: Constant_113_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_113(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_113_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_113_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_448
// Description:	Constant
// Input:
// Output:
//	- name: Constant_448_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_448(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_448_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_448_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_472
// Description:	Constant
// Input:
// Output:
//	- name: Constant_472_0	type: float	shape: Shape{3, 1, 384, 384}
void Constant_float_cuda_Constant_472(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_472_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_472_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_437
// Description:	Constant
// Input:
// Output:
//	- name: Constant_437_0	type: float	shape: Shape{1, 1, 2048, 320}
void Constant_float_cuda_Constant_437(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_437_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_437_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2621440];
    bin_file.read(tmp_mem, 2621440);
    cudaMemcpyAsync(output0, tmp_mem, 2621440, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_386
// Description:	Constant
// Input:
// Output:
//	- name: Constant_386_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_386(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_386_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_386_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_464
// Description:	Constant
// Input:
// Output:
//	- name: Constant_464_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_464(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_464_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_464_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_452
// Description:	Constant
// Input:
// Output:
//	- name: Constant_452_0	type: float	shape: Shape{3, 1, 384, 384}
void Constant_float_cuda_Constant_452(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_452_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_452_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_478
// Description:	Constant
// Input:
// Output:
//	- name: Constant_478_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_478(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_478_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_478_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_365
// Description:	Constant
// Input:
// Output:
//	- name: Constant_365_0	type: float	shape: Shape{3, 3, 192, 320}
void Constant_float_cuda_Constant_365(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_365_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_365_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2211840];
    bin_file.read(tmp_mem, 2211840);
    cudaMemcpyAsync(output0, tmp_mem, 2211840, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_451
// Description:	Constant
// Input:
// Output:
//	- name: Constant_451_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_451(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_451_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_451_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_453
// Description:	Constant
// Input:
// Output:
//	- name: Constant_453_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_453(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_453_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_453_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_164_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: Constant_19_0	type: float	shape: Shape{80}
void Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[320];
    bin_file.read(tmp_mem, 320);
    cudaMemcpyAsync(output0, tmp_mem, 320, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_471
// Description:	Constant
// Input:
// Output:
//	- name: Constant_471_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_471(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_471_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_471_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_445
// Description:	Constant
// Input:
// Output:
//	- name: Constant_445_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_445(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_445_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_445_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: Constant_61_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_429
// Description:	Constant
// Input:
// Output:
//	- name: Constant_429_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_429(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_429_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_429_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_150
// Description:	Constant
// Input:
// Output:
//	- name: Constant_150_0	type: float	shape: Shape{3, 3, 96, 96}
void Constant_float_cuda_Constant_150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[331776];
    bin_file.read(tmp_mem, 331776);
    cudaMemcpyAsync(output0, tmp_mem, 331776, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_141
// Description:	Constant
// Input:
// Output:
//	- name: Constant_141_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_141(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_141_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_141_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_264
// Description:	Constant
// Input:
// Output:
//	- name: Constant_264_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_264(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_264_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_264_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_468
// Description:	Constant
// Input:
// Output:
//	- name: Constant_468_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_468(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_468_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_468_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_400
// Description:	Constant
// Input:
// Output:
//	- name: Constant_400_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_400(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_400_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_400_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_402
// Description:	Constant
// Input:
// Output:
//	- name: Constant_402_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_402(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_402_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_402_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_278
// Description:	Constant
// Input:
// Output:
//	- name: Constant_278_0	type: float	shape: Shape{1, 1, 768, 160}
void Constant_float_cuda_Constant_278(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_278_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_278_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[491520];
    bin_file.read(tmp_mem, 491520);
    cudaMemcpyAsync(output0, tmp_mem, 491520, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_373
// Description:	Constant
// Input:
// Output:
//	- name: Constant_373_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_373(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_373_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_373_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_378
// Description:	Constant
// Input:
// Output:
//	- name: Constant_378_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_378(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_378_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_378_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_92
// Description:	Constant
// Input:
// Output:
//	- name: Constant_92_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_92(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_92_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_92_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: Constant_144_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_144_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_439
// Description:	Constant
// Input:
// Output:
//	- name: Constant_439_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_439(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_439_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_439_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_485
// Description:	Constant
// Input:
// Output:
//	- name: Constant_485_0	type: float	shape: Shape{1001}
void Constant_float_cuda_Constant_485(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_485_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_485_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_426
// Description:	Constant
// Input:
// Output:
//	- name: Constant_426_0	type: float	shape: Shape{3, 1, 384, 384}
void Constant_float_cuda_Constant_426(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_426_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_426_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_360
// Description:	Constant
// Input:
// Output:
//	- name: Constant_360_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_360(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_360_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_360_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_430
// Description:	Constant
// Input:
// Output:
//	- name: Constant_430_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_430(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_430_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_430_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_137
// Description:	Constant
// Input:
// Output:
//	- name: Constant_137_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_389
// Description:	Constant
// Input:
// Output:
//	- name: Constant_389_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_389(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_389_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_389_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_477
// Description:	Constant
// Input:
// Output:
//	- name: Constant_477_0	type: float	shape: Shape{1, 1, 2048, 192}
void Constant_float_cuda_Constant_477(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_477_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_477_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1572864];
    bin_file.read(tmp_mem, 1572864);
    cudaMemcpyAsync(output0, tmp_mem, 1572864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_267
// Description:	Constant
// Input:
// Output:
//	- name: Constant_267_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_267(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_267_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_267_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: Constant_23_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: Constant_91_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_91(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_470
// Description:	Constant
// Input:
// Output:
//	- name: Constant_470_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_470(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_470_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_470_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_467
// Description:	Constant
// Input:
// Output:
//	- name: Constant_467_0	type: float	shape: Shape{1, 3, 384, 384}
void Constant_float_cuda_Constant_467(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_467_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_467_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_283
// Description:	Constant
// Input:
// Output:
//	- name: Constant_283_0	type: float	shape: Shape{7, 1, 160, 160}
void Constant_float_cuda_Constant_283(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_283_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_283_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_204
// Description:	Constant
// Input:
// Output:
//	- name: Constant_204_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_204(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_204_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_204_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_392
// Description:	Constant
// Input:
// Output:
//	- name: Constant_392_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_392(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_392_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_392_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_409
// Description:	Constant
// Input:
// Output:
//	- name: Constant_409_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_409(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_409_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_409_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_475
// Description:	Constant
// Input:
// Output:
//	- name: Constant_475_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_475(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_475_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_475_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_121
// Description:	Constant
// Input:
// Output:
//	- name: Constant_121_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_121(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_121_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_121_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_120
// Description:	Constant
// Input:
// Output:
//	- name: Constant_120_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_111
// Description:	Constant
// Input:
// Output:
//	- name: Constant_111_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_111(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_111_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_111_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_476
// Description:	Constant
// Input:
// Output:
//	- name: Constant_476_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_476(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_476_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_476_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_318
// Description:	Constant
// Input:
// Output:
//	- name: Constant_318_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_318(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_318_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_318_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_301
// Description:	Constant
// Input:
// Output:
//	- name: Constant_301_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_301(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_301_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_301_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_86
// Description:	Constant
// Input:
// Output:
//	- name: Constant_86_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_86(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_86_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_86_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_473
// Description:	Constant
// Input:
// Output:
//	- name: Constant_473_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_473(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_473_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_473_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_146
// Description:	Constant
// Input:
// Output:
//	- name: Constant_146_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_456
// Description:	Constant
// Input:
// Output:
//	- name: Constant_456_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_456(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_456_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_456_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_110
// Description:	Constant
// Input:
// Output:
//	- name: Constant_110_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_110(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_110_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_110_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_474
// Description:	Constant
// Input:
// Output:
//	- name: Constant_474_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_474(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_474_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_474_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_126_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_428
// Description:	Constant
// Input:
// Output:
//	- name: Constant_428_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_428(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_428_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_428_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_463
// Description:	Constant
// Input:
// Output:
//	- name: Constant_463_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_463(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_463_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_463_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_250
// Description:	Constant
// Input:
// Output:
//	- name: Constant_250_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_250(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_250_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_250_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_65
// Description:	Constant
// Input:
// Output:
//	- name: Constant_65_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_65(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_65_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_65_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_461
// Description:	Constant
// Input:
// Output:
//	- name: Constant_461_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_461(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_461_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_461_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_361
// Description:	Constant
// Input:
// Output:
//	- name: Constant_361_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_361(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_361_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_361_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_108
// Description:	Constant
// Input:
// Output:
//	- name: Constant_108_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_108_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_212
// Description:	Constant
// Input:
// Output:
//	- name: Constant_212_0	type: float	shape: Shape{1, 1, 768, 160}
void Constant_float_cuda_Constant_212(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_212_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_212_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[491520];
    bin_file.read(tmp_mem, 491520);
    cudaMemcpyAsync(output0, tmp_mem, 491520, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_393
// Description:	Constant
// Input:
// Output:
//	- name: Constant_393_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_393(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_393_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_393_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_124
// Description:	Constant
// Input:
// Output:
//	- name: Constant_124_0	type: float	shape: Shape{3, 3, 96, 96}
void Constant_float_cuda_Constant_124(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_124_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_124_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[331776];
    bin_file.read(tmp_mem, 331776);
    cudaMemcpyAsync(output0, tmp_mem, 331776, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: Constant_76_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_479
// Description:	Constant
// Input:
// Output:
//	- name: Constant_479_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_479(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_479_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_479_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_396
// Description:	Constant
// Input:
// Output:
//	- name: Constant_396_0	type: float	shape: Shape{1, 1, 1280, 384}
void Constant_float_cuda_Constant_396(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_396_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_396_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1966080];
    bin_file.read(tmp_mem, 1966080);
    cudaMemcpyAsync(output0, tmp_mem, 1966080, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_311
// Description:	Constant
// Input:
// Output:
//	- name: Constant_311_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_311(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_311_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_311_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_328
// Description:	Constant
// Input:
// Output:
//	- name: Constant_328_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_328(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_328_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_328_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_343
// Description:	Constant
// Input:
// Output:
//	- name: Constant_343_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_343(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_343_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_343_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_232
// Description:	Constant
// Input:
// Output:
//	- name: Constant_232_0	type: float	shape: Shape{7, 1, 160, 160}
void Constant_float_cuda_Constant_232(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_232_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_232_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_187
// Description:	Constant
// Input:
// Output:
//	- name: Constant_187_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_187(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_187_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_187_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_325
// Description:	Constant
// Input:
// Output:
//	- name: Constant_325_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_325(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_325_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_325_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_321
// Description:	Constant
// Input:
// Output:
//	- name: Constant_321_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_321(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_321_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_321_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_322
// Description:	Constant
// Input:
// Output:
//	- name: Constant_322_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_322(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_322_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_322_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_433
// Description:	Constant
// Input:
// Output:
//	- name: Constant_433_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_433(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_433_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_433_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_457
// Description:	Constant
// Input:
// Output:
//	- name: Constant_457_0	type: float	shape: Shape{1, 1, 2048, 448}
void Constant_float_cuda_Constant_457(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_457_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_457_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3670016];
    bin_file.read(tmp_mem, 3670016);
    cudaMemcpyAsync(output0, tmp_mem, 3670016, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_105
// Description:	Constant
// Input:
// Output:
//	- name: Constant_105_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_105(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_105_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_105_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_319
// Description:	Constant
// Input:
// Output:
//	- name: Constant_319_0	type: float	shape: Shape{1, 7, 192, 192}
void Constant_float_cuda_Constant_319(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_319_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_319_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_189
// Description:	Constant
// Input:
// Output:
//	- name: Constant_189_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_189(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_189_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_189_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_183
// Description:	Constant
// Input:
// Output:
//	- name: Constant_183_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_183(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_183_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_295
// Description:	Constant
// Input:
// Output:
//	- name: Constant_295_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_295(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_295_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_295_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_300
// Description:	Constant
// Input:
// Output:
//	- name: Constant_300_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_300(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_300_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_300_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_353
// Description:	Constant
// Input:
// Output:
//	- name: Constant_353_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_353(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_353_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_353_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_138
// Description:	Constant
// Input:
// Output:
//	- name: Constant_138_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_138(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_138_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_138_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: Constant_193_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_193(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_193_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_101
// Description:	Constant
// Input:
// Output:
//	- name: Constant_101_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_101(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_101_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_101_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_136
// Description:	Constant
// Input:
// Output:
//	- name: Constant_136_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_136(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_136_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_136_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: Constant_87_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_87(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_87_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_87_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_166
// Description:	Constant
// Input:
// Output:
//	- name: Constant_166_0	type: float	shape: Shape{1, 7, 128, 128}
void Constant_float_cuda_Constant_166(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_166_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_166_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[458752];
    bin_file.read(tmp_mem, 458752);
    cudaMemcpyAsync(output0, tmp_mem, 458752, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_425
// Description:	Constant
// Input:
// Output:
//	- name: Constant_425_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_425(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_425_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_425_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_240
// Description:	Constant
// Input:
// Output:
//	- name: Constant_240_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_240(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_240_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_240_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_285
// Description:	Constant
// Input:
// Output:
//	- name: Constant_285_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_285(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_285_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_285_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: Constant_26_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_26_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_352
// Description:	Constant
// Input:
// Output:
//	- name: Constant_352_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_352(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_352_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_352_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_73
// Description:	Constant
// Input:
// Output:
//	- name: Constant_73_0	type: float	shape: Shape{5, 5, 48, 64}
void Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_73_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[307200];
    bin_file.read(tmp_mem, 307200);
    cudaMemcpyAsync(output0, tmp_mem, 307200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: Constant_52_0	type: float	shape: Shape{3, 3, 96, 96}
void Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[331776];
    bin_file.read(tmp_mem, 331776);
    cudaMemcpyAsync(output0, tmp_mem, 331776, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_119
// Description:	Constant
// Input:
// Output:
//	- name: Constant_119_0	type: float	shape: Shape{3, 3, 64, 96}
void Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_119_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[221184];
    bin_file.read(tmp_mem, 221184);
    cudaMemcpyAsync(output0, tmp_mem, 221184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_115
// Description:	Constant
// Input:
// Output:
//	- name: Constant_115_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_115(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_115_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_115_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_315
// Description:	Constant
// Input:
// Output:
//	- name: Constant_315_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_315(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_315_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_315_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_208
// Description:	Constant
// Input:
// Output:
//	- name: Constant_208_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_208(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_208_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_208_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_104_0	type: float	shape: Shape{1, 1, 288, 48}
void Constant_float_cuda_Constant_104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[55296];
    bin_file.read(tmp_mem, 55296);
    cudaMemcpyAsync(output0, tmp_mem, 55296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_466
// Description:	Constant
// Input:
// Output:
//	- name: Constant_466_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_466(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_466_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_466_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_114
// Description:	Constant
// Input:
// Output:
//	- name: Constant_114_0	type: float	shape: Shape{1, 1, 288, 64}
void Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_114_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[73728];
    bin_file.read(tmp_mem, 73728);
    cudaMemcpyAsync(output0, tmp_mem, 73728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_375
// Description:	Constant
// Input:
// Output:
//	- name: Constant_375_0	type: float	shape: Shape{1, 7, 192, 192}
void Constant_float_cuda_Constant_375(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_375_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_375_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_339
// Description:	Constant
// Input:
// Output:
//	- name: Constant_339_0	type: float	shape: Shape{1, 7, 192, 192}
void Constant_float_cuda_Constant_339(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_339_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_339_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_213
// Description:	Constant
// Input:
// Output:
//	- name: Constant_213_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_213(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_213_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_213_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_344
// Description:	Constant
// Input:
// Output:
//	- name: Constant_344_0	type: float	shape: Shape{7, 1, 192, 192}
void Constant_float_cuda_Constant_344(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_344_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_344_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_112_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_77
// Description:	Constant
// Input:
// Output:
//	- name: Constant_77_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_77(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_77_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_77_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_95
// Description:	Constant
// Input:
// Output:
//	- name: Constant_95_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_95(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_95_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_95_0 failed.\n");
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
//	- name: Constant_158_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_154
// Description:	Constant
// Input:
// Output:
//	- name: Constant_154_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_154(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_154_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_154_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: Constant_117_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_117(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_117_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_117_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_70
// Description:	Constant
// Input:
// Output:
//	- name: Constant_70_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_70(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_70_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_70_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_177
// Description:	Constant
// Input:
// Output:
//	- name: Constant_177_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_71
// Description:	Constant
// Input:
// Output:
//	- name: Constant_71_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_71(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_71_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_71_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_442
// Description:	Constant
// Input:
// Output:
//	- name: Constant_442_0	type: float	shape: Shape{1, 1, 2048, 384}
void Constant_float_cuda_Constant_442(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_442_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_442_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3145728];
    bin_file.read(tmp_mem, 3145728);
    cudaMemcpyAsync(output0, tmp_mem, 3145728, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_69
// Description:	Constant
// Input:
// Output:
//	- name: Constant_69_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_69(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_69_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_69_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_85
// Description:	Constant
// Input:
// Output:
//	- name: Constant_85_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_85(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_85_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_85_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_371
// Description:	Constant
// Input:
// Output:
//	- name: Constant_371_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_371(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_371_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_371_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_90
// Description:	Constant
// Input:
// Output:
//	- name: Constant_90_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_173
// Description:	Constant
// Input:
// Output:
//	- name: Constant_173_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_173(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_173_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_173_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_130
// Description:	Constant
// Input:
// Output:
//	- name: Constant_130_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_130(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_130_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_130_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_255
// Description:	Constant
// Input:
// Output:
//	- name: Constant_255_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_255(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_255_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_255_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: Constant_67_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_67(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_67_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_67_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_147
// Description:	Constant
// Input:
// Output:
//	- name: Constant_147_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_147(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_147_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_147_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_156
// Description:	Constant
// Input:
// Output:
//	- name: Constant_156_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_156_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_191
// Description:	Constant
// Input:
// Output:
//	- name: Constant_191_0	type: float	shape: Shape{7, 1, 128, 128}
void Constant_float_cuda_Constant_191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[458752];
    bin_file.read(tmp_mem, 458752);
    cudaMemcpyAsync(output0, tmp_mem, 458752, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_145
// Description:	Constant
// Input:
// Output:
//	- name: Constant_145_0	type: float	shape: Shape{3, 3, 64, 96}
void Constant_float_cuda_Constant_145(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_145_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_145_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[221184];
    bin_file.read(tmp_mem, 221184);
    cudaMemcpyAsync(output0, tmp_mem, 221184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: Constant_9_0	type: float	shape: Shape{32}
void Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[128];
    bin_file.read(tmp_mem, 128);
    cudaMemcpyAsync(output0, tmp_mem, 128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_211
// Description:	Constant
// Input:
// Output:
//	- name: Constant_211_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_211(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_211_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_211_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_181
// Description:	Constant
// Input:
// Output:
//	- name: Constant_181_0	type: float	shape: Shape{7, 1, 128, 128}
void Constant_float_cuda_Constant_181(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_181_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_181_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[458752];
    bin_file.read(tmp_mem, 458752);
    cudaMemcpyAsync(output0, tmp_mem, 458752, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_160
// Description:	Constant
// Input:
// Output:
//	- name: Constant_160_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_160(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_160_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_160_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_200
// Description:	Constant
// Input:
// Output:
//	- name: Constant_200_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_200(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_200_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_200_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_170_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_199
// Description:	Constant
// Input:
// Output:
//	- name: Constant_199_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_199(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_199_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_199_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_406
// Description:	Constant
// Input:
// Output:
//	- name: Constant_406_0	type: float	shape: Shape{3, 1, 384, 384}
void Constant_float_cuda_Constant_406(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_406_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_406_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1769472];
    bin_file.read(tmp_mem, 1769472);
    cudaMemcpyAsync(output0, tmp_mem, 1769472, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_316
// Description:	Constant
// Input:
// Output:
//	- name: Constant_316_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_316(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_316_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_316_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_175
// Description:	Constant
// Input:
// Output:
//	- name: Constant_175_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_175(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_175_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_175_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_174
// Description:	Constant
// Input:
// Output:
//	- name: Constant_174_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_174_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_407
// Description:	Constant
// Input:
// Output:
//	- name: Constant_407_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_407(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_407_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_407_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_413
// Description:	Constant
// Input:
// Output:
//	- name: Constant_413_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_413(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_413_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_413_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_172
// Description:	Constant
// Input:
// Output:
//	- name: Constant_172_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_172(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_172_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_172_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_203
// Description:	Constant
// Input:
// Output:
//	- name: Constant_203_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_203(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_203_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_203_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_159
// Description:	Constant
// Input:
// Output:
//	- name: Constant_159_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_159(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_159_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_159_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_293
// Description:	Constant
// Input:
// Output:
//	- name: Constant_293_0	type: float	shape: Shape{7, 1, 160, 160}
void Constant_float_cuda_Constant_293(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_293_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_293_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_209
// Description:	Constant
// Input:
// Output:
//	- name: Constant_209_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_209(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_209_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_209_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_151
// Description:	Constant
// Input:
// Output:
//	- name: Constant_151_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_151(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_151_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_151_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_253
// Description:	Constant
// Input:
// Output:
//	- name: Constant_253_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_253(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_253_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_253_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_184
// Description:	Constant
// Input:
// Output:
//	- name: Constant_184_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_184(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_184_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_184_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_256
// Description:	Constant
// Input:
// Output:
//	- name: Constant_256_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_256(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_256_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_256_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_254
// Description:	Constant
// Input:
// Output:
//	- name: Constant_254_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_254_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_194
// Description:	Constant
// Input:
// Output:
//	- name: Constant_194_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_194_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_227
// Description:	Constant
// Input:
// Output:
//	- name: Constant_227_0	type: float	shape: Shape{1, 1, 768, 160}
void Constant_float_cuda_Constant_227(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_227_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_227_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[491520];
    bin_file.read(tmp_mem, 491520);
    cudaMemcpyAsync(output0, tmp_mem, 491520, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_412
// Description:	Constant
// Input:
// Output:
//	- name: Constant_412_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_412(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_412_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_412_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_388
// Description:	Constant
// Input:
// Output:
//	- name: Constant_388_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_388(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_388_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_388_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_188
// Description:	Constant
// Input:
// Output:
//	- name: Constant_188_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_188_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_233
// Description:	Constant
// Input:
// Output:
//	- name: Constant_233_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_233(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_233_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_233_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_235
// Description:	Constant
// Input:
// Output:
//	- name: Constant_235_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_235(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_235_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_235_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_234
// Description:	Constant
// Input:
// Output:
//	- name: Constant_234_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_234(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_234_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_234_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_237
// Description:	Constant
// Input:
// Output:
//	- name: Constant_237_0	type: float	shape: Shape{1, 7, 160, 160}
void Constant_float_cuda_Constant_237(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_237_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_237_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_131
// Description:	Constant
// Input:
// Output:
//	- name: Constant_131_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_131(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_131_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_131_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_238
// Description:	Constant
// Input:
// Output:
//	- name: Constant_238_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_238(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_238_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_238_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_241
// Description:	Constant
// Input:
// Output:
//	- name: Constant_241_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_241(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_241_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_241_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_239
// Description:	Constant
// Input:
// Output:
//	- name: Constant_239_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_239(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_239_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_239_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_242
// Description:	Constant
// Input:
// Output:
//	- name: Constant_242_0	type: float	shape: Shape{7, 1, 160, 160}
void Constant_float_cuda_Constant_242(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_242_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_242_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_282
// Description:	Constant
// Input:
// Output:
//	- name: Constant_282_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_282(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_282_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_282_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_354
// Description:	Constant
// Input:
// Output:
//	- name: Constant_354_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_354(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_354_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_354_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_243
// Description:	Constant
// Input:
// Output:
//	- name: Constant_243_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_243(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_243_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_243_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_309
// Description:	Constant
// Input:
// Output:
//	- name: Constant_309_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_309(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_309_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_309_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_97
// Description:	Constant
// Input:
// Output:
//	- name: Constant_97_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_97(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_97_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_97_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_245
// Description:	Constant
// Input:
// Output:
//	- name: Constant_245_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_245(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_245_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_245_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_440
// Description:	Constant
// Input:
// Output:
//	- name: Constant_440_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_440(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_440_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_440_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_248
// Description:	Constant
// Input:
// Output:
//	- name: Constant_248_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_248(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_248_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_248_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_152
// Description:	Constant
// Input:
// Output:
//	- name: Constant_152_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_152(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_152_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_152_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_231
// Description:	Constant
// Input:
// Output:
//	- name: Constant_231_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_231(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_231_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_231_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_288
// Description:	Constant
// Input:
// Output:
//	- name: Constant_288_0	type: float	shape: Shape{1, 7, 160, 160}
void Constant_float_cuda_Constant_288(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_288_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_288_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_312
// Description:	Constant
// Input:
// Output:
//	- name: Constant_312_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_312(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_312_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_312_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_244
// Description:	Constant
// Input:
// Output:
//	- name: Constant_244_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_244(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_244_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_244_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_289
// Description:	Constant
// Input:
// Output:
//	- name: Constant_289_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_289(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_289_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_289_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_271
// Description:	Constant
// Input:
// Output:
//	- name: Constant_271_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_271(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_271_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_271_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_247
// Description:	Constant
// Input:
// Output:
//	- name: Constant_247_0	type: float	shape: Shape{1, 7, 160, 192}
void Constant_float_cuda_Constant_247(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_247_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_247_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[860160];
    bin_file.read(tmp_mem, 860160);
    cudaMemcpyAsync(output0, tmp_mem, 860160, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_252
// Description:	Constant
// Input:
// Output:
//	- name: Constant_252_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_252(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_252_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_252_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_249
// Description:	Constant
// Input:
// Output:
//	- name: Constant_249_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_249(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_249_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_249_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_313
// Description:	Constant
// Input:
// Output:
//	- name: Constant_313_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_313(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_313_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_313_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_216
// Description:	Constant
// Input:
// Output:
//	- name: Constant_216_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_216(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_216_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_216_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_269
// Description:	Constant
// Input:
// Output:
//	- name: Constant_269_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_269(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_269_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_269_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_224
// Description:	Constant
// Input:
// Output:
//	- name: Constant_224_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_224(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_224_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_224_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_215
// Description:	Constant
// Input:
// Output:
//	- name: Constant_215_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_215(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_215_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_215_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_214
// Description:	Constant
// Input:
// Output:
//	- name: Constant_214_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_214(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_214_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_214_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_277
// Description:	Constant
// Input:
// Output:
//	- name: Constant_277_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_277(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_277_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_277_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_217
// Description:	Constant
// Input:
// Output:
//	- name: Constant_217_0	type: float	shape: Shape{1, 7, 160, 160}
void Constant_float_cuda_Constant_217(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_217_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_217_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_218
// Description:	Constant
// Input:
// Output:
//	- name: Constant_218_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_218_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_220
// Description:	Constant
// Input:
// Output:
//	- name: Constant_220_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_220(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_220_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_220_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_222
// Description:	Constant
// Input:
// Output:
//	- name: Constant_222_0	type: float	shape: Shape{7, 1, 160, 192}
void Constant_float_cuda_Constant_222(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_222_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_222_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[860160];
    bin_file.read(tmp_mem, 860160);
    cudaMemcpyAsync(output0, tmp_mem, 860160, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_103
// Description:	Constant
// Input:
// Output:
//	- name: Constant_103_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_103(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_103_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_103_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_292
// Description:	Constant
// Input:
// Output:
//	- name: Constant_292_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_292(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_292_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_292_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_324
// Description:	Constant
// Input:
// Output:
//	- name: Constant_324_0	type: float	shape: Shape{7, 1, 192, 192}
void Constant_float_cuda_Constant_324(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_324_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_324_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_182_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_225
// Description:	Constant
// Input:
// Output:
//	- name: Constant_225_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_225(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_225_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_225_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_431
// Description:	Constant
// Input:
// Output:
//	- name: Constant_431_0	type: float	shape: Shape{1, 1, 1280, 192}
void Constant_float_cuda_Constant_431(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_431_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_431_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[983040];
    bin_file.read(tmp_mem, 983040);
    cudaMemcpyAsync(output0, tmp_mem, 983040, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_63
// Description:	Constant
// Input:
// Output:
//	- name: Constant_63_0	type: float	shape: Shape{1, 1, 256, 64}
void Constant_float_cuda_Constant_63(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_63_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_63_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_210
// Description:	Constant
// Input:
// Output:
//	- name: Constant_210_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_210(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_210_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_210_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_303
// Description:	Constant
// Input:
// Output:
//	- name: Constant_303_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_303(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_303_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_303_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_427
// Description:	Constant
// Input:
// Output:
//	- name: Constant_427_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_427(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_427_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_427_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_307
// Description:	Constant
// Input:
// Output:
//	- name: Constant_307_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_307(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_307_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_307_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_149
// Description:	Constant
// Input:
// Output:
//	- name: Constant_149_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_149_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_306
// Description:	Constant
// Input:
// Output:
//	- name: Constant_306_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_306(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_306_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_306_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_305
// Description:	Constant
// Input:
// Output:
//	- name: Constant_305_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_305(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_305_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_305_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_372
// Description:	Constant
// Input:
// Output:
//	- name: Constant_372_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_372(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_372_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_372_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_279
// Description:	Constant
// Input:
// Output:
//	- name: Constant_279_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_279(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_279_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_279_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_127
// Description:	Constant
// Input:
// Output:
//	- name: Constant_127_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_127(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_127_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_127_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_281
// Description:	Constant
// Input:
// Output:
//	- name: Constant_281_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_281(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_281_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_281_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_302
// Description:	Constant
// Input:
// Output:
//	- name: Constant_302_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_302(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_302_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_302_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_287
// Description:	Constant
// Input:
// Output:
//	- name: Constant_287_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_287(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_287_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_287_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_435
// Description:	Constant
// Input:
// Output:
//	- name: Constant_435_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_435(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_435_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_435_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_291
// Description:	Constant
// Input:
// Output:
//	- name: Constant_291_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_291(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_291_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_291_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_394
// Description:	Constant
// Input:
// Output:
//	- name: Constant_394_0	type: float	shape: Shape{320}
void Constant_float_cuda_Constant_394(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_394_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_394_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1280];
    bin_file.read(tmp_mem, 1280);
    cudaMemcpyAsync(output0, tmp_mem, 1280, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_41
// Description:	Constant
// Input:
// Output:
//	- name: Constant_41_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_41(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_41_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_41_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_294
// Description:	Constant
// Input:
// Output:
//	- name: Constant_294_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_294(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_294_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_294_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_297
// Description:	Constant
// Input:
// Output:
//	- name: Constant_297_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_297(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_297_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_297_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_298
// Description:	Constant
// Input:
// Output:
//	- name: Constant_298_0	type: float	shape: Shape{1, 7, 160, 192}
void Constant_float_cuda_Constant_298(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_298_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_298_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[860160];
    bin_file.read(tmp_mem, 860160);
    cudaMemcpyAsync(output0, tmp_mem, 860160, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: Constant_93_0	type: float	shape: Shape{1, 1, 256, 64}
void Constant_float_cuda_Constant_93(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_299
// Description:	Constant
// Input:
// Output:
//	- name: Constant_299_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_299(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_299_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_299_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_79
// Description:	Constant
// Input:
// Output:
//	- name: Constant_79_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_79(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_79_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_79_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_266
// Description:	Constant
// Input:
// Output:
//	- name: Constant_266_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_266(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_266_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_266_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_268
// Description:	Constant
// Input:
// Output:
//	- name: Constant_268_0	type: float	shape: Shape{1, 7, 160, 160}
void Constant_float_cuda_Constant_268(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_268_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_268_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[716800];
    bin_file.read(tmp_mem, 716800);
    cudaMemcpyAsync(output0, tmp_mem, 716800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_270
// Description:	Constant
// Input:
// Output:
//	- name: Constant_270_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_270(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_270_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_270_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_273
// Description:	Constant
// Input:
// Output:
//	- name: Constant_273_0	type: float	shape: Shape{7, 1, 160, 192}
void Constant_float_cuda_Constant_273(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_273_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_273_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[860160];
    bin_file.read(tmp_mem, 860160);
    cudaMemcpyAsync(output0, tmp_mem, 860160, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_424
// Description:	Constant
// Input:
// Output:
//	- name: Constant_424_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_424(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_424_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_424_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_333
// Description:	Constant
// Input:
// Output:
//	- name: Constant_333_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_333(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_333_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_333_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_274
// Description:	Constant
// Input:
// Output:
//	- name: Constant_274_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_274(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_274_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_274_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_276
// Description:	Constant
// Input:
// Output:
//	- name: Constant_276_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_276(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_276_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_276_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_275
// Description:	Constant
// Input:
// Output:
//	- name: Constant_275_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_275(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_275_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_275_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_419
// Description:	Constant
// Input:
// Output:
//	- name: Constant_419_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_419(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_419_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_419_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_195
// Description:	Constant
// Input:
// Output:
//	- name: Constant_195_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_195(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_195_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_258
// Description:	Constant
// Input:
// Output:
//	- name: Constant_258_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_258(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_258_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_258_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_262
// Description:	Constant
// Input:
// Output:
//	- name: Constant_262_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_262(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_262_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_262_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_260
// Description:	Constant
// Input:
// Output:
//	- name: Constant_260_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_260(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_260_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_260_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_355
// Description:	Constant
// Input:
// Output:
//	- name: Constant_355_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_355(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_355_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_355_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_362
// Description:	Constant
// Input:
// Output:
//	- name: Constant_362_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_362(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_362_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_362_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_357
// Description:	Constant
// Input:
// Output:
//	- name: Constant_357_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_357(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_357_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_357_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_230
// Description:	Constant
// Input:
// Output:
//	- name: Constant_230_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_230(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_230_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_230_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_356
// Description:	Constant
// Input:
// Output:
//	- name: Constant_356_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_356(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_356_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_356_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_329
// Description:	Constant
// Input:
// Output:
//	- name: Constant_329_0	type: float	shape: Shape{1, 1, 768, 192}
void Constant_float_cuda_Constant_329(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_329_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_329_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_334
// Description:	Constant
// Input:
// Output:
//	- name: Constant_334_0	type: float	shape: Shape{7, 1, 192, 192}
void Constant_float_cuda_Constant_334(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_334_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_334_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_335
// Description:	Constant
// Input:
// Output:
//	- name: Constant_335_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_335(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_335_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_335_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_106
// Description:	Constant
// Input:
// Output:
//	- name: Constant_106_0	type: float	shape: Shape{48}
void Constant_float_cuda_Constant_106(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_106_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_106_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[192];
    bin_file.read(tmp_mem, 192);
    cudaMemcpyAsync(output0, tmp_mem, 192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_338
// Description:	Constant
// Input:
// Output:
//	- name: Constant_338_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_338(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_338_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_338_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_336
// Description:	Constant
// Input:
// Output:
//	- name: Constant_336_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_336(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_336_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_336_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_415
// Description:	Constant
// Input:
// Output:
//	- name: Constant_415_0	type: float	shape: Shape{448}
void Constant_float_cuda_Constant_415(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_415_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_415_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1792];
    bin_file.read(tmp_mem, 1792);
    cudaMemcpyAsync(output0, tmp_mem, 1792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_342
// Description:	Constant
// Input:
// Output:
//	- name: Constant_342_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_342(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_342_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_342_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_75
// Description:	Constant
// Input:
// Output:
//	- name: Constant_75_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_75(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_75_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_75_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_341
// Description:	Constant
// Input:
// Output:
//	- name: Constant_341_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_341(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_341_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_341_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_348
// Description:	Constant
// Input:
// Output:
//	- name: Constant_348_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_348(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_348_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_348_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_446
// Description:	Constant
// Input:
// Output:
//	- name: Constant_446_0	type: float	shape: Shape{384}
void Constant_float_cuda_Constant_446(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_446_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_446_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1536];
    bin_file.read(tmp_mem, 1536);
    cudaMemcpyAsync(output0, tmp_mem, 1536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_153
// Description:	Constant
// Input:
// Output:
//	- name: Constant_153_0	type: float	shape: Shape{96}
void Constant_float_cuda_Constant_153(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_153_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_153_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[384];
    bin_file.read(tmp_mem, 384);
    cudaMemcpyAsync(output0, tmp_mem, 384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_346
// Description:	Constant
// Input:
// Output:
//	- name: Constant_346_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_346(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_346_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_346_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_381
// Description:	Constant
// Input:
// Output:
//	- name: Constant_381_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_381(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_381_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_381_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_296
// Description:	Constant
// Input:
// Output:
//	- name: Constant_296_0	type: float	shape: Shape{160}
void Constant_float_cuda_Constant_296(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_296_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_296_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[640];
    bin_file.read(tmp_mem, 640);
    cudaMemcpyAsync(output0, tmp_mem, 640, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Convolution_814
// Description:	Convolution
// Input:
//	- name: Concat_810_0	type: float	shape: Shape{32, 1280, 8, 8}
//	- name: Reshape_813_0	type: float	shape: Shape{384, 1280, 1, 1}
// Output:
//	- name: Convolution_814_0	type: float	shape: Shape{32, 384, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_814(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 1280, 1, 1));
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
// Node name:	Constant_349
// Description:	Constant
// Input:
// Output:
//	- name: Constant_349_0	type: float	shape: Shape{1, 7, 192, 192}
void Constant_float_cuda_Constant_349(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_349_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_349_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1032192];
    bin_file.read(tmp_mem, 1032192);
    cudaMemcpyAsync(output0, tmp_mem, 1032192, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Convolution_697
// Description:	Convolution
// Input:
//	- name: Relu_695_0	type: float	shape: Shape{32, 160, 17, 17}
//	- name: Reshape_696_0	type: float	shape: Shape{192, 160, 1, 7}
// Output:
//	- name: Convolution_697_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_697(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 160, 1, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Concat_538
// Description:	Concat
// Input:
//	- name: Relu_521_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Relu_532_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Relu_537_0	type: float	shape: Shape{32, 96, 35, 35}
//	- name: Relu_529_0	type: float	shape: Shape{32, 32, 35, 35}
// Output:
//	- name: Concat_538_0	type: float	shape: Shape{32, 256, 35, 35}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_float_cuda_Concat_538(float* input0, float* input1, float* input2, float* input3, float* output0)
{
    uint32_t inputs_strides[] = {78400, 78400, 117600, 39200};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 10035200)
    {
        uint32_t block_id = tid / 313600;
        uint32_t block_idx = tid % 313600;
        uint32_t output_idx = block_id * 313600 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
        if(block_idx < inputs_strides[3])
        {
            output0[output_idx] = input3[block_id * inputs_strides[3] + block_idx];
            return;
        }
        block_idx -= inputs_strides[3];
    }

}
extern void Concat_float_float_float_float_float_cuda_Concat_538_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    Concat_float_float_float_float_float_cuda_Concat_538<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}
// Node name:	Reshape_696
// Description:	Reshape
// Input:
//	- name: Constant_247_0	type: float	shape: Shape{1, 7, 160, 192}
// Output:
//	- name: Reshape_696_0	type: float	shape: Shape{192, 160, 1, 7}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_696(float* input0, float* output0)
{
    uint32_t input_strides0 = 30720;
    uint32_t input_strides1 = 192;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 7;
    uint32_t trans_strides2 = 1120;
    size_t nx = 192;
    size_t ny = 160;
    size_t nz = 7;
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
extern void Reshape_float_float_cuda_Reshape_696_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_696<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_626
// Description:	BatchNormInference
// Input:
//	- name: Constant_177_0	type: float	shape: Shape{128}
//	- name: Constant_178_0	type: float	shape: Shape{128}
//	- name: Convolution_622_0	type: float	shape: Shape{32, 128, 17, 17}
//	- name: Constant_179_0	type: float	shape: Shape{128}
//	- name: Constant_180_0	type: float	shape: Shape{128}
// Output:
//	- name: BatchNormInference_626_0	type: float	shape: Shape{32, 128, 17, 17}
extern "C" __launch_bounds__(289) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 17 * 17;
    const int c_id = blockIdx.x % 128;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 17 * 17; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_643
// Description:	Convolution
// Input:
//	- name: Relu_640_0	type: float	shape: Shape{32, 128, 17, 17}
//	- name: Reshape_642_0	type: float	shape: Shape{192, 128, 7, 1}
// Output:
//	- name: Convolution_643_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_643(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 128, 7, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 3, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_580
// Description:	Convolution
// Input:
//	- name: AvgPool_575_0	type: float	shape: Shape{32, 288, 35, 35}
//	- name: Reshape_579_0	type: float	shape: Shape{64, 288, 1, 1}
// Output:
//	- name: Convolution_580_0	type: float	shape: Shape{32, 64, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_580(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 288, 1, 1));
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
// Node name:	Convolution_492
// Description:	Convolution
// Input:
//	- name: Relu_490_0	type: float	shape: Shape{32, 32, 149, 149}
//	- name: Reshape_491_0	type: float	shape: Shape{32, 32, 3, 3}
// Output:
//	- name: Convolution_492_0	type: float	shape: Shape{32, 32, 147, 147}
void Convolution_float_float_float_cuda_lib_Convolution_492(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 32, 149, 149));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 32, 147, 147));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 32, 3, 3));
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
// Node name:	BatchNormInference_497
// Description:	BatchNormInference
// Input:
//	- name: Constant_13_0	type: float	shape: Shape{64}
//	- name: Constant_14_0	type: float	shape: Shape{64}
//	- name: Convolution_496_0	type: float	shape: Shape{32, 64, 147, 147}
//	- name: Constant_15_0	type: float	shape: Shape{64}
//	- name: Constant_16_0	type: float	shape: Shape{64}
// Output:
//	- name: BatchNormInference_497_0	type: float	shape: Shape{32, 64, 147, 147}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_497(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 147 * 147;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 147 * 147; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_497_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_497<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	MaxPool_499
// Description:	MaxPool
// Input:
//	- name: Relu_498_0	type: float	shape: Shape{32, 64, 147, 147}
// Output:
//	- name: MaxPool_499_0	type: float	shape: Shape{32, 64, 73, 73}
void MaxPool_float_float_cuda_lib_MaxPool_499(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 147, 147));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 73, 73));
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
// Node name:	Reshape_525
// Description:	Reshape
// Input:
//	- name: Constant_37_0	type: float	shape: Shape{5, 5, 48, 64}
// Output:
//	- name: Reshape_525_0	type: float	shape: Shape{64, 48, 5, 5}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_525(float* input0, float* output0)
{
    uint32_t input_strides0 = 3072;
    uint32_t input_strides1 = 64;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 25;
    uint32_t trans_strides2 = 1200;
    size_t nx = 64;
    size_t ny = 48;
    size_t nz = 25;
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
extern void Reshape_float_float_cuda_Reshape_525_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_525<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_853
// Description:	Reshape
// Input:
//	- name: Constant_457_0	type: float	shape: Shape{1, 1, 2048, 448}
// Output:
//	- name: Reshape_853_0	type: float	shape: Shape{448, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_853(float* input0, float* output0)
{
    uint32_t input_strides0 = 448;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 2048;
    size_t nx = 448;
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
extern void Reshape_float_float_cuda_Reshape_853_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_853<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	AvgPool_545
// Description:	AvgPool
// Input:
//	- name: Concat_538_0	type: float	shape: Shape{32, 256, 35, 35}
// Output:
//	- name: AvgPool_545_0	type: float	shape: Shape{32, 256, 35, 35}
void AvgPool_float_float_cuda_lib_AvgPool_545(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 35, 35));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 35, 35));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Convolution_807
// Description:	Convolution
// Input:
//	- name: Relu_805_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Reshape_806_0	type: float	shape: Shape{192, 192, 3, 3}
// Output:
//	- name: Convolution_807_0	type: float	shape: Shape{32, 192, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_807(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 192, 3, 3));
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
// Node name:	BatchNormInference_493
// Description:	BatchNormInference
// Input:
//	- name: Constant_8_0	type: float	shape: Shape{32}
//	- name: Constant_9_0	type: float	shape: Shape{32}
//	- name: Convolution_492_0	type: float	shape: Shape{32, 32, 147, 147}
//	- name: Constant_10_0	type: float	shape: Shape{32}
//	- name: Constant_11_0	type: float	shape: Shape{32}
// Output:
//	- name: BatchNormInference_493_0	type: float	shape: Shape{32, 32, 147, 147}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_493(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 147 * 147;
    const int c_id = blockIdx.x % 32;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 147 * 147; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_493_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_493<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	AvgPool_575
// Description:	AvgPool
// Input:
//	- name: Concat_568_0	type: float	shape: Shape{32, 288, 35, 35}
// Output:
//	- name: AvgPool_575_0	type: float	shape: Shape{32, 288, 35, 35}
void AvgPool_float_float_cuda_lib_AvgPool_575(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Reshape_495
// Description:	Reshape
// Input:
//	- name: Constant_12_0	type: float	shape: Shape{3, 3, 32, 64}
// Output:
//	- name: Reshape_495_0	type: float	shape: Shape{64, 32, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_495(float* input0, float* output0)
{
    uint32_t input_strides0 = 2048;
    uint32_t input_strides1 = 64;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 288;
    size_t nx = 64;
    size_t ny = 32;
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
extern void Reshape_float_float_cuda_Reshape_495_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_495<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_654
// Description:	Reshape
// Input:
//	- name: Constant_196_0	type: float	shape: Shape{1, 7, 128, 192}
// Output:
//	- name: Reshape_654_0	type: float	shape: Shape{192, 128, 1, 7}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_654(float* input0, float* output0)
{
    uint32_t input_strides0 = 24576;
    uint32_t input_strides1 = 192;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 7;
    uint32_t trans_strides2 = 896;
    size_t nx = 192;
    size_t ny = 128;
    size_t nz = 7;
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
extern void Reshape_float_float_cuda_Reshape_654_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_654<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_531
// Description:	BatchNormInference
// Input:
//	- name: Constant_48_0	type: float	shape: Shape{96}
//	- name: Constant_49_0	type: float	shape: Shape{96}
//	- name: Convolution_528_0	type: float	shape: Shape{32, 96, 35, 35}
//	- name: Constant_50_0	type: float	shape: Shape{96}
//	- name: Constant_51_0	type: float	shape: Shape{96}
// Output:
//	- name: BatchNormInference_531_0	type: float	shape: Shape{32, 96, 35, 35}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 35 * 35;
    const int c_id = blockIdx.x % 96;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 35 * 35; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_491
// Description:	Reshape
// Input:
//	- name: Constant_7_0	type: float	shape: Shape{3, 3, 32, 32}
// Output:
//	- name: Reshape_491_0	type: float	shape: Shape{32, 32, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_491(float* input0, float* output0)
{
    uint32_t input_strides0 = 1024;
    uint32_t input_strides1 = 32;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 288;
    size_t nx = 32;
    size_t ny = 32;
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
extern void Reshape_float_float_cuda_Reshape_491_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_491<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_604
// Description:	BatchNormInference
// Input:
//	- name: Constant_136_0	type: float	shape: Shape{384}
//	- name: Constant_137_0	type: float	shape: Shape{384}
//	- name: Convolution_600_0	type: float	shape: Shape{32, 384, 17, 17}
//	- name: Constant_138_0	type: float	shape: Shape{384}
//	- name: Constant_139_0	type: float	shape: Shape{384}
// Output:
//	- name: BatchNormInference_604_0	type: float	shape: Shape{32, 384, 17, 17}
extern "C" __launch_bounds__(289) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_604(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 17 * 17;
    const int c_id = blockIdx.x % 384;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 17 * 17; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_604_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_604<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_527
// Description:	Reshape
// Input:
//	- name: Constant_47_0	type: float	shape: Shape{3, 3, 64, 96}
// Output:
//	- name: Reshape_527_0	type: float	shape: Shape{96, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_527(float* input0, float* output0)
{
    uint32_t input_strides0 = 6144;
    uint32_t input_strides1 = 96;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 576;
    size_t nx = 96;
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
extern void Reshape_float_float_cuda_Reshape_527_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_527<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	MaxPool_508
// Description:	MaxPool
// Input:
//	- name: Relu_507_0	type: float	shape: Shape{32, 192, 71, 71}
// Output:
//	- name: MaxPool_508_0	type: float	shape: Shape{32, 192, 35, 35}
void MaxPool_float_float_cuda_lib_MaxPool_508(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 71, 71));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
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
// Node name:	Reshape_486
// Description:	Reshape
// Input:
//	- name: Parameter_0_0	type: float	shape: Shape{32, 299, 299, 3}
// Output:
//	- name: Reshape_486_0	type: float	shape: Shape{32, 3, 299, 299}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_486(float* input0, float* output0)
{
    uint32_t input_strides0 = 268203;
    uint32_t input_strides1 = 3;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 268203;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 89401;
    size_t nx = 3;
    size_t ny = 89401;
    size_t nz = 32;
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
extern void Reshape_float_float_cuda_Reshape_486_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_486<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_511
// Description:	Reshape
// Input:
//	- name: Constant_32_0	type: float	shape: Shape{1, 1, 192, 48}
// Output:
//	- name: Reshape_511_0	type: float	shape: Shape{48, 192, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_511(float* input0, float* output0)
{
    uint32_t input_strides0 = 48;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 192;
    size_t nx = 48;
    size_t ny = 192;
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
extern void Reshape_float_float_cuda_Reshape_511_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_511<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_510
// Description:	Convolution
// Input:
//	- name: MaxPool_508_0	type: float	shape: Shape{32, 192, 35, 35}
//	- name: Reshape_509_0	type: float	shape: Shape{64, 192, 1, 1}
// Output:
//	- name: Convolution_510_0	type: float	shape: Shape{32, 64, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_510(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 192, 1, 1));
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
// Node name:	Convolution_505
// Description:	Convolution
// Input:
//	- name: Relu_503_0	type: float	shape: Shape{32, 80, 73, 73}
//	- name: Reshape_504_0	type: float	shape: Shape{192, 80, 3, 3}
// Output:
//	- name: Convolution_505_0	type: float	shape: Shape{32, 192, 71, 71}
void Convolution_float_float_float_cuda_lib_Convolution_505(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 80, 73, 73));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 71, 71));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 80, 3, 3));
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
// Node name:	Convolution_850
// Description:	Convolution
// Input:
//	- name: Concat_848_0	type: float	shape: Shape{32, 2048, 8, 8}
//	- name: Reshape_849_0	type: float	shape: Shape{320, 2048, 1, 1}
// Output:
//	- name: Convolution_850_0	type: float	shape: Shape{32, 320, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_850(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 320, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 320, 2048, 1, 1));
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
// Node name:	Concat_568
// Description:	Concat
// Input:
//	- name: Relu_551_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Relu_562_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Relu_567_0	type: float	shape: Shape{32, 96, 35, 35}
//	- name: Relu_559_0	type: float	shape: Shape{32, 64, 35, 35}
// Output:
//	- name: Concat_568_0	type: float	shape: Shape{32, 288, 35, 35}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_float_cuda_Concat_568(float* input0, float* input1, float* input2, float* input3, float* output0)
{
    uint32_t inputs_strides[] = {78400, 78400, 117600, 78400};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 11289600)
    {
        uint32_t block_id = tid / 352800;
        uint32_t block_idx = tid % 352800;
        uint32_t output_idx = block_id * 352800 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
        if(block_idx < inputs_strides[3])
        {
            output0[output_idx] = input3[block_id * inputs_strides[3] + block_idx];
            return;
        }
        block_idx -= inputs_strides[3];
    }

}
extern void Concat_float_float_float_float_float_cuda_Concat_568_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0) {
    Concat_float_float_float_float_float_cuda_Concat_568<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0);
}
// Node name:	Convolution_526
// Description:	Convolution
// Input:
//	- name: Relu_522_0	type: float	shape: Shape{32, 48, 35, 35}
//	- name: Reshape_525_0	type: float	shape: Shape{64, 48, 5, 5}
// Output:
//	- name: Convolution_526_0	type: float	shape: Shape{32, 64, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_526(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 48, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 48, 5, 5));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_811
// Description:	Reshape
// Input:
//	- name: Constant_391_0	type: float	shape: Shape{1, 1, 1280, 320}
// Output:
//	- name: Reshape_811_0	type: float	shape: Shape{320, 1280, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_811(float* input0, float* output0)
{
    uint32_t input_strides0 = 320;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1280;
    size_t nx = 320;
    size_t ny = 1280;
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
extern void Reshape_float_float_cuda_Reshape_811_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_811<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Constant_223
// Description:	Constant
// Input:
// Output:
//	- name: Constant_223_0	type: float	shape: Shape{192}
void Constant_float_cuda_Constant_223(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_223_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_223_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[768];
    bin_file.read(tmp_mem, 768);
    cudaMemcpyAsync(output0, tmp_mem, 768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	BatchNormInference_530
// Description:	BatchNormInference
// Input:
//	- name: Constant_38_0	type: float	shape: Shape{64}
//	- name: Constant_39_0	type: float	shape: Shape{64}
//	- name: Convolution_526_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Constant_40_0	type: float	shape: Shape{64}
//	- name: Constant_41_0	type: float	shape: Shape{64}
// Output:
//	- name: BatchNormInference_530_0	type: float	shape: Shape{32, 64, 35, 35}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 35 * 35;
    const int c_id = blockIdx.x % 64;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 35 * 35; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_500
// Description:	Reshape
// Input:
//	- name: Constant_17_0	type: float	shape: Shape{1, 1, 64, 80}
// Output:
//	- name: Reshape_500_0	type: float	shape: Shape{80, 64, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_500(float* input0, float* output0)
{
    uint32_t input_strides0 = 80;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 64;
    size_t nx = 80;
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
extern void Reshape_float_float_cuda_Reshape_500_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_500<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_534
// Description:	Reshape
// Input:
//	- name: Constant_52_0	type: float	shape: Shape{3, 3, 96, 96}
// Output:
//	- name: Reshape_534_0	type: float	shape: Shape{96, 96, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_534(float* input0, float* output0)
{
    uint32_t input_strides0 = 9216;
    uint32_t input_strides1 = 96;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 864;
    size_t nx = 96;
    size_t ny = 96;
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
extern void Reshape_float_float_cuda_Reshape_534_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_534<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	MaxPool_603
// Description:	MaxPool
// Input:
//	- name: Concat_598_0	type: float	shape: Shape{32, 288, 35, 35}
// Output:
//	- name: MaxPool_603_0	type: float	shape: Shape{32, 288, 17, 17}
void MaxPool_float_float_cuda_lib_MaxPool_603(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 17, 17));
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
// Node name:	AvgPool_515
// Description:	AvgPool
// Input:
//	- name: MaxPool_508_0	type: float	shape: Shape{32, 192, 35, 35}
// Output:
//	- name: AvgPool_515_0	type: float	shape: Shape{32, 192, 35, 35}
void AvgPool_float_float_cuda_lib_AvgPool_515(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Reshape_831
// Description:	Reshape
// Input:
//	- name: Constant_416_0	type: float	shape: Shape{3, 3, 448, 384}
// Output:
//	- name: Reshape_831_0	type: float	shape: Shape{384, 448, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_831(float* input0, float* output0)
{
    uint32_t input_strides0 = 172032;
    uint32_t input_strides1 = 384;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 4032;
    size_t nx = 384;
    size_t ny = 448;
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
extern void Reshape_float_float_cuda_Reshape_831_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_831<<<grids, blocks, mem, stream>>>(input0, output0);
}

extern "C" void cuda_init()
{
CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:452417856
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,356933632));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 356933632));
Reshape_486_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_487_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+34329984);
Convolution_488_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+34333440);
BatchNormInference_489_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+125268736);
Relu_490_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+125268736);
Reshape_491_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_492_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+36864);
BatchNormInference_493_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+88547328);
Relu_494_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+88547328);
Reshape_495_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_496_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+177057792);
BatchNormInference_497_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_498_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
MaxPool_499_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+177020928);
Reshape_500_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_501_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20480);
BatchNormInference_502_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+54589440);
Relu_503_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+54589440);
Reshape_504_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_505_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+109158400);
BatchNormInference_506_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+233046016);
Relu_507_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+233046016);
MaxPool_508_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_511_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30105600);
Convolution_512_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30142464);
BatchNormInference_517_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37668864);
Relu_522_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+37668864);
Reshape_525_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30105600);
Convolution_526_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45195264);
BatchNormInference_530_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30105600);
Relu_532_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30105600);
Reshape_509_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40140800);
Convolution_510_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40189952);
BatchNormInference_516_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+50225152);
Relu_521_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+50225152);
AvgPool_515_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
Reshape_519_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40140800);
Convolution_520_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40165376);
BatchNormInference_524_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45182976);
Relu_529_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45182976);
Reshape_513_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40140800);
Convolution_514_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
BatchNormInference_518_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_523_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_527_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Convolution_528_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10256384);
BatchNormInference_531_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
Relu_533_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
Reshape_534_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_535_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+331776);
BatchNormInference_536_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
Relu_537_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60260352);
Concat_538_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+75313152);
AvgPool_545_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_549_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40140800);
Convolution_550_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40206336);
BatchNormInference_554_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_559_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_543_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Convolution_544_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10100736);
BatchNormInference_548_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20135936);
Relu_553_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20135936);
Reshape_557_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Convolution_558_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30171136);
BatchNormInference_561_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Relu_563_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Reshape_564_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_565_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25419776);
BatchNormInference_566_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Relu_567_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Reshape_541_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_542_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25137152);
BatchNormInference_547_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32663552);
Relu_552_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32663552);
Reshape_555_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_556_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40189952);
BatchNormInference_560_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Relu_562_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Reshape_539_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35123200);
Convolution_540_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35188736);
BatchNormInference_546_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45223936);
Relu_551_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45223936);
Concat_568_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+55259136);
AvgPool_575_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_579_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45158400);
Convolution_580_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100417536);
BatchNormInference_584_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_589_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_573_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Convolution_574_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10108928);
BatchNormInference_578_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20144128);
Relu_583_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20144128);
Reshape_587_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Convolution_588_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30179328);
BatchNormInference_591_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Relu_593_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Reshape_594_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_595_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25419776);
BatchNormInference_596_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Relu_597_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10035200);
Reshape_571_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_572_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25143296);
BatchNormInference_577_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32669696);
Relu_582_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32669696);
Reshape_585_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Convolution_586_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40196096);
BatchNormInference_590_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Relu_592_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25088000);
Reshape_569_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35123200);
Convolution_570_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35196928);
BatchNormInference_576_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45232128);
Relu_581_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+45232128);
Concat_598_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+55267328);
MaxPool_603_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_601_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Convolution_602_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10727424);
BatchNormInference_605_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20762624);
Relu_607_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20762624);
Reshape_608_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Convolution_609_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+30797824);
BatchNormInference_610_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Relu_611_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Reshape_612_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25706496);
Convolution_613_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+26038272);
BatchNormInference_614_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Relu_615_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10653696);
Reshape_599_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_600_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+18186240);
BatchNormInference_604_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32391168);
Relu_606_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32391168);
Concat_616_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+46596096);
AvgPool_623_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_627_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28409856);
Convolution_628_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
BatchNormInference_632_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_637_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_621_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_622_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7495680);
BatchNormInference_626_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12230656);
Relu_631_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12230656);
Reshape_635_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_636_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16965632);
BatchNormInference_639_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_641_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_644_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11837440);
Convolution_645_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12296192);
BatchNormInference_647_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_649_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_650_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11837440);
Convolution_651_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12296192);
BatchNormInference_652_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_653_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_654_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11837440);
Convolution_655_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12525568);
BatchNormInference_656_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19628032);
Relu_657_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19628032);
Reshape_619_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_620_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7495680);
BatchNormInference_625_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12230656);
Relu_630_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12230656);
Reshape_633_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_634_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+26730496);
BatchNormInference_638_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_640_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_642_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11837440);
Convolution_643_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12525568);
BatchNormInference_646_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+26730496);
Relu_648_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+26730496);
Reshape_617_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_618_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7692288);
BatchNormInference_624_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33832960);
Relu_629_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+33832960);
Concat_658_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+40935424);
AvgPool_665_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_669_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28409856);
Convolution_670_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
BatchNormInference_674_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_679_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_663_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_664_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7593984);
BatchNormInference_668_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Relu_673_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Reshape_677_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_678_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19431424);
BatchNormInference_681_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_683_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_686_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_687_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13737984);
BatchNormInference_689_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_691_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_692_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_693_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13737984);
BatchNormInference_694_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_695_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_696_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_697_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13881344);
BatchNormInference_698_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20983808);
Relu_699_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20983808);
Reshape_661_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_662_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7593984);
BatchNormInference_667_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Relu_672_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Reshape_675_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_676_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
BatchNormInference_680_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_682_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_684_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_685_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13881344);
BatchNormInference_688_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
Relu_690_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
Reshape_659_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_660_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7692288);
BatchNormInference_666_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35188736);
Relu_671_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35188736);
Concat_700_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+42291200);
AvgPool_707_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_711_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28409856);
Convolution_712_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
BatchNormInference_716_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_721_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_705_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_706_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7593984);
BatchNormInference_710_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Relu_715_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Reshape_719_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_720_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19431424);
BatchNormInference_723_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_725_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_728_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_729_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13737984);
BatchNormInference_731_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_733_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_734_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_735_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13737984);
BatchNormInference_736_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_737_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_738_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_739_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13881344);
BatchNormInference_740_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20983808);
Relu_741_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+20983808);
Reshape_703_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_704_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7593984);
BatchNormInference_709_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Relu_714_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13512704);
Reshape_717_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_718_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
BatchNormInference_722_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_724_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_726_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13021184);
Convolution_727_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13881344);
BatchNormInference_730_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
Relu_732_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28086272);
Reshape_701_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_702_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7692288);
BatchNormInference_708_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35188736);
Relu_713_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35188736);
Concat_742_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+42291200);
AvgPool_749_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_753_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28409856);
Convolution_754_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
BatchNormInference_758_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_763_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_747_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_748_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7692288);
BatchNormInference_752_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14794752);
Relu_757_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14794752);
Reshape_761_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Convolution_762_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21897216);
BatchNormInference_765_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_767_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_770_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_771_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15237120);
BatchNormInference_773_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_775_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_776_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_777_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15237120);
BatchNormInference_778_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_779_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_780_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_781_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15237120);
BatchNormInference_782_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Relu_783_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7102464);
Reshape_745_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_746_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14794752);
BatchNormInference_751_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21897216);
Relu_756_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21897216);
Reshape_759_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Convolution_760_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
BatchNormInference_764_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Relu_766_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Reshape_768_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21307392);
Convolution_769_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+22339584);
BatchNormInference_772_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Relu_774_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14204928);
Reshape_743_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21307392);
Convolution_744_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21897216);
BatchNormInference_750_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
Relu_755_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28999680);
Concat_784_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+36102144);
MaxPool_789_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_787_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Convolution_788_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6881280);
BatchNormInference_791_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13983744);
Relu_793_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13983744);
Reshape_796_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Convolution_797_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+21086208);
BatchNormInference_799_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Relu_801_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Reshape_802_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13393920);
Convolution_803_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14426112);
BatchNormInference_804_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Relu_805_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Reshape_806_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13393920);
Convolution_807_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+14721024);
BatchNormInference_808_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Relu_809_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6291456);
Reshape_785_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_786_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+8454144);
BatchNormInference_790_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15556608);
Relu_792_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15556608);
Reshape_794_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_795_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10076160);
BatchNormInference_798_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12697600);
Relu_800_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12697600);
Concat_810_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15319040);
AvgPool_817_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_821_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+10485760);
Convolution_822_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11468800);
BatchNormInference_826_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_833_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_815_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Convolution_816_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3866624);
BatchNormInference_820_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7536640);
Relu_825_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7536640);
Reshape_831_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25804800);
Convolution_832_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
BatchNormInference_836_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Relu_839_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Reshape_842_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Convolution_843_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
BatchNormInference_845_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Relu_847_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Reshape_840_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_841_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9633792);
BatchNormInference_844_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Relu_846_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Reshape_813_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_814_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9830400);
BatchNormInference_819_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25804800);
Relu_824_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25804800);
Reshape_829_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_830_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9633792);
BatchNormInference_835_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28950528);
Relu_838_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+28950528);
Reshape_827_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_828_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9633792);
BatchNormInference_834_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25804800);
Relu_837_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25804800);
Reshape_811_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_812_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9502720);
BatchNormInference_818_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12124160);
Relu_823_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12124160);
Concat_848_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+32096256);
MaxPool_855_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_859_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16777216);
Convolution_860_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+18350080);
BatchNormInference_864_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_871_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_853_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Convolution_854_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+5242880);
BatchNormInference_858_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Relu_863_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Reshape_869_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+5242880);
Convolution_870_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+11436032);
BatchNormInference_874_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Relu_877_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Reshape_880_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Convolution_881_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6488064);
BatchNormInference_883_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9633792);
Relu_885_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9633792);
Reshape_878_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Convolution_879_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6488064);
BatchNormInference_882_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Relu_884_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1572864);
Reshape_851_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Convolution_852_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
BatchNormInference_857_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Relu_862_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Reshape_867_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_868_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
BatchNormInference_873_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15925248);
Relu_876_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+15925248);
Reshape_865_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+7864320);
Convolution_866_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
BatchNormInference_872_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Relu_875_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4718592);
Reshape_849_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
Convolution_850_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19070976);
BatchNormInference_856_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
Relu_861_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+12779520);
Concat_886_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+19070976);
AvgPool_887_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_888_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_889_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+262144);
Broadcast_890_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_891_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+262144);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,95484224));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 95484224));
Constant_484_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8200192);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8203648);
Constant_5_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8203776);
Constant_4_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8203904);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8204032);
Constant_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8204160);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8241024);
Constant_10_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8241152);
Constant_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8241280);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8241408);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8241536);
Constant_16_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8315264);
Constant_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8315520);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8315776);
Constant_13_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8316032);
Constant_17_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8316288);
Constant_21_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8336768);
Constant_20_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8337088);
Constant_19_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8337408);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8337728);
Constant_22_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8338048);
Constant_26_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8891008);
Constant_25_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8891776);
Constant_24_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8892544);
Constant_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8893312);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8894080);
Constant_36_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8930944);
Constant_35_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8931136);
Constant_34_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8931328);
Constant_33_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8931520);
Constant_37_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8931712);
Constant_41_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9238912);
Constant_40_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9239168);
Constant_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9239424);
Constant_38_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9239680);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9239936);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9289088);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9289344);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9289600);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9289856);
Constant_57_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9290112);
Constant_61_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9314688);
Constant_60_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9314816);
Constant_59_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9314944);
Constant_58_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9315072);
Constant_42_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9315200);
Constant_46_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9364352);
Constant_45_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9364608);
Constant_44_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9364864);
Constant_43_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9365120);
Constant_47_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9365376);
Constant_51_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9586560);
Constant_50_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9586944);
Constant_49_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9587328);
Constant_48_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9587712);
Constant_52_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9588096);
Constant_56_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9919872);
Constant_55_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9920256);
Constant_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9920640);
Constant_53_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9921024);
Constant_93_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9921408);
Constant_94_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9986944);
Constant_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9987200);
Constant_96_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9987456);
Constant_95_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9987712);
Constant_78_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9987968);
Constant_79_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10053504);
Constant_82_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10053760);
Constant_81_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10054016);
Constant_80_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10054272);
Constant_83_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10054528);
Constant_84_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10275712);
Constant_87_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10276096);
Constant_86_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10276480);
Constant_85_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10276864);
Constant_88_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10277248);
Constant_89_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10609024);
Constant_92_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10609408);
Constant_91_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10609792);
Constant_90_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10610176);
Constant_68_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10610560);
Constant_69_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10659712);
Constant_72_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10659904);
Constant_71_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10660096);
Constant_70_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10660288);
Constant_73_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10660480);
Constant_74_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10967680);
Constant_77_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10967936);
Constant_76_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10968192);
Constant_75_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10968448);
Constant_63_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10968704);
Constant_64_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11034240);
Constant_67_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11034496);
Constant_66_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11034752);
Constant_65_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11035008);
Constant_129_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11035264);
Constant_130_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11108992);
Constant_133_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11109248);
Constant_132_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11109504);
Constant_131_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11109760);
Constant_114_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11110016);
Constant_115_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11183744);
Constant_118_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11184000);
Constant_117_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11184256);
Constant_116_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11184512);
Constant_119_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11184768);
Constant_120_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11405952);
Constant_123_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11406336);
Constant_122_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11406720);
Constant_121_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11407104);
Constant_124_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11407488);
Constant_125_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11739264);
Constant_128_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11739648);
Constant_127_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11740032);
Constant_126_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11740416);
Constant_104_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11740800);
Constant_105_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11796096);
Constant_108_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11796288);
Constant_107_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11796480);
Constant_106_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11796672);
Constant_109_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+11796864);
Constant_110_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12104064);
Constant_113_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12104320);
Constant_112_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12104576);
Constant_111_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12104832);
Constant_99_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12105088);
Constant_100_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12178816);
Constant_103_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12179072);
Constant_102_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12179328);
Constant_101_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12179584);
Constant_140_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12179840);
Constant_144_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12253568);
Constant_143_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12253824);
Constant_142_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12254080);
Constant_141_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12254336);
Constant_145_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12254592);
Constant_149_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12475776);
Constant_148_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12476160);
Constant_147_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12476544);
Constant_146_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12476928);
Constant_150_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12477312);
Constant_154_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12809088);
Constant_153_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12809472);
Constant_152_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12809856);
Constant_151_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12810240);
Constant_135_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12810624);
Constant_139_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16791936);
Constant_138_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16793472);
Constant_137_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16795008);
Constant_136_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16796544);
Constant_201_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16798080);
Constant_205_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17387904);
Constant_204_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17388672);
Constant_203_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17389440);
Constant_202_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17390208);
Constant_176_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17390976);
Constant_180_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17784192);
Constant_179_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17784704);
Constant_178_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17785216);
Constant_177_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17785728);
Constant_181_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+17786240);
Constant_185_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18244992);
Constant_184_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18245504);
Constant_183_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18246016);
Constant_182_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18246528);
Constant_186_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18247040);
Constant_190_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18705792);
Constant_189_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18706304);
Constant_188_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18706816);
Constant_187_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18707328);
Constant_191_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18707840);
Constant_195_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19166592);
Constant_194_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19167104);
Constant_193_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19167616);
Constant_192_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19168128);
Constant_196_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19168640);
Constant_200_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19856768);
Constant_199_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19857536);
Constant_198_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19858304);
Constant_197_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19859072);
Constant_161_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+19859840);
Constant_165_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20253056);
Constant_164_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20253568);
Constant_163_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20254080);
Constant_162_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20254592);
Constant_166_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20255104);
Constant_170_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20713856);
Constant_169_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20714368);
Constant_168_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20714880);
Constant_167_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20715392);
Constant_171_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20715904);
Constant_175_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21404032);
Constant_174_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21404800);
Constant_173_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21405568);
Constant_172_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21406336);
Constant_156_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21407104);
Constant_160_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21996928);
Constant_159_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21997696);
Constant_158_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21998464);
Constant_157_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+21999232);
Constant_252_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22000000);
Constant_253_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22589824);
Constant_256_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22590592);
Constant_255_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22591360);
Constant_254_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22592128);
Constant_227_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+22592896);
Constant_228_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23084416);
Constant_231_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23085056);
Constant_230_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23085696);
Constant_229_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23086336);
Constant_232_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23086976);
Constant_233_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23803776);
Constant_236_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23804416);
Constant_235_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23805056);
Constant_234_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23805696);
Constant_237_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+23806336);
Constant_238_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24523136);
Constant_241_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24523776);
Constant_240_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24524416);
Constant_239_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24525056);
Constant_242_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+24525696);
Constant_243_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25242496);
Constant_246_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25243136);
Constant_245_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25243776);
Constant_244_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25244416);
Constant_247_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25245056);
Constant_248_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26105216);
Constant_251_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26105984);
Constant_250_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26106752);
Constant_249_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26107520);
Constant_212_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26108288);
Constant_213_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26599808);
Constant_216_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26600448);
Constant_215_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26601088);
Constant_214_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26601728);
Constant_217_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+26602368);
Constant_218_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+27319168);
Constant_221_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+27319808);
Constant_220_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+27320448);
Constant_219_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+27321088);
Constant_222_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+27321728);
Constant_223_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28181888);
Constant_226_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28182656);
Constant_225_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28183424);
Constant_224_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28184192);
Constant_207_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28184960);
Constant_208_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28774784);
Constant_211_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28775552);
Constant_210_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28776320);
Constant_209_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28777088);
Constant_303_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+28777856);
Constant_304_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29367680);
Constant_307_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29368448);
Constant_306_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29369216);
Constant_305_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29369984);
Constant_278_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29370752);
Constant_279_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29862272);
Constant_282_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29862912);
Constant_281_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29863552);
Constant_280_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29864192);
Constant_283_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29864832);
Constant_284_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30581632);
Constant_287_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30582272);
Constant_286_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30582912);
Constant_285_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30583552);
Constant_288_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+30584192);
Constant_289_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31300992);
Constant_292_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31301632);
Constant_291_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31302272);
Constant_290_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31302912);
Constant_293_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+31303552);
Constant_294_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32020352);
Constant_297_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32020992);
Constant_296_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32021632);
Constant_295_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32022272);
Constant_298_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32022912);
Constant_299_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32883072);
Constant_302_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32883840);
Constant_301_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32884608);
Constant_300_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32885376);
Constant_263_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+32886144);
Constant_264_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33377664);
Constant_267_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33378304);
Constant_266_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33378944);
Constant_265_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33379584);
Constant_268_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33380224);
Constant_269_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34097024);
Constant_272_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34097664);
Constant_271_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34098304);
Constant_270_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34098944);
Constant_273_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34099584);
Constant_274_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34959744);
Constant_277_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34960512);
Constant_276_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34961280);
Constant_275_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34962048);
Constant_258_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+34962816);
Constant_259_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35552640);
Constant_262_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35553408);
Constant_261_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35554176);
Constant_260_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35554944);
Constant_354_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35555712);
Constant_355_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36145536);
Constant_358_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36146304);
Constant_357_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36147072);
Constant_356_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36147840);
Constant_329_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36148608);
Constant_330_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36738432);
Constant_333_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36739200);
Constant_332_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36739968);
Constant_331_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36740736);
Constant_334_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+36741504);
Constant_335_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37773696);
Constant_338_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37774464);
Constant_337_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37775232);
Constant_336_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37776000);
Constant_339_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37776768);
Constant_340_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38808960);
Constant_343_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38809728);
Constant_342_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38810496);
Constant_341_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38811264);
Constant_344_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+38812032);
Constant_345_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39844224);
Constant_348_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39844992);
Constant_347_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39845760);
Constant_346_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39846528);
Constant_349_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39847296);
Constant_350_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40879488);
Constant_353_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40880256);
Constant_352_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40881024);
Constant_351_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40881792);
Constant_314_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+40882560);
Constant_315_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41472384);
Constant_318_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41473152);
Constant_317_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41473920);
Constant_316_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41474688);
Constant_319_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41475456);
Constant_320_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+42507648);
Constant_323_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+42508416);
Constant_322_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+42509184);
Constant_321_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+42509952);
Constant_324_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+42510720);
Constant_325_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+43542912);
Constant_328_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+43543680);
Constant_327_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+43544448);
Constant_326_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+43545216);
Constant_309_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+43545984);
Constant_310_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44135808);
Constant_313_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44136576);
Constant_312_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44137344);
Constant_311_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44138112);
Constant_370_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44138880);
Constant_374_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44728704);
Constant_373_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44729472);
Constant_372_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44730240);
Constant_371_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44731008);
Constant_375_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+44731776);
Constant_379_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45763968);
Constant_378_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45764736);
Constant_377_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45765504);
Constant_376_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45766272);
Constant_380_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+45767040);
Constant_384_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+46799232);
Constant_383_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+46800000);
Constant_382_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+46800768);
Constant_381_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+46801536);
Constant_385_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+46802304);
Constant_389_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48129408);
Constant_388_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48130176);
Constant_387_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48130944);
Constant_386_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48131712);
Constant_360_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48132480);
Constant_364_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48722304);
Constant_363_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48723072);
Constant_362_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48723840);
Constant_361_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48724608);
Constant_365_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+48725376);
Constant_369_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50937216);
Constant_368_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50938496);
Constant_367_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50939776);
Constant_366_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50941056);
Constant_431_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+50942336);
Constant_435_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+51925376);
Constant_434_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+51926144);
Constant_433_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+51926912);
Constant_432_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+51927680);
Constant_411_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+51928448);
Constant_415_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+54222208);
Constant_414_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+54224000);
Constant_413_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+54225792);
Constant_412_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+54227584);
Constant_416_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+54229376);
Constant_420_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+60422528);
Constant_419_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+60424064);
Constant_418_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+60425600);
Constant_417_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+60427136);
Constant_426_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+60428672);
Constant_430_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62198144);
Constant_429_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62199680);
Constant_428_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62201216);
Constant_427_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62202752);
Constant_421_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+62204288);
Constant_425_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+63973760);
Constant_424_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+63975296);
Constant_423_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+63976832);
Constant_422_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+63978368);
Constant_396_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+63979904);
Constant_400_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+65945984);
Constant_399_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+65947520);
Constant_398_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+65949056);
Constant_397_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+65950592);
Constant_406_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+65952128);
Constant_410_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+67721600);
Constant_409_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+67723136);
Constant_408_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+67724672);
Constant_407_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+67726208);
Constant_401_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+67727744);
Constant_405_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69497216);
Constant_404_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69498752);
Constant_403_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69500288);
Constant_402_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69501824);
Constant_391_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+69503360);
Constant_395_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+71141760);
Constant_394_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+71143040);
Constant_393_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+71144320);
Constant_392_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+71145600);
Constant_477_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+71146880);
Constant_478_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72719744);
Constant_481_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72720512);
Constant_480_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72721280);
Constant_479_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72722048);
Constant_457_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+72722816);
Constant_458_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76392832);
Constant_461_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76394624);
Constant_460_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76396416);
Constant_459_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76398208);
Constant_462_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+76400000);
Constant_463_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+82593152);
Constant_466_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+82594688);
Constant_465_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+82596224);
Constant_464_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+82597760);
Constant_472_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+82599296);
Constant_473_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+84368768);
Constant_476_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+84370304);
Constant_475_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+84371840);
Constant_474_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+84373376);
Constant_467_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+84374912);
Constant_468_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+86144384);
Constant_471_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+86145920);
Constant_470_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+86147456);
Constant_469_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+86148992);
Constant_442_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+86150528);
Constant_443_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+89296256);
Constant_446_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+89297792);
Constant_445_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+89299328);
Constant_444_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+89300864);
Constant_452_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+89302400);
Constant_453_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+91071872);
Constant_456_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+91073408);
Constant_455_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+91074944);
Constant_454_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+91076480);
Constant_447_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+91078016);
Constant_448_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92847488);
Constant_451_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92849024);
Constant_450_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92850560);
Constant_449_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92852096);
Constant_437_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+92853632);
Constant_438_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95475072);
Constant_441_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95476352);
Constant_440_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95477632);
Constant_439_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95478912);
Constant_485_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+95480192);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
 // name=cg/affine0/weights
Constant_float_cuda_Constant_484(0, Constant_484_0);
 // name=cg/conv0/conv2d/kernel
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=cg/conv0/batchnorm0/moving_variance
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=cg/conv0/batchnorm0/moving_mean
Constant_float_cuda_Constant_5(0, Constant_5_0);
 // name=cg/conv0/batchnorm0/beta
Constant_float_cuda_Constant_4(0, Constant_4_0);
 // name=cg/conv0/batchnorm0/Const
Constant_float_cuda_Constant_3(0, Constant_3_0);
 // name=cg/conv1/conv2d/kernel
Constant_float_cuda_Constant_7(0, Constant_7_0);
 // name=cg/conv1/batchnorm1/moving_variance
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=cg/conv1/batchnorm1/moving_mean
Constant_float_cuda_Constant_10(0, Constant_10_0);
 // name=cg/conv1/batchnorm1/beta
Constant_float_cuda_Constant_9(0, Constant_9_0);
 // name=cg/conv1/batchnorm1/Const
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=cg/conv2/conv2d/kernel
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=cg/conv2/batchnorm2/moving_variance
Constant_float_cuda_Constant_16(0, Constant_16_0);
 // name=cg/conv2/batchnorm2/moving_mean
Constant_float_cuda_Constant_15(0, Constant_15_0);
 // name=cg/conv2/batchnorm2/beta
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=cg/conv2/batchnorm2/Const
Constant_float_cuda_Constant_13(0, Constant_13_0);
 // name=cg/conv3/conv2d/kernel
Constant_float_cuda_Constant_17(0, Constant_17_0);
 // name=cg/conv3/batchnorm3/moving_variance
Constant_float_cuda_Constant_21(0, Constant_21_0);
 // name=cg/conv3/batchnorm3/moving_mean
Constant_float_cuda_Constant_20(0, Constant_20_0);
 // name=cg/conv3/batchnorm3/beta
Constant_float_cuda_Constant_19(0, Constant_19_0);
 // name=cg/conv3/batchnorm3/Const
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=cg/conv4/conv2d/kernel
Constant_float_cuda_Constant_22(0, Constant_22_0);
 // name=cg/conv4/batchnorm4/moving_variance
Constant_float_cuda_Constant_26(0, Constant_26_0);
 // name=cg/conv4/batchnorm4/moving_mean
Constant_float_cuda_Constant_25(0, Constant_25_0);
 // name=cg/conv4/batchnorm4/beta
Constant_float_cuda_Constant_24(0, Constant_24_0);
 // name=cg/conv4/batchnorm4/Const
Constant_float_cuda_Constant_23(0, Constant_23_0);
 // name=cg/incept_v3_a0/conv6/conv2d/kernel
Constant_float_cuda_Constant_32(0, Constant_32_0);
 // name=cg/incept_v3_a0/conv6/batchnorm6/moving_variance
Constant_float_cuda_Constant_36(0, Constant_36_0);
 // name=cg/incept_v3_a0/conv6/batchnorm6/moving_mean
Constant_float_cuda_Constant_35(0, Constant_35_0);
 // name=cg/incept_v3_a0/conv6/batchnorm6/beta
Constant_float_cuda_Constant_34(0, Constant_34_0);
 // name=cg/incept_v3_a0/conv6/batchnorm6/Const
Constant_float_cuda_Constant_33(0, Constant_33_0);
 // name=cg/incept_v3_a0/conv7/conv2d/kernel
Constant_float_cuda_Constant_37(0, Constant_37_0);
 // name=cg/incept_v3_a0/conv7/batchnorm7/moving_variance
Constant_float_cuda_Constant_41(0, Constant_41_0);
 // name=cg/incept_v3_a0/conv7/batchnorm7/moving_mean
Constant_float_cuda_Constant_40(0, Constant_40_0);
 // name=cg/incept_v3_a0/conv7/batchnorm7/beta
Constant_float_cuda_Constant_39(0, Constant_39_0);
 // name=cg/incept_v3_a0/conv7/batchnorm7/Const
Constant_float_cuda_Constant_38(0, Constant_38_0);
 // name=cg/incept_v3_a0/conv5/conv2d/kernel
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=cg/incept_v3_a0/conv5/batchnorm5/moving_variance
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=cg/incept_v3_a0/conv5/batchnorm5/moving_mean
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=cg/incept_v3_a0/conv5/batchnorm5/beta
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=cg/incept_v3_a0/conv5/batchnorm5/Const
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=cg/incept_v3_a0/conv11/conv2d/kernel
Constant_float_cuda_Constant_57(0, Constant_57_0);
 // name=cg/incept_v3_a0/conv11/batchnorm11/moving_variance
Constant_float_cuda_Constant_61(0, Constant_61_0);
 // name=cg/incept_v3_a0/conv11/batchnorm11/moving_mean
Constant_float_cuda_Constant_60(0, Constant_60_0);
 // name=cg/incept_v3_a0/conv11/batchnorm11/beta
Constant_float_cuda_Constant_59(0, Constant_59_0);
 // name=cg/incept_v3_a0/conv11/batchnorm11/Const
Constant_float_cuda_Constant_58(0, Constant_58_0);
 // name=cg/incept_v3_a0/conv8/conv2d/kernel
Constant_float_cuda_Constant_42(0, Constant_42_0);
 // name=cg/incept_v3_a0/conv8/batchnorm8/moving_variance
Constant_float_cuda_Constant_46(0, Constant_46_0);
 // name=cg/incept_v3_a0/conv8/batchnorm8/moving_mean
Constant_float_cuda_Constant_45(0, Constant_45_0);
 // name=cg/incept_v3_a0/conv8/batchnorm8/beta
Constant_float_cuda_Constant_44(0, Constant_44_0);
 // name=cg/incept_v3_a0/conv8/batchnorm8/Const
Constant_float_cuda_Constant_43(0, Constant_43_0);
 // name=cg/incept_v3_a0/conv9/conv2d/kernel
Constant_float_cuda_Constant_47(0, Constant_47_0);
 // name=cg/incept_v3_a0/conv9/batchnorm9/moving_variance
Constant_float_cuda_Constant_51(0, Constant_51_0);
 // name=cg/incept_v3_a0/conv9/batchnorm9/moving_mean
Constant_float_cuda_Constant_50(0, Constant_50_0);
 // name=cg/incept_v3_a0/conv9/batchnorm9/beta
Constant_float_cuda_Constant_49(0, Constant_49_0);
 // name=cg/incept_v3_a0/conv9/batchnorm9/Const
Constant_float_cuda_Constant_48(0, Constant_48_0);
 // name=cg/incept_v3_a0/conv10/conv2d/kernel
Constant_float_cuda_Constant_52(0, Constant_52_0);
 // name=cg/incept_v3_a0/conv10/batchnorm10/moving_variance
Constant_float_cuda_Constant_56(0, Constant_56_0);
 // name=cg/incept_v3_a0/conv10/batchnorm10/moving_mean
Constant_float_cuda_Constant_55(0, Constant_55_0);
 // name=cg/incept_v3_a0/conv10/batchnorm10/beta
Constant_float_cuda_Constant_54(0, Constant_54_0);
 // name=cg/incept_v3_a0/conv10/batchnorm10/Const
Constant_float_cuda_Constant_53(0, Constant_53_0);
 // name=cg/incept_v3_a0/conv18/conv2d/kernel
Constant_float_cuda_Constant_93(0, Constant_93_0);
 // name=cg/incept_v3_a0_1/conv18/batchnorm18/Const
Constant_float_cuda_Constant_94(0, Constant_94_0);
 // name=cg/incept_v3_a0/conv18/batchnorm18/moving_variance
Constant_float_cuda_Constant_97(0, Constant_97_0);
 // name=cg/incept_v3_a0/conv18/batchnorm18/moving_mean
Constant_float_cuda_Constant_96(0, Constant_96_0);
 // name=cg/incept_v3_a0/conv18/batchnorm18/beta
Constant_float_cuda_Constant_95(0, Constant_95_0);
 // name=cg/incept_v3_a0/conv15/conv2d/kernel
Constant_float_cuda_Constant_78(0, Constant_78_0);
 // name=cg/incept_v3_a0_1/conv15/batchnorm15/Const
Constant_float_cuda_Constant_79(0, Constant_79_0);
 // name=cg/incept_v3_a0/conv15/batchnorm15/moving_variance
Constant_float_cuda_Constant_82(0, Constant_82_0);
 // name=cg/incept_v3_a0/conv15/batchnorm15/moving_mean
Constant_float_cuda_Constant_81(0, Constant_81_0);
 // name=cg/incept_v3_a0/conv15/batchnorm15/beta
Constant_float_cuda_Constant_80(0, Constant_80_0);
 // name=cg/incept_v3_a0/conv16/conv2d/kernel
Constant_float_cuda_Constant_83(0, Constant_83_0);
 // name=cg/incept_v3_a0_1/conv16/batchnorm16/Const
Constant_float_cuda_Constant_84(0, Constant_84_0);
 // name=cg/incept_v3_a0/conv16/batchnorm16/moving_variance
Constant_float_cuda_Constant_87(0, Constant_87_0);
 // name=cg/incept_v3_a0/conv16/batchnorm16/moving_mean
Constant_float_cuda_Constant_86(0, Constant_86_0);
 // name=cg/incept_v3_a0/conv16/batchnorm16/beta
Constant_float_cuda_Constant_85(0, Constant_85_0);
 // name=cg/incept_v3_a0/conv17/conv2d/kernel
Constant_float_cuda_Constant_88(0, Constant_88_0);
 // name=cg/incept_v3_a0_1/conv17/batchnorm17/Const
Constant_float_cuda_Constant_89(0, Constant_89_0);
 // name=cg/incept_v3_a0/conv17/batchnorm17/moving_variance
Constant_float_cuda_Constant_92(0, Constant_92_0);
 // name=cg/incept_v3_a0/conv17/batchnorm17/moving_mean
Constant_float_cuda_Constant_91(0, Constant_91_0);
 // name=cg/incept_v3_a0/conv17/batchnorm17/beta
Constant_float_cuda_Constant_90(0, Constant_90_0);
 // name=cg/incept_v3_a0/conv13/conv2d/kernel
Constant_float_cuda_Constant_68(0, Constant_68_0);
 // name=cg/incept_v3_a0_1/conv13/batchnorm13/Const
Constant_float_cuda_Constant_69(0, Constant_69_0);
 // name=cg/incept_v3_a0/conv13/batchnorm13/moving_variance
Constant_float_cuda_Constant_72(0, Constant_72_0);
 // name=cg/incept_v3_a0/conv13/batchnorm13/moving_mean
Constant_float_cuda_Constant_71(0, Constant_71_0);
 // name=cg/incept_v3_a0/conv13/batchnorm13/beta
Constant_float_cuda_Constant_70(0, Constant_70_0);
 // name=cg/incept_v3_a0/conv14/conv2d/kernel
Constant_float_cuda_Constant_73(0, Constant_73_0);
 // name=cg/incept_v3_a0_1/conv14/batchnorm14/Const
Constant_float_cuda_Constant_74(0, Constant_74_0);
 // name=cg/incept_v3_a0/conv14/batchnorm14/moving_variance
Constant_float_cuda_Constant_77(0, Constant_77_0);
 // name=cg/incept_v3_a0/conv14/batchnorm14/moving_mean
Constant_float_cuda_Constant_76(0, Constant_76_0);
 // name=cg/incept_v3_a0/conv14/batchnorm14/beta
Constant_float_cuda_Constant_75(0, Constant_75_0);
 // name=cg/incept_v3_a0/conv12/conv2d/kernel
Constant_float_cuda_Constant_63(0, Constant_63_0);
 // name=cg/incept_v3_a0_1/conv12/batchnorm12/Const
Constant_float_cuda_Constant_64(0, Constant_64_0);
 // name=cg/incept_v3_a0/conv12/batchnorm12/moving_variance
Constant_float_cuda_Constant_67(0, Constant_67_0);
 // name=cg/incept_v3_a0/conv12/batchnorm12/moving_mean
Constant_float_cuda_Constant_66(0, Constant_66_0);
 // name=cg/incept_v3_a0/conv12/batchnorm12/beta
Constant_float_cuda_Constant_65(0, Constant_65_0);
 // name=cg/incept_v3_a0/conv25/conv2d/kernel
Constant_float_cuda_Constant_129(0, Constant_129_0);
 // name=cg/incept_v3_a0_2/conv25/batchnorm25/Const
Constant_float_cuda_Constant_130(0, Constant_130_0);
 // name=cg/incept_v3_a0/conv25/batchnorm25/moving_variance
Constant_float_cuda_Constant_133(0, Constant_133_0);
 // name=cg/incept_v3_a0/conv25/batchnorm25/moving_mean
Constant_float_cuda_Constant_132(0, Constant_132_0);
 // name=cg/incept_v3_a0/conv25/batchnorm25/beta
Constant_float_cuda_Constant_131(0, Constant_131_0);
 // name=cg/incept_v3_a0/conv22/conv2d/kernel
Constant_float_cuda_Constant_114(0, Constant_114_0);
 // name=cg/incept_v3_a0_2/conv22/batchnorm22/Const
Constant_float_cuda_Constant_115(0, Constant_115_0);
 // name=cg/incept_v3_a0/conv22/batchnorm22/moving_variance
Constant_float_cuda_Constant_118(0, Constant_118_0);
 // name=cg/incept_v3_a0/conv22/batchnorm22/moving_mean
Constant_float_cuda_Constant_117(0, Constant_117_0);
 // name=cg/incept_v3_a0/conv22/batchnorm22/beta
Constant_float_cuda_Constant_116(0, Constant_116_0);
 // name=cg/incept_v3_a0/conv23/conv2d/kernel
Constant_float_cuda_Constant_119(0, Constant_119_0);
 // name=cg/incept_v3_a0_2/conv23/batchnorm23/Const
Constant_float_cuda_Constant_120(0, Constant_120_0);
 // name=cg/incept_v3_a0/conv23/batchnorm23/moving_variance
Constant_float_cuda_Constant_123(0, Constant_123_0);
 // name=cg/incept_v3_a0/conv23/batchnorm23/moving_mean
Constant_float_cuda_Constant_122(0, Constant_122_0);
 // name=cg/incept_v3_a0/conv23/batchnorm23/beta
Constant_float_cuda_Constant_121(0, Constant_121_0);
 // name=cg/incept_v3_a0/conv24/conv2d/kernel
Constant_float_cuda_Constant_124(0, Constant_124_0);
 // name=cg/incept_v3_a0_2/conv24/batchnorm24/Const
Constant_float_cuda_Constant_125(0, Constant_125_0);
 // name=cg/incept_v3_a0/conv24/batchnorm24/moving_variance
Constant_float_cuda_Constant_128(0, Constant_128_0);
 // name=cg/incept_v3_a0/conv24/batchnorm24/moving_mean
Constant_float_cuda_Constant_127(0, Constant_127_0);
 // name=cg/incept_v3_a0/conv24/batchnorm24/beta
Constant_float_cuda_Constant_126(0, Constant_126_0);
 // name=cg/incept_v3_a0/conv20/conv2d/kernel
Constant_float_cuda_Constant_104(0, Constant_104_0);
 // name=cg/incept_v3_a0_2/conv20/batchnorm20/Const
Constant_float_cuda_Constant_105(0, Constant_105_0);
 // name=cg/incept_v3_a0/conv20/batchnorm20/moving_variance
Constant_float_cuda_Constant_108(0, Constant_108_0);
 // name=cg/incept_v3_a0/conv20/batchnorm20/moving_mean
Constant_float_cuda_Constant_107(0, Constant_107_0);
 // name=cg/incept_v3_a0/conv20/batchnorm20/beta
Constant_float_cuda_Constant_106(0, Constant_106_0);
 // name=cg/incept_v3_a0/conv21/conv2d/kernel
Constant_float_cuda_Constant_109(0, Constant_109_0);
 // name=cg/incept_v3_a0_2/conv21/batchnorm21/Const
Constant_float_cuda_Constant_110(0, Constant_110_0);
 // name=cg/incept_v3_a0/conv21/batchnorm21/moving_variance
Constant_float_cuda_Constant_113(0, Constant_113_0);
 // name=cg/incept_v3_a0/conv21/batchnorm21/moving_mean
Constant_float_cuda_Constant_112(0, Constant_112_0);
 // name=cg/incept_v3_a0/conv21/batchnorm21/beta
Constant_float_cuda_Constant_111(0, Constant_111_0);
 // name=cg/incept_v3_a0/conv19/conv2d/kernel
Constant_float_cuda_Constant_99(0, Constant_99_0);
 // name=cg/incept_v3_a0_2/conv19/batchnorm19/Const
Constant_float_cuda_Constant_100(0, Constant_100_0);
 // name=cg/incept_v3_a0/conv19/batchnorm19/moving_variance
Constant_float_cuda_Constant_103(0, Constant_103_0);
 // name=cg/incept_v3_a0/conv19/batchnorm19/moving_mean
Constant_float_cuda_Constant_102(0, Constant_102_0);
 // name=cg/incept_v3_a0/conv19/batchnorm19/beta
Constant_float_cuda_Constant_101(0, Constant_101_0);
 // name=cg/incept_v3_b0/conv27/conv2d/kernel
Constant_float_cuda_Constant_140(0, Constant_140_0);
 // name=cg/incept_v3_b0/conv27/batchnorm27/moving_variance
Constant_float_cuda_Constant_144(0, Constant_144_0);
 // name=cg/incept_v3_b0/conv27/batchnorm27/moving_mean
Constant_float_cuda_Constant_143(0, Constant_143_0);
 // name=cg/incept_v3_b0/conv27/batchnorm27/beta
Constant_float_cuda_Constant_142(0, Constant_142_0);
 // name=cg/incept_v3_b0/conv27/batchnorm27/Const
Constant_float_cuda_Constant_141(0, Constant_141_0);
 // name=cg/incept_v3_b0/conv28/conv2d/kernel
Constant_float_cuda_Constant_145(0, Constant_145_0);
 // name=cg/incept_v3_b0/conv28/batchnorm28/moving_variance
Constant_float_cuda_Constant_149(0, Constant_149_0);
 // name=cg/incept_v3_b0/conv28/batchnorm28/moving_mean
Constant_float_cuda_Constant_148(0, Constant_148_0);
 // name=cg/incept_v3_b0/conv28/batchnorm28/beta
Constant_float_cuda_Constant_147(0, Constant_147_0);
 // name=cg/incept_v3_b0/conv28/batchnorm28/Const
Constant_float_cuda_Constant_146(0, Constant_146_0);
 // name=cg/incept_v3_b0/conv29/conv2d/kernel
Constant_float_cuda_Constant_150(0, Constant_150_0);
 // name=cg/incept_v3_b0/conv29/batchnorm29/moving_variance
Constant_float_cuda_Constant_154(0, Constant_154_0);
 // name=cg/incept_v3_b0/conv29/batchnorm29/moving_mean
Constant_float_cuda_Constant_153(0, Constant_153_0);
 // name=cg/incept_v3_b0/conv29/batchnorm29/beta
Constant_float_cuda_Constant_152(0, Constant_152_0);
 // name=cg/incept_v3_b0/conv29/batchnorm29/Const
Constant_float_cuda_Constant_151(0, Constant_151_0);
 // name=cg/incept_v3_b0/conv26/conv2d/kernel
Constant_float_cuda_Constant_135(0, Constant_135_0);
 // name=cg/incept_v3_b0/conv26/batchnorm26/moving_variance
Constant_float_cuda_Constant_139(0, Constant_139_0);
 // name=cg/incept_v3_b0/conv26/batchnorm26/moving_mean
Constant_float_cuda_Constant_138(0, Constant_138_0);
 // name=cg/incept_v3_b0/conv26/batchnorm26/beta
Constant_float_cuda_Constant_137(0, Constant_137_0);
 // name=cg/incept_v3_b0/conv26/batchnorm26/Const
Constant_float_cuda_Constant_136(0, Constant_136_0);
 // name=cg/incept_v3_c0/conv39/conv2d/kernel
Constant_float_cuda_Constant_201(0, Constant_201_0);
 // name=cg/incept_v3_c0/conv39/batchnorm39/moving_variance
Constant_float_cuda_Constant_205(0, Constant_205_0);
 // name=cg/incept_v3_c0/conv39/batchnorm39/moving_mean
Constant_float_cuda_Constant_204(0, Constant_204_0);
 // name=cg/incept_v3_c0/conv39/batchnorm39/beta
Constant_float_cuda_Constant_203(0, Constant_203_0);
 // name=cg/incept_v3_c0/conv39/batchnorm39/Const
Constant_float_cuda_Constant_202(0, Constant_202_0);
 // name=cg/incept_v3_c0/conv34/conv2d/kernel
Constant_float_cuda_Constant_176(0, Constant_176_0);
 // name=cg/incept_v3_c0/conv34/batchnorm34/moving_variance
Constant_float_cuda_Constant_180(0, Constant_180_0);
 // name=cg/incept_v3_c0/conv34/batchnorm34/moving_mean
Constant_float_cuda_Constant_179(0, Constant_179_0);
 // name=cg/incept_v3_c0/conv34/batchnorm34/beta
Constant_float_cuda_Constant_178(0, Constant_178_0);
 // name=cg/incept_v3_c0/conv34/batchnorm34/Const
Constant_float_cuda_Constant_177(0, Constant_177_0);
 // name=cg/incept_v3_c0/conv35/conv2d/kernel
Constant_float_cuda_Constant_181(0, Constant_181_0);
 // name=cg/incept_v3_c0/conv35/batchnorm35/moving_variance
Constant_float_cuda_Constant_185(0, Constant_185_0);
 // name=cg/incept_v3_c0/conv35/batchnorm35/moving_mean
Constant_float_cuda_Constant_184(0, Constant_184_0);
 // name=cg/incept_v3_c0/conv35/batchnorm35/beta
Constant_float_cuda_Constant_183(0, Constant_183_0);
 // name=cg/incept_v3_c0/conv35/batchnorm35/Const
Constant_float_cuda_Constant_182(0, Constant_182_0);
 // name=cg/incept_v3_c0/conv36/conv2d/kernel
Constant_float_cuda_Constant_186(0, Constant_186_0);
 // name=cg/incept_v3_c0/conv36/batchnorm36/moving_variance
Constant_float_cuda_Constant_190(0, Constant_190_0);
 // name=cg/incept_v3_c0/conv36/batchnorm36/moving_mean
Constant_float_cuda_Constant_189(0, Constant_189_0);
 // name=cg/incept_v3_c0/conv36/batchnorm36/beta
Constant_float_cuda_Constant_188(0, Constant_188_0);
 // name=cg/incept_v3_c0/conv36/batchnorm36/Const
Constant_float_cuda_Constant_187(0, Constant_187_0);
 // name=cg/incept_v3_c0/conv37/conv2d/kernel
Constant_float_cuda_Constant_191(0, Constant_191_0);
 // name=cg/incept_v3_c0/conv37/batchnorm37/moving_variance
Constant_float_cuda_Constant_195(0, Constant_195_0);
 // name=cg/incept_v3_c0/conv37/batchnorm37/moving_mean
Constant_float_cuda_Constant_194(0, Constant_194_0);
 // name=cg/incept_v3_c0/conv37/batchnorm37/beta
Constant_float_cuda_Constant_193(0, Constant_193_0);
 // name=cg/incept_v3_c0/conv37/batchnorm37/Const
Constant_float_cuda_Constant_192(0, Constant_192_0);
 // name=cg/incept_v3_c0/conv38/conv2d/kernel
Constant_float_cuda_Constant_196(0, Constant_196_0);
 // name=cg/incept_v3_c0/conv38/batchnorm38/moving_variance
Constant_float_cuda_Constant_200(0, Constant_200_0);
 // name=cg/incept_v3_c0/conv38/batchnorm38/moving_mean
Constant_float_cuda_Constant_199(0, Constant_199_0);
 // name=cg/incept_v3_c0/conv38/batchnorm38/beta
Constant_float_cuda_Constant_198(0, Constant_198_0);
 // name=cg/incept_v3_c0/conv38/batchnorm38/Const
Constant_float_cuda_Constant_197(0, Constant_197_0);
 // name=cg/incept_v3_c0/conv31/conv2d/kernel
Constant_float_cuda_Constant_161(0, Constant_161_0);
 // name=cg/incept_v3_c0/conv31/batchnorm31/moving_variance
Constant_float_cuda_Constant_165(0, Constant_165_0);
 // name=cg/incept_v3_c0/conv31/batchnorm31/moving_mean
Constant_float_cuda_Constant_164(0, Constant_164_0);
 // name=cg/incept_v3_c0/conv31/batchnorm31/beta
Constant_float_cuda_Constant_163(0, Constant_163_0);
 // name=cg/incept_v3_c0/conv31/batchnorm31/Const
Constant_float_cuda_Constant_162(0, Constant_162_0);
 // name=cg/incept_v3_c0/conv32/conv2d/kernel
Constant_float_cuda_Constant_166(0, Constant_166_0);
 // name=cg/incept_v3_c0/conv32/batchnorm32/moving_variance
Constant_float_cuda_Constant_170(0, Constant_170_0);
 // name=cg/incept_v3_c0/conv32/batchnorm32/moving_mean
Constant_float_cuda_Constant_169(0, Constant_169_0);
 // name=cg/incept_v3_c0/conv32/batchnorm32/beta
Constant_float_cuda_Constant_168(0, Constant_168_0);
 // name=cg/incept_v3_c0/conv32/batchnorm32/Const
Constant_float_cuda_Constant_167(0, Constant_167_0);
 // name=cg/incept_v3_c0/conv33/conv2d/kernel
Constant_float_cuda_Constant_171(0, Constant_171_0);
 // name=cg/incept_v3_c0/conv33/batchnorm33/moving_variance
Constant_float_cuda_Constant_175(0, Constant_175_0);
 // name=cg/incept_v3_c0/conv33/batchnorm33/moving_mean
Constant_float_cuda_Constant_174(0, Constant_174_0);
 // name=cg/incept_v3_c0/conv33/batchnorm33/beta
Constant_float_cuda_Constant_173(0, Constant_173_0);
 // name=cg/incept_v3_c0/conv33/batchnorm33/Const
Constant_float_cuda_Constant_172(0, Constant_172_0);
 // name=cg/incept_v3_c0/conv30/conv2d/kernel
Constant_float_cuda_Constant_156(0, Constant_156_0);
 // name=cg/incept_v3_c0/conv30/batchnorm30/moving_variance
Constant_float_cuda_Constant_160(0, Constant_160_0);
 // name=cg/incept_v3_c0/conv30/batchnorm30/moving_mean
Constant_float_cuda_Constant_159(0, Constant_159_0);
 // name=cg/incept_v3_c0/conv30/batchnorm30/beta
Constant_float_cuda_Constant_158(0, Constant_158_0);
 // name=cg/incept_v3_c0/conv30/batchnorm30/Const
Constant_float_cuda_Constant_157(0, Constant_157_0);
 // name=cg/incept_v3_c0/conv49/conv2d/kernel
Constant_float_cuda_Constant_252(0, Constant_252_0);
 // name=cg/incept_v3_c0_1/conv49/batchnorm49/Const
Constant_float_cuda_Constant_253(0, Constant_253_0);
 // name=cg/incept_v3_c0/conv49/batchnorm49/moving_variance
Constant_float_cuda_Constant_256(0, Constant_256_0);
 // name=cg/incept_v3_c0/conv49/batchnorm49/moving_mean
Constant_float_cuda_Constant_255(0, Constant_255_0);
 // name=cg/incept_v3_c0/conv49/batchnorm49/beta
Constant_float_cuda_Constant_254(0, Constant_254_0);
 // name=cg/incept_v3_c0/conv44/conv2d/kernel
Constant_float_cuda_Constant_227(0, Constant_227_0);
 // name=cg/incept_v3_c0_1/conv44/batchnorm44/Const
Constant_float_cuda_Constant_228(0, Constant_228_0);
 // name=cg/incept_v3_c0/conv44/batchnorm44/moving_variance
Constant_float_cuda_Constant_231(0, Constant_231_0);
 // name=cg/incept_v3_c0/conv44/batchnorm44/moving_mean
Constant_float_cuda_Constant_230(0, Constant_230_0);
 // name=cg/incept_v3_c0/conv44/batchnorm44/beta
Constant_float_cuda_Constant_229(0, Constant_229_0);
 // name=cg/incept_v3_c0/conv45/conv2d/kernel
Constant_float_cuda_Constant_232(0, Constant_232_0);
 // name=cg/incept_v3_c0_1/conv45/batchnorm45/Const
Constant_float_cuda_Constant_233(0, Constant_233_0);
 // name=cg/incept_v3_c0/conv45/batchnorm45/moving_variance
Constant_float_cuda_Constant_236(0, Constant_236_0);
 // name=cg/incept_v3_c0/conv45/batchnorm45/moving_mean
Constant_float_cuda_Constant_235(0, Constant_235_0);
 // name=cg/incept_v3_c0/conv45/batchnorm45/beta
Constant_float_cuda_Constant_234(0, Constant_234_0);
 // name=cg/incept_v3_c0/conv46/conv2d/kernel
Constant_float_cuda_Constant_237(0, Constant_237_0);
 // name=cg/incept_v3_c0_1/conv46/batchnorm46/Const
Constant_float_cuda_Constant_238(0, Constant_238_0);
 // name=cg/incept_v3_c0/conv46/batchnorm46/moving_variance
Constant_float_cuda_Constant_241(0, Constant_241_0);
 // name=cg/incept_v3_c0/conv46/batchnorm46/moving_mean
Constant_float_cuda_Constant_240(0, Constant_240_0);
 // name=cg/incept_v3_c0/conv46/batchnorm46/beta
Constant_float_cuda_Constant_239(0, Constant_239_0);
 // name=cg/incept_v3_c0/conv47/conv2d/kernel
Constant_float_cuda_Constant_242(0, Constant_242_0);
 // name=cg/incept_v3_c0_1/conv47/batchnorm47/Const
Constant_float_cuda_Constant_243(0, Constant_243_0);
 // name=cg/incept_v3_c0/conv47/batchnorm47/moving_variance
Constant_float_cuda_Constant_246(0, Constant_246_0);
 // name=cg/incept_v3_c0/conv47/batchnorm47/moving_mean
Constant_float_cuda_Constant_245(0, Constant_245_0);
 // name=cg/incept_v3_c0/conv47/batchnorm47/beta
Constant_float_cuda_Constant_244(0, Constant_244_0);
 // name=cg/incept_v3_c0/conv48/conv2d/kernel
Constant_float_cuda_Constant_247(0, Constant_247_0);
 // name=cg/incept_v3_c0_1/conv48/batchnorm48/Const
Constant_float_cuda_Constant_248(0, Constant_248_0);
 // name=cg/incept_v3_c0/conv48/batchnorm48/moving_variance
Constant_float_cuda_Constant_251(0, Constant_251_0);
 // name=cg/incept_v3_c0/conv48/batchnorm48/moving_mean
Constant_float_cuda_Constant_250(0, Constant_250_0);
 // name=cg/incept_v3_c0/conv48/batchnorm48/beta
Constant_float_cuda_Constant_249(0, Constant_249_0);
 // name=cg/incept_v3_c0/conv41/conv2d/kernel
Constant_float_cuda_Constant_212(0, Constant_212_0);
 // name=cg/incept_v3_c0_1/conv41/batchnorm41/Const
Constant_float_cuda_Constant_213(0, Constant_213_0);
 // name=cg/incept_v3_c0/conv41/batchnorm41/moving_variance
Constant_float_cuda_Constant_216(0, Constant_216_0);
 // name=cg/incept_v3_c0/conv41/batchnorm41/moving_mean
Constant_float_cuda_Constant_215(0, Constant_215_0);
 // name=cg/incept_v3_c0/conv41/batchnorm41/beta
Constant_float_cuda_Constant_214(0, Constant_214_0);
 // name=cg/incept_v3_c0/conv42/conv2d/kernel
Constant_float_cuda_Constant_217(0, Constant_217_0);
 // name=cg/incept_v3_c0_1/conv42/batchnorm42/Const
Constant_float_cuda_Constant_218(0, Constant_218_0);
 // name=cg/incept_v3_c0/conv42/batchnorm42/moving_variance
Constant_float_cuda_Constant_221(0, Constant_221_0);
 // name=cg/incept_v3_c0/conv42/batchnorm42/moving_mean
Constant_float_cuda_Constant_220(0, Constant_220_0);
 // name=cg/incept_v3_c0/conv42/batchnorm42/beta
Constant_float_cuda_Constant_219(0, Constant_219_0);
 // name=cg/incept_v3_c0/conv43/conv2d/kernel
Constant_float_cuda_Constant_222(0, Constant_222_0);
 // name=cg/incept_v3_c0_1/conv43/batchnorm43/Const
Constant_float_cuda_Constant_223(0, Constant_223_0);
 // name=cg/incept_v3_c0/conv43/batchnorm43/moving_variance
Constant_float_cuda_Constant_226(0, Constant_226_0);
 // name=cg/incept_v3_c0/conv43/batchnorm43/moving_mean
Constant_float_cuda_Constant_225(0, Constant_225_0);
 // name=cg/incept_v3_c0/conv43/batchnorm43/beta
Constant_float_cuda_Constant_224(0, Constant_224_0);
 // name=cg/incept_v3_c0/conv40/conv2d/kernel
Constant_float_cuda_Constant_207(0, Constant_207_0);
 // name=cg/incept_v3_c0_1/conv40/batchnorm40/Const
Constant_float_cuda_Constant_208(0, Constant_208_0);
 // name=cg/incept_v3_c0/conv40/batchnorm40/moving_variance
Constant_float_cuda_Constant_211(0, Constant_211_0);
 // name=cg/incept_v3_c0/conv40/batchnorm40/moving_mean
Constant_float_cuda_Constant_210(0, Constant_210_0);
 // name=cg/incept_v3_c0/conv40/batchnorm40/beta
Constant_float_cuda_Constant_209(0, Constant_209_0);
 // name=cg/incept_v3_c0/conv59/conv2d/kernel
Constant_float_cuda_Constant_303(0, Constant_303_0);
 // name=cg/incept_v3_c0_2/conv59/batchnorm59/Const
Constant_float_cuda_Constant_304(0, Constant_304_0);
 // name=cg/incept_v3_c0/conv59/batchnorm59/moving_variance
Constant_float_cuda_Constant_307(0, Constant_307_0);
 // name=cg/incept_v3_c0/conv59/batchnorm59/moving_mean
Constant_float_cuda_Constant_306(0, Constant_306_0);
 // name=cg/incept_v3_c0/conv59/batchnorm59/beta
Constant_float_cuda_Constant_305(0, Constant_305_0);
 // name=cg/incept_v3_c0/conv54/conv2d/kernel
Constant_float_cuda_Constant_278(0, Constant_278_0);
 // name=cg/incept_v3_c0_2/conv54/batchnorm54/Const
Constant_float_cuda_Constant_279(0, Constant_279_0);
 // name=cg/incept_v3_c0/conv54/batchnorm54/moving_variance
Constant_float_cuda_Constant_282(0, Constant_282_0);
 // name=cg/incept_v3_c0/conv54/batchnorm54/moving_mean
Constant_float_cuda_Constant_281(0, Constant_281_0);
 // name=cg/incept_v3_c0/conv54/batchnorm54/beta
Constant_float_cuda_Constant_280(0, Constant_280_0);
 // name=cg/incept_v3_c0/conv55/conv2d/kernel
Constant_float_cuda_Constant_283(0, Constant_283_0);
 // name=cg/incept_v3_c0_2/conv55/batchnorm55/Const
Constant_float_cuda_Constant_284(0, Constant_284_0);
 // name=cg/incept_v3_c0/conv55/batchnorm55/moving_variance
Constant_float_cuda_Constant_287(0, Constant_287_0);
 // name=cg/incept_v3_c0/conv55/batchnorm55/moving_mean
Constant_float_cuda_Constant_286(0, Constant_286_0);
 // name=cg/incept_v3_c0/conv55/batchnorm55/beta
Constant_float_cuda_Constant_285(0, Constant_285_0);
 // name=cg/incept_v3_c0/conv56/conv2d/kernel
Constant_float_cuda_Constant_288(0, Constant_288_0);
 // name=cg/incept_v3_c0_2/conv56/batchnorm56/Const
Constant_float_cuda_Constant_289(0, Constant_289_0);
 // name=cg/incept_v3_c0/conv56/batchnorm56/moving_variance
Constant_float_cuda_Constant_292(0, Constant_292_0);
 // name=cg/incept_v3_c0/conv56/batchnorm56/moving_mean
Constant_float_cuda_Constant_291(0, Constant_291_0);
 // name=cg/incept_v3_c0/conv56/batchnorm56/beta
Constant_float_cuda_Constant_290(0, Constant_290_0);
 // name=cg/incept_v3_c0/conv57/conv2d/kernel
Constant_float_cuda_Constant_293(0, Constant_293_0);
 // name=cg/incept_v3_c0_2/conv57/batchnorm57/Const
Constant_float_cuda_Constant_294(0, Constant_294_0);
 // name=cg/incept_v3_c0/conv57/batchnorm57/moving_variance
Constant_float_cuda_Constant_297(0, Constant_297_0);
 // name=cg/incept_v3_c0/conv57/batchnorm57/moving_mean
Constant_float_cuda_Constant_296(0, Constant_296_0);
 // name=cg/incept_v3_c0/conv57/batchnorm57/beta
Constant_float_cuda_Constant_295(0, Constant_295_0);
 // name=cg/incept_v3_c0/conv58/conv2d/kernel
Constant_float_cuda_Constant_298(0, Constant_298_0);
 // name=cg/incept_v3_c0_2/conv58/batchnorm58/Const
Constant_float_cuda_Constant_299(0, Constant_299_0);
 // name=cg/incept_v3_c0/conv58/batchnorm58/moving_variance
Constant_float_cuda_Constant_302(0, Constant_302_0);
 // name=cg/incept_v3_c0/conv58/batchnorm58/moving_mean
Constant_float_cuda_Constant_301(0, Constant_301_0);
 // name=cg/incept_v3_c0/conv58/batchnorm58/beta
Constant_float_cuda_Constant_300(0, Constant_300_0);
 // name=cg/incept_v3_c0/conv51/conv2d/kernel
Constant_float_cuda_Constant_263(0, Constant_263_0);
 // name=cg/incept_v3_c0_2/conv51/batchnorm51/Const
Constant_float_cuda_Constant_264(0, Constant_264_0);
 // name=cg/incept_v3_c0/conv51/batchnorm51/moving_variance
Constant_float_cuda_Constant_267(0, Constant_267_0);
 // name=cg/incept_v3_c0/conv51/batchnorm51/moving_mean
Constant_float_cuda_Constant_266(0, Constant_266_0);
 // name=cg/incept_v3_c0/conv51/batchnorm51/beta
Constant_float_cuda_Constant_265(0, Constant_265_0);
 // name=cg/incept_v3_c0/conv52/conv2d/kernel
Constant_float_cuda_Constant_268(0, Constant_268_0);
 // name=cg/incept_v3_c0_2/conv52/batchnorm52/Const
Constant_float_cuda_Constant_269(0, Constant_269_0);
 // name=cg/incept_v3_c0/conv52/batchnorm52/moving_variance
Constant_float_cuda_Constant_272(0, Constant_272_0);
 // name=cg/incept_v3_c0/conv52/batchnorm52/moving_mean
Constant_float_cuda_Constant_271(0, Constant_271_0);
 // name=cg/incept_v3_c0/conv52/batchnorm52/beta
Constant_float_cuda_Constant_270(0, Constant_270_0);
 // name=cg/incept_v3_c0/conv53/conv2d/kernel
Constant_float_cuda_Constant_273(0, Constant_273_0);
 // name=cg/incept_v3_c0_2/conv53/batchnorm53/Const
Constant_float_cuda_Constant_274(0, Constant_274_0);
 // name=cg/incept_v3_c0/conv53/batchnorm53/moving_variance
Constant_float_cuda_Constant_277(0, Constant_277_0);
 // name=cg/incept_v3_c0/conv53/batchnorm53/moving_mean
Constant_float_cuda_Constant_276(0, Constant_276_0);
 // name=cg/incept_v3_c0/conv53/batchnorm53/beta
Constant_float_cuda_Constant_275(0, Constant_275_0);
 // name=cg/incept_v3_c0/conv50/conv2d/kernel
Constant_float_cuda_Constant_258(0, Constant_258_0);
 // name=cg/incept_v3_c0_2/conv50/batchnorm50/Const
Constant_float_cuda_Constant_259(0, Constant_259_0);
 // name=cg/incept_v3_c0/conv50/batchnorm50/moving_variance
Constant_float_cuda_Constant_262(0, Constant_262_0);
 // name=cg/incept_v3_c0/conv50/batchnorm50/moving_mean
Constant_float_cuda_Constant_261(0, Constant_261_0);
 // name=cg/incept_v3_c0/conv50/batchnorm50/beta
Constant_float_cuda_Constant_260(0, Constant_260_0);
 // name=cg/incept_v3_c0/conv69/conv2d/kernel
Constant_float_cuda_Constant_354(0, Constant_354_0);
 // name=cg/incept_v3_c0_3/conv69/batchnorm69/Const
Constant_float_cuda_Constant_355(0, Constant_355_0);
 // name=cg/incept_v3_c0/conv69/batchnorm69/moving_variance
Constant_float_cuda_Constant_358(0, Constant_358_0);
 // name=cg/incept_v3_c0/conv69/batchnorm69/moving_mean
Constant_float_cuda_Constant_357(0, Constant_357_0);
 // name=cg/incept_v3_c0/conv69/batchnorm69/beta
Constant_float_cuda_Constant_356(0, Constant_356_0);
 // name=cg/incept_v3_c0/conv64/conv2d/kernel
Constant_float_cuda_Constant_329(0, Constant_329_0);
 // name=cg/incept_v3_c0_3/conv64/batchnorm64/Const
Constant_float_cuda_Constant_330(0, Constant_330_0);
 // name=cg/incept_v3_c0/conv64/batchnorm64/moving_variance
Constant_float_cuda_Constant_333(0, Constant_333_0);
 // name=cg/incept_v3_c0/conv64/batchnorm64/moving_mean
Constant_float_cuda_Constant_332(0, Constant_332_0);
 // name=cg/incept_v3_c0/conv64/batchnorm64/beta
Constant_float_cuda_Constant_331(0, Constant_331_0);
 // name=cg/incept_v3_c0/conv65/conv2d/kernel
Constant_float_cuda_Constant_334(0, Constant_334_0);
 // name=cg/incept_v3_c0_3/conv65/batchnorm65/Const
Constant_float_cuda_Constant_335(0, Constant_335_0);
 // name=cg/incept_v3_c0/conv65/batchnorm65/moving_variance
Constant_float_cuda_Constant_338(0, Constant_338_0);
 // name=cg/incept_v3_c0/conv65/batchnorm65/moving_mean
Constant_float_cuda_Constant_337(0, Constant_337_0);
 // name=cg/incept_v3_c0/conv65/batchnorm65/beta
Constant_float_cuda_Constant_336(0, Constant_336_0);
 // name=cg/incept_v3_c0/conv66/conv2d/kernel
Constant_float_cuda_Constant_339(0, Constant_339_0);
 // name=cg/incept_v3_c0_3/conv66/batchnorm66/Const
Constant_float_cuda_Constant_340(0, Constant_340_0);
 // name=cg/incept_v3_c0/conv66/batchnorm66/moving_variance
Constant_float_cuda_Constant_343(0, Constant_343_0);
 // name=cg/incept_v3_c0/conv66/batchnorm66/moving_mean
Constant_float_cuda_Constant_342(0, Constant_342_0);
 // name=cg/incept_v3_c0/conv66/batchnorm66/beta
Constant_float_cuda_Constant_341(0, Constant_341_0);
 // name=cg/incept_v3_c0/conv67/conv2d/kernel
Constant_float_cuda_Constant_344(0, Constant_344_0);
 // name=cg/incept_v3_c0_3/conv67/batchnorm67/Const
Constant_float_cuda_Constant_345(0, Constant_345_0);
 // name=cg/incept_v3_c0/conv67/batchnorm67/moving_variance
Constant_float_cuda_Constant_348(0, Constant_348_0);
 // name=cg/incept_v3_c0/conv67/batchnorm67/moving_mean
Constant_float_cuda_Constant_347(0, Constant_347_0);
 // name=cg/incept_v3_c0/conv67/batchnorm67/beta
Constant_float_cuda_Constant_346(0, Constant_346_0);
 // name=cg/incept_v3_c0/conv68/conv2d/kernel
Constant_float_cuda_Constant_349(0, Constant_349_0);
 // name=cg/incept_v3_c0_3/conv68/batchnorm68/Const
Constant_float_cuda_Constant_350(0, Constant_350_0);
 // name=cg/incept_v3_c0/conv68/batchnorm68/moving_variance
Constant_float_cuda_Constant_353(0, Constant_353_0);
 // name=cg/incept_v3_c0/conv68/batchnorm68/moving_mean
Constant_float_cuda_Constant_352(0, Constant_352_0);
 // name=cg/incept_v3_c0/conv68/batchnorm68/beta
Constant_float_cuda_Constant_351(0, Constant_351_0);
 // name=cg/incept_v3_c0/conv61/conv2d/kernel
Constant_float_cuda_Constant_314(0, Constant_314_0);
 // name=cg/incept_v3_c0_3/conv61/batchnorm61/Const
Constant_float_cuda_Constant_315(0, Constant_315_0);
 // name=cg/incept_v3_c0/conv61/batchnorm61/moving_variance
Constant_float_cuda_Constant_318(0, Constant_318_0);
 // name=cg/incept_v3_c0/conv61/batchnorm61/moving_mean
Constant_float_cuda_Constant_317(0, Constant_317_0);
 // name=cg/incept_v3_c0/conv61/batchnorm61/beta
Constant_float_cuda_Constant_316(0, Constant_316_0);
 // name=cg/incept_v3_c0/conv62/conv2d/kernel
Constant_float_cuda_Constant_319(0, Constant_319_0);
 // name=cg/incept_v3_c0_3/conv62/batchnorm62/Const
Constant_float_cuda_Constant_320(0, Constant_320_0);
 // name=cg/incept_v3_c0/conv62/batchnorm62/moving_variance
Constant_float_cuda_Constant_323(0, Constant_323_0);
 // name=cg/incept_v3_c0/conv62/batchnorm62/moving_mean
Constant_float_cuda_Constant_322(0, Constant_322_0);
 // name=cg/incept_v3_c0/conv62/batchnorm62/beta
Constant_float_cuda_Constant_321(0, Constant_321_0);
 // name=cg/incept_v3_c0/conv63/conv2d/kernel
Constant_float_cuda_Constant_324(0, Constant_324_0);
 // name=cg/incept_v3_c0_3/conv63/batchnorm63/Const
Constant_float_cuda_Constant_325(0, Constant_325_0);
 // name=cg/incept_v3_c0/conv63/batchnorm63/moving_variance
Constant_float_cuda_Constant_328(0, Constant_328_0);
 // name=cg/incept_v3_c0/conv63/batchnorm63/moving_mean
Constant_float_cuda_Constant_327(0, Constant_327_0);
 // name=cg/incept_v3_c0/conv63/batchnorm63/beta
Constant_float_cuda_Constant_326(0, Constant_326_0);
 // name=cg/incept_v3_c0/conv60/conv2d/kernel
Constant_float_cuda_Constant_309(0, Constant_309_0);
 // name=cg/incept_v3_c0_3/conv60/batchnorm60/Const
Constant_float_cuda_Constant_310(0, Constant_310_0);
 // name=cg/incept_v3_c0/conv60/batchnorm60/moving_variance
Constant_float_cuda_Constant_313(0, Constant_313_0);
 // name=cg/incept_v3_c0/conv60/batchnorm60/moving_mean
Constant_float_cuda_Constant_312(0, Constant_312_0);
 // name=cg/incept_v3_c0/conv60/batchnorm60/beta
Constant_float_cuda_Constant_311(0, Constant_311_0);
 // name=cg/incept_v3_d0/conv72/conv2d/kernel
Constant_float_cuda_Constant_370(0, Constant_370_0);
 // name=cg/incept_v3_d0/conv72/batchnorm72/moving_variance
Constant_float_cuda_Constant_374(0, Constant_374_0);
 // name=cg/incept_v3_d0/conv72/batchnorm72/moving_mean
Constant_float_cuda_Constant_373(0, Constant_373_0);
 // name=cg/incept_v3_d0/conv72/batchnorm72/beta
Constant_float_cuda_Constant_372(0, Constant_372_0);
 // name=cg/incept_v3_d0/conv72/batchnorm72/Const
Constant_float_cuda_Constant_371(0, Constant_371_0);
 // name=cg/incept_v3_d0/conv73/conv2d/kernel
Constant_float_cuda_Constant_375(0, Constant_375_0);
 // name=cg/incept_v3_d0/conv73/batchnorm73/moving_variance
Constant_float_cuda_Constant_379(0, Constant_379_0);
 // name=cg/incept_v3_d0/conv73/batchnorm73/moving_mean
Constant_float_cuda_Constant_378(0, Constant_378_0);
 // name=cg/incept_v3_d0/conv73/batchnorm73/beta
Constant_float_cuda_Constant_377(0, Constant_377_0);
 // name=cg/incept_v3_d0/conv73/batchnorm73/Const
Constant_float_cuda_Constant_376(0, Constant_376_0);
 // name=cg/incept_v3_d0/conv74/conv2d/kernel
Constant_float_cuda_Constant_380(0, Constant_380_0);
 // name=cg/incept_v3_d0/conv74/batchnorm74/moving_variance
Constant_float_cuda_Constant_384(0, Constant_384_0);
 // name=cg/incept_v3_d0/conv74/batchnorm74/moving_mean
Constant_float_cuda_Constant_383(0, Constant_383_0);
 // name=cg/incept_v3_d0/conv74/batchnorm74/beta
Constant_float_cuda_Constant_382(0, Constant_382_0);
 // name=cg/incept_v3_d0/conv74/batchnorm74/Const
Constant_float_cuda_Constant_381(0, Constant_381_0);
 // name=cg/incept_v3_d0/conv75/conv2d/kernel
Constant_float_cuda_Constant_385(0, Constant_385_0);
 // name=cg/incept_v3_d0/conv75/batchnorm75/moving_variance
Constant_float_cuda_Constant_389(0, Constant_389_0);
 // name=cg/incept_v3_d0/conv75/batchnorm75/moving_mean
Constant_float_cuda_Constant_388(0, Constant_388_0);
 // name=cg/incept_v3_d0/conv75/batchnorm75/beta
Constant_float_cuda_Constant_387(0, Constant_387_0);
 // name=cg/incept_v3_d0/conv75/batchnorm75/Const
Constant_float_cuda_Constant_386(0, Constant_386_0);
 // name=cg/incept_v3_d0/conv70/conv2d/kernel
Constant_float_cuda_Constant_360(0, Constant_360_0);
 // name=cg/incept_v3_d0/conv70/batchnorm70/moving_variance
Constant_float_cuda_Constant_364(0, Constant_364_0);
 // name=cg/incept_v3_d0/conv70/batchnorm70/moving_mean
Constant_float_cuda_Constant_363(0, Constant_363_0);
 // name=cg/incept_v3_d0/conv70/batchnorm70/beta
Constant_float_cuda_Constant_362(0, Constant_362_0);
 // name=cg/incept_v3_d0/conv70/batchnorm70/Const
Constant_float_cuda_Constant_361(0, Constant_361_0);
 // name=cg/incept_v3_d0/conv71/conv2d/kernel
Constant_float_cuda_Constant_365(0, Constant_365_0);
 // name=cg/incept_v3_d0/conv71/batchnorm71/moving_variance
Constant_float_cuda_Constant_369(0, Constant_369_0);
 // name=cg/incept_v3_d0/conv71/batchnorm71/moving_mean
Constant_float_cuda_Constant_368(0, Constant_368_0);
 // name=cg/incept_v3_d0/conv71/batchnorm71/beta
Constant_float_cuda_Constant_367(0, Constant_367_0);
 // name=cg/incept_v3_d0/conv71/batchnorm71/Const
Constant_float_cuda_Constant_366(0, Constant_366_0);
 // name=cg/incept_v3_e0/conv84/conv2d/kernel
Constant_float_cuda_Constant_431(0, Constant_431_0);
 // name=cg/incept_v3_e0/conv84/batchnorm84/moving_variance
Constant_float_cuda_Constant_435(0, Constant_435_0);
 // name=cg/incept_v3_e0/conv84/batchnorm84/moving_mean
Constant_float_cuda_Constant_434(0, Constant_434_0);
 // name=cg/incept_v3_e0/conv84/batchnorm84/beta
Constant_float_cuda_Constant_433(0, Constant_433_0);
 // name=cg/incept_v3_e0/conv84/batchnorm84/Const
Constant_float_cuda_Constant_432(0, Constant_432_0);
 // name=cg/incept_v3_e0/conv80/conv2d/kernel
Constant_float_cuda_Constant_411(0, Constant_411_0);
 // name=cg/incept_v3_e0/conv80/batchnorm80/moving_variance
Constant_float_cuda_Constant_415(0, Constant_415_0);
 // name=cg/incept_v3_e0/conv80/batchnorm80/moving_mean
Constant_float_cuda_Constant_414(0, Constant_414_0);
 // name=cg/incept_v3_e0/conv80/batchnorm80/beta
Constant_float_cuda_Constant_413(0, Constant_413_0);
 // name=cg/incept_v3_e0/conv80/batchnorm80/Const
Constant_float_cuda_Constant_412(0, Constant_412_0);
 // name=cg/incept_v3_e0/conv81/conv2d/kernel
Constant_float_cuda_Constant_416(0, Constant_416_0);
 // name=cg/incept_v3_e0/conv81/batchnorm81/moving_variance
Constant_float_cuda_Constant_420(0, Constant_420_0);
 // name=cg/incept_v3_e0/conv81/batchnorm81/moving_mean
Constant_float_cuda_Constant_419(0, Constant_419_0);
 // name=cg/incept_v3_e0/conv81/batchnorm81/beta
Constant_float_cuda_Constant_418(0, Constant_418_0);
 // name=cg/incept_v3_e0/conv81/batchnorm81/Const
Constant_float_cuda_Constant_417(0, Constant_417_0);
 // name=cg/incept_v3_e0/conv83/conv2d/kernel
Constant_float_cuda_Constant_426(0, Constant_426_0);
 // name=cg/incept_v3_e0/conv83/batchnorm83/moving_variance
Constant_float_cuda_Constant_430(0, Constant_430_0);
 // name=cg/incept_v3_e0/conv83/batchnorm83/moving_mean
Constant_float_cuda_Constant_429(0, Constant_429_0);
 // name=cg/incept_v3_e0/conv83/batchnorm83/beta
Constant_float_cuda_Constant_428(0, Constant_428_0);
 // name=cg/incept_v3_e0/conv83/batchnorm83/Const
Constant_float_cuda_Constant_427(0, Constant_427_0);
 // name=cg/incept_v3_e0/conv82/conv2d/kernel
Constant_float_cuda_Constant_421(0, Constant_421_0);
 // name=cg/incept_v3_e0/conv82/batchnorm82/moving_variance
Constant_float_cuda_Constant_425(0, Constant_425_0);
 // name=cg/incept_v3_e0/conv82/batchnorm82/moving_mean
Constant_float_cuda_Constant_424(0, Constant_424_0);
 // name=cg/incept_v3_e0/conv82/batchnorm82/beta
Constant_float_cuda_Constant_423(0, Constant_423_0);
 // name=cg/incept_v3_e0/conv82/batchnorm82/Const
Constant_float_cuda_Constant_422(0, Constant_422_0);
 // name=cg/incept_v3_e0/conv77/conv2d/kernel
Constant_float_cuda_Constant_396(0, Constant_396_0);
 // name=cg/incept_v3_e0/conv77/batchnorm77/moving_variance
Constant_float_cuda_Constant_400(0, Constant_400_0);
 // name=cg/incept_v3_e0/conv77/batchnorm77/moving_mean
Constant_float_cuda_Constant_399(0, Constant_399_0);
 // name=cg/incept_v3_e0/conv77/batchnorm77/beta
Constant_float_cuda_Constant_398(0, Constant_398_0);
 // name=cg/incept_v3_e0/conv77/batchnorm77/Const
Constant_float_cuda_Constant_397(0, Constant_397_0);
 // name=cg/incept_v3_e0/conv79/conv2d/kernel
Constant_float_cuda_Constant_406(0, Constant_406_0);
 // name=cg/incept_v3_e0/conv79/batchnorm79/moving_variance
Constant_float_cuda_Constant_410(0, Constant_410_0);
 // name=cg/incept_v3_e0/conv79/batchnorm79/moving_mean
Constant_float_cuda_Constant_409(0, Constant_409_0);
 // name=cg/incept_v3_e0/conv79/batchnorm79/beta
Constant_float_cuda_Constant_408(0, Constant_408_0);
 // name=cg/incept_v3_e0/conv79/batchnorm79/Const
Constant_float_cuda_Constant_407(0, Constant_407_0);
 // name=cg/incept_v3_e0/conv78/conv2d/kernel
Constant_float_cuda_Constant_401(0, Constant_401_0);
 // name=cg/incept_v3_e0/conv78/batchnorm78/moving_variance
Constant_float_cuda_Constant_405(0, Constant_405_0);
 // name=cg/incept_v3_e0/conv78/batchnorm78/moving_mean
Constant_float_cuda_Constant_404(0, Constant_404_0);
 // name=cg/incept_v3_e0/conv78/batchnorm78/beta
Constant_float_cuda_Constant_403(0, Constant_403_0);
 // name=cg/incept_v3_e0/conv78/batchnorm78/Const
Constant_float_cuda_Constant_402(0, Constant_402_0);
 // name=cg/incept_v3_e0/conv76/conv2d/kernel
Constant_float_cuda_Constant_391(0, Constant_391_0);
 // name=cg/incept_v3_e0/conv76/batchnorm76/moving_variance
Constant_float_cuda_Constant_395(0, Constant_395_0);
 // name=cg/incept_v3_e0/conv76/batchnorm76/moving_mean
Constant_float_cuda_Constant_394(0, Constant_394_0);
 // name=cg/incept_v3_e0/conv76/batchnorm76/beta
Constant_float_cuda_Constant_393(0, Constant_393_0);
 // name=cg/incept_v3_e0/conv76/batchnorm76/Const
Constant_float_cuda_Constant_392(0, Constant_392_0);
 // name=cg/incept_v3_e0/conv93/conv2d/kernel
Constant_float_cuda_Constant_477(0, Constant_477_0);
 // name=cg/incept_v3_e0_1/conv93/batchnorm93/Const
Constant_float_cuda_Constant_478(0, Constant_478_0);
 // name=cg/incept_v3_e0/conv93/batchnorm93/moving_variance
Constant_float_cuda_Constant_481(0, Constant_481_0);
 // name=cg/incept_v3_e0/conv93/batchnorm93/moving_mean
Constant_float_cuda_Constant_480(0, Constant_480_0);
 // name=cg/incept_v3_e0/conv93/batchnorm93/beta
Constant_float_cuda_Constant_479(0, Constant_479_0);
 // name=cg/incept_v3_e0/conv89/conv2d/kernel
Constant_float_cuda_Constant_457(0, Constant_457_0);
 // name=cg/incept_v3_e0_1/conv89/batchnorm89/Const
Constant_float_cuda_Constant_458(0, Constant_458_0);
 // name=cg/incept_v3_e0/conv89/batchnorm89/moving_variance
Constant_float_cuda_Constant_461(0, Constant_461_0);
 // name=cg/incept_v3_e0/conv89/batchnorm89/moving_mean
Constant_float_cuda_Constant_460(0, Constant_460_0);
 // name=cg/incept_v3_e0/conv89/batchnorm89/beta
Constant_float_cuda_Constant_459(0, Constant_459_0);
 // name=cg/incept_v3_e0/conv90/conv2d/kernel
Constant_float_cuda_Constant_462(0, Constant_462_0);
 // name=cg/incept_v3_e0_1/conv90/batchnorm90/Const
Constant_float_cuda_Constant_463(0, Constant_463_0);
 // name=cg/incept_v3_e0/conv90/batchnorm90/moving_variance
Constant_float_cuda_Constant_466(0, Constant_466_0);
 // name=cg/incept_v3_e0/conv90/batchnorm90/moving_mean
Constant_float_cuda_Constant_465(0, Constant_465_0);
 // name=cg/incept_v3_e0/conv90/batchnorm90/beta
Constant_float_cuda_Constant_464(0, Constant_464_0);
 // name=cg/incept_v3_e0/conv92/conv2d/kernel
Constant_float_cuda_Constant_472(0, Constant_472_0);
 // name=cg/incept_v3_e0_1/conv92/batchnorm92/Const
Constant_float_cuda_Constant_473(0, Constant_473_0);
 // name=cg/incept_v3_e0/conv92/batchnorm92/moving_variance
Constant_float_cuda_Constant_476(0, Constant_476_0);
 // name=cg/incept_v3_e0/conv92/batchnorm92/moving_mean
Constant_float_cuda_Constant_475(0, Constant_475_0);
 // name=cg/incept_v3_e0/conv92/batchnorm92/beta
Constant_float_cuda_Constant_474(0, Constant_474_0);
 // name=cg/incept_v3_e0/conv91/conv2d/kernel
Constant_float_cuda_Constant_467(0, Constant_467_0);
 // name=cg/incept_v3_e0_1/conv91/batchnorm91/Const
Constant_float_cuda_Constant_468(0, Constant_468_0);
 // name=cg/incept_v3_e0/conv91/batchnorm91/moving_variance
Constant_float_cuda_Constant_471(0, Constant_471_0);
 // name=cg/incept_v3_e0/conv91/batchnorm91/moving_mean
Constant_float_cuda_Constant_470(0, Constant_470_0);
 // name=cg/incept_v3_e0/conv91/batchnorm91/beta
Constant_float_cuda_Constant_469(0, Constant_469_0);
 // name=cg/incept_v3_e0/conv86/conv2d/kernel
Constant_float_cuda_Constant_442(0, Constant_442_0);
 // name=cg/incept_v3_e0_1/conv86/batchnorm86/Const
Constant_float_cuda_Constant_443(0, Constant_443_0);
 // name=cg/incept_v3_e0/conv86/batchnorm86/moving_variance
Constant_float_cuda_Constant_446(0, Constant_446_0);
 // name=cg/incept_v3_e0/conv86/batchnorm86/moving_mean
Constant_float_cuda_Constant_445(0, Constant_445_0);
 // name=cg/incept_v3_e0/conv86/batchnorm86/beta
Constant_float_cuda_Constant_444(0, Constant_444_0);
 // name=cg/incept_v3_e0/conv88/conv2d/kernel
Constant_float_cuda_Constant_452(0, Constant_452_0);
 // name=cg/incept_v3_e0_1/conv88/batchnorm88/Const
Constant_float_cuda_Constant_453(0, Constant_453_0);
 // name=cg/incept_v3_e0/conv88/batchnorm88/moving_variance
Constant_float_cuda_Constant_456(0, Constant_456_0);
 // name=cg/incept_v3_e0/conv88/batchnorm88/moving_mean
Constant_float_cuda_Constant_455(0, Constant_455_0);
 // name=cg/incept_v3_e0/conv88/batchnorm88/beta
Constant_float_cuda_Constant_454(0, Constant_454_0);
 // name=cg/incept_v3_e0/conv87/conv2d/kernel
Constant_float_cuda_Constant_447(0, Constant_447_0);
 // name=cg/incept_v3_e0_1/conv87/batchnorm87/Const
Constant_float_cuda_Constant_448(0, Constant_448_0);
 // name=cg/incept_v3_e0/conv87/batchnorm87/moving_variance
Constant_float_cuda_Constant_451(0, Constant_451_0);
 // name=cg/incept_v3_e0/conv87/batchnorm87/moving_mean
Constant_float_cuda_Constant_450(0, Constant_450_0);
 // name=cg/incept_v3_e0/conv87/batchnorm87/beta
Constant_float_cuda_Constant_449(0, Constant_449_0);
 // name=cg/incept_v3_e0/conv85/conv2d/kernel
Constant_float_cuda_Constant_437(0, Constant_437_0);
 // name=cg/incept_v3_e0_1/conv85/batchnorm85/Const
Constant_float_cuda_Constant_438(0, Constant_438_0);
 // name=cg/incept_v3_e0/conv85/batchnorm85/moving_variance
Constant_float_cuda_Constant_441(0, Constant_441_0);
 // name=cg/incept_v3_e0/conv85/batchnorm85/moving_mean
Constant_float_cuda_Constant_440(0, Constant_440_0);
 // name=cg/incept_v3_e0/conv85/batchnorm85/beta
Constant_float_cuda_Constant_439(0, Constant_439_0);
 // name=cg/affine0/biases
Constant_float_cuda_Constant_485(0, Constant_485_0);
}

// Node name:	Convolution_812
// Description:	Convolution
// Input:
//	- name: Concat_810_0	type: float	shape: Shape{32, 1280, 8, 8}
//	- name: Reshape_811_0	type: float	shape: Shape{320, 1280, 1, 1}
// Output:
//	- name: Convolution_812_0	type: float	shape: Shape{32, 320, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_812(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 320, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 320, 1280, 1, 1));
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
// Node name:	MaxPool_855
// Description:	MaxPool
// Input:
//	- name: Concat_848_0	type: float	shape: Shape{32, 2048, 8, 8}
// Output:
//	- name: MaxPool_855_0	type: float	shape: Shape{32, 2048, 8, 8}
void MaxPool_float_float_cuda_lib_MaxPool_855(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Convolution_852
// Description:	Convolution
// Input:
//	- name: Concat_848_0	type: float	shape: Shape{32, 2048, 8, 8}
//	- name: Reshape_851_0	type: float	shape: Shape{384, 2048, 1, 1}
// Output:
//	- name: Convolution_852_0	type: float	shape: Shape{32, 384, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_852(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 2048, 1, 1));
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
// Node name:	Convolution_841
// Description:	Convolution
// Input:
//	- name: Relu_839_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Reshape_840_0	type: float	shape: Shape{384, 384, 1, 3}
// Output:
//	- name: Convolution_841_0	type: float	shape: Shape{32, 384, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_841(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 384, 1, 3));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Concat_848
// Description:	Concat
// Input:
//	- name: Relu_823_0	type: float	shape: Shape{32, 320, 8, 8}
//	- name: Relu_837_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Relu_838_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Relu_846_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Relu_847_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Relu_833_0	type: float	shape: Shape{32, 192, 8, 8}
// Output:
//	- name: Concat_848_0	type: float	shape: Shape{32, 2048, 8, 8}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_float_float_float_cuda_Concat_848(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0)
{
    uint32_t inputs_strides[] = {20480, 24576, 24576, 24576, 24576, 12288};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 4194304)
    {
        uint32_t block_id = tid / 131072;
        uint32_t block_idx = tid % 131072;
        uint32_t output_idx = block_id * 131072 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
        if(block_idx < inputs_strides[3])
        {
            output0[output_idx] = input3[block_id * inputs_strides[3] + block_idx];
            return;
        }
        block_idx -= inputs_strides[3];
        if(block_idx < inputs_strides[4])
        {
            output0[output_idx] = input4[block_id * inputs_strides[4] + block_idx];
            return;
        }
        block_idx -= inputs_strides[4];
        if(block_idx < inputs_strides[5])
        {
            output0[output_idx] = input5[block_id * inputs_strides[5] + block_idx];
            return;
        }
        block_idx -= inputs_strides[5];
    }

}
extern void Concat_float_float_float_float_float_float_float_cuda_Concat_848_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0) {
    Concat_float_float_float_float_float_float_float_cuda_Concat_848<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0);
}
// Node name:	Add_891
// Description:	Add
// Input:
//	- name: Dot_889_0	type: float	shape: Shape{32, 1001}
//	- name: Broadcast_890_0	type: float	shape: Shape{32, 1001}
// Output:
//	- name: Add_891_0	type: float	shape: Shape{32, 1001}
extern "C" __launch_bounds__(416) __global__ void Add_float_float_float_cuda_Add_891(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 416 + threadIdx.x] = add(input0[blockIdx.x * 416 + threadIdx.x], input1[blockIdx.x * 416 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_891_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_891<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_859
// Description:	Reshape
// Input:
//	- name: Constant_477_0	type: float	shape: Shape{1, 1, 2048, 192}
// Output:
//	- name: Reshape_859_0	type: float	shape: Shape{192, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_859(float* input0, float* output0)
{
    uint32_t input_strides0 = 192;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 2048;
    size_t nx = 192;
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
extern void Reshape_float_float_cuda_Reshape_859_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_859<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_860
// Description:	Convolution
// Input:
//	- name: MaxPool_855_0	type: float	shape: Shape{32, 2048, 8, 8}
//	- name: Reshape_859_0	type: float	shape: Shape{192, 2048, 1, 1}
// Output:
//	- name: Convolution_860_0	type: float	shape: Shape{32, 192, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_860(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 2048, 1, 1));
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
// Node name:	Convolution_488
// Description:	Convolution
// Input:
//	- name: Reshape_486_0	type: float	shape: Shape{32, 3, 299, 299}
//	- name: Reshape_487_0	type: float	shape: Shape{32, 3, 3, 3}
// Output:
//	- name: Convolution_488_0	type: float	shape: Shape{32, 32, 149, 149}
void Convolution_float_float_float_cuda_lib_Convolution_488(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 3, 299, 299));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 32, 149, 149));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 3, 3));
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
// Node name:	Convolution_843
// Description:	Convolution
// Input:
//	- name: Relu_839_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Reshape_842_0	type: float	shape: Shape{384, 384, 3, 1}
// Output:
//	- name: Convolution_843_0	type: float	shape: Shape{32, 384, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_843(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 384, 3, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Reshape_851
// Description:	Reshape
// Input:
//	- name: Constant_442_0	type: float	shape: Shape{1, 1, 2048, 384}
// Output:
//	- name: Reshape_851_0	type: float	shape: Shape{384, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_851(float* input0, float* output0)
{
    uint32_t input_strides0 = 384;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 2048;
    size_t nx = 384;
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
extern void Reshape_float_float_cuda_Reshape_851_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_851<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_889
// Description:	Dot
// Input:
//	- name: Reshape_888_0	type: float	shape: Shape{32, 2048}
//	- name: Constant_484_0	type: float	shape: Shape{2048, 1001}
// Output:
//	- name: Dot_889_0	type: float	shape: Shape{32, 1001}
void Dot_float_float_float_cuda_lib_Dot_889(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 32, 2048, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 2048, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Convolution_795
// Description:	Convolution
// Input:
//	- name: Relu_792_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Reshape_794_0	type: float	shape: Shape{320, 192, 3, 3}
// Output:
//	- name: Convolution_795_0	type: float	shape: Shape{32, 320, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_795(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 320, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 320, 192, 3, 3));
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
// Node name:	BatchNormInference_517
// Description:	BatchNormInference
// Input:
//	- name: Constant_33_0	type: float	shape: Shape{48}
//	- name: Constant_34_0	type: float	shape: Shape{48}
//	- name: Convolution_512_0	type: float	shape: Shape{32, 48, 35, 35}
//	- name: Constant_35_0	type: float	shape: Shape{48}
//	- name: Constant_36_0	type: float	shape: Shape{48}
// Output:
//	- name: BatchNormInference_517_0	type: float	shape: Shape{32, 48, 35, 35}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 35 * 35;
    const int c_id = blockIdx.x % 48;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 35 * 35; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	BatchNormInference_820
// Description:	BatchNormInference
// Input:
//	- name: Constant_412_0	type: float	shape: Shape{448}
//	- name: Constant_413_0	type: float	shape: Shape{448}
//	- name: Convolution_816_0	type: float	shape: Shape{32, 448, 8, 8}
//	- name: Constant_414_0	type: float	shape: Shape{448}
//	- name: Constant_415_0	type: float	shape: Shape{448}
// Output:
//	- name: BatchNormInference_820_0	type: float	shape: Shape{32, 448, 8, 8}
extern "C" __launch_bounds__(64) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_820(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 8 * 8;
    const int c_id = blockIdx.x % 448;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 8 * 8; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_820_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_820<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_822
// Description:	Convolution
// Input:
//	- name: AvgPool_817_0	type: float	shape: Shape{32, 1280, 8, 8}
//	- name: Reshape_821_0	type: float	shape: Shape{192, 1280, 1, 1}
// Output:
//	- name: Convolution_822_0	type: float	shape: Shape{32, 192, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_822(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 1280, 1, 1));
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
// Node name:	Result_892
// Description:	Result
// Input:
//	- name: Add_891_0	type: float	shape: Shape{32, 1001}
// Output:
//	- name: Result_892_0	type: float	shape: Shape{32, 1001}
void Result_float_float_cuda_lib_Result_892(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Reshape_849
// Description:	Reshape
// Input:
//	- name: Constant_437_0	type: float	shape: Shape{1, 1, 2048, 320}
// Output:
//	- name: Reshape_849_0	type: float	shape: Shape{320, 2048, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_849(float* input0, float* output0)
{
    uint32_t input_strides0 = 320;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 2048;
    size_t nx = 320;
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
extern void Reshape_float_float_cuda_Reshape_849_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_849<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_816
// Description:	Convolution
// Input:
//	- name: Concat_810_0	type: float	shape: Shape{32, 1280, 8, 8}
//	- name: Reshape_815_0	type: float	shape: Shape{448, 1280, 1, 1}
// Output:
//	- name: Convolution_816_0	type: float	shape: Shape{32, 448, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_816(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 448, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 448, 1280, 1, 1));
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
// Node name:	Reshape_815
// Description:	Reshape
// Input:
//	- name: Constant_411_0	type: float	shape: Shape{1, 1, 1280, 448}
// Output:
//	- name: Reshape_815_0	type: float	shape: Shape{448, 1280, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_815(float* input0, float* output0)
{
    uint32_t input_strides0 = 448;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1280;
    size_t nx = 448;
    size_t ny = 1280;
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
extern void Reshape_float_float_cuda_Reshape_815_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_815<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	MaxPool_789
// Description:	MaxPool
// Input:
//	- name: Concat_784_0	type: float	shape: Shape{32, 768, 17, 17}
// Output:
//	- name: MaxPool_789_0	type: float	shape: Shape{32, 768, 8, 8}
void MaxPool_float_float_cuda_lib_MaxPool_789(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 8, 8));
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
// Node name:	BatchNormInference_502
// Description:	BatchNormInference
// Input:
//	- name: Constant_18_0	type: float	shape: Shape{80}
//	- name: Constant_19_0	type: float	shape: Shape{80}
//	- name: Convolution_501_0	type: float	shape: Shape{32, 80, 73, 73}
//	- name: Constant_20_0	type: float	shape: Shape{80}
//	- name: Constant_21_0	type: float	shape: Shape{80}
// Output:
//	- name: BatchNormInference_502_0	type: float	shape: Shape{32, 80, 73, 73}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_502(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 73 * 73;
    const int c_id = blockIdx.x % 80;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 73 * 73; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_502_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_502<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	AvgPool_817
// Description:	AvgPool
// Input:
//	- name: Concat_810_0	type: float	shape: Shape{32, 1280, 8, 8}
// Output:
//	- name: AvgPool_817_0	type: float	shape: Shape{32, 1280, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_817(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 1280, 8, 8));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Reshape_794
// Description:	Reshape
// Input:
//	- name: Constant_365_0	type: float	shape: Shape{3, 3, 192, 320}
// Output:
//	- name: Reshape_794_0	type: float	shape: Shape{320, 192, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_794(float* input0, float* output0)
{
    uint32_t input_strides0 = 61440;
    uint32_t input_strides1 = 320;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 1728;
    size_t nx = 320;
    size_t ny = 192;
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
extern void Reshape_float_float_cuda_Reshape_794_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_794<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_808
// Description:	BatchNormInference
// Input:
//	- name: Constant_386_0	type: float	shape: Shape{192}
//	- name: Constant_387_0	type: float	shape: Shape{192}
//	- name: Convolution_807_0	type: float	shape: Shape{32, 192, 8, 8}
//	- name: Constant_388_0	type: float	shape: Shape{192}
//	- name: Constant_389_0	type: float	shape: Shape{192}
// Output:
//	- name: BatchNormInference_808_0	type: float	shape: Shape{32, 192, 8, 8}
extern "C" __launch_bounds__(64) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 8 * 8;
    const int c_id = blockIdx.x % 192;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 8 * 8; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Concat_810
// Description:	Concat
// Input:
//	- name: Relu_800_0	type: float	shape: Shape{32, 320, 8, 8}
//	- name: Relu_809_0	type: float	shape: Shape{32, 192, 8, 8}
//	- name: MaxPool_789_0	type: float	shape: Shape{32, 768, 8, 8}
// Output:
//	- name: Concat_810_0	type: float	shape: Shape{32, 1280, 8, 8}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_cuda_Concat_810(float* input0, float* input1, float* input2, float* output0)
{
    uint32_t inputs_strides[] = {20480, 12288, 49152};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 2621440)
    {
        uint32_t block_id = tid / 81920;
        uint32_t block_idx = tid % 81920;
        uint32_t output_idx = block_id * 81920 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
    }

}
extern void Concat_float_float_float_float_cuda_Concat_810_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    Concat_float_float_float_float_cuda_Concat_810<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Convolution_678
// Description:	Convolution
// Input:
//	- name: Relu_673_0	type: float	shape: Shape{32, 160, 17, 17}
//	- name: Reshape_677_0	type: float	shape: Shape{160, 160, 7, 1}
// Output:
//	- name: Convolution_678_0	type: float	shape: Shape{32, 160, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_678(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 160, 160, 7, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 3, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_628
// Description:	Convolution
// Input:
//	- name: AvgPool_623_0	type: float	shape: Shape{32, 768, 17, 17}
//	- name: Reshape_627_0	type: float	shape: Shape{192, 768, 1, 1}
// Output:
//	- name: Convolution_628_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_628(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 768, 1, 1));
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
// Node name:	Reshape_519
// Description:	Reshape
// Input:
//	- name: Constant_57_0	type: float	shape: Shape{1, 1, 192, 32}
// Output:
//	- name: Reshape_519_0	type: float	shape: Shape{32, 192, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_519(float* input0, float* output0)
{
    uint32_t input_strides0 = 32;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 192;
    size_t nx = 32;
    size_t ny = 192;
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
extern void Reshape_float_float_cuda_Reshape_519_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_519<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_806
// Description:	Reshape
// Input:
//	- name: Constant_385_0	type: float	shape: Shape{3, 3, 192, 192}
// Output:
//	- name: Reshape_806_0	type: float	shape: Shape{192, 192, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_806(float* input0, float* output0)
{
    uint32_t input_strides0 = 36864;
    uint32_t input_strides1 = 192;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 1728;
    size_t nx = 192;
    size_t ny = 192;
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
extern void Reshape_float_float_cuda_Reshape_806_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_806<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_854
// Description:	Convolution
// Input:
//	- name: Concat_848_0	type: float	shape: Shape{32, 2048, 8, 8}
//	- name: Reshape_853_0	type: float	shape: Shape{448, 2048, 1, 1}
// Output:
//	- name: Convolution_854_0	type: float	shape: Shape{32, 448, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_854(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 448, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 448, 2048, 1, 1));
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
// Node name:	Reshape_504
// Description:	Reshape
// Input:
//	- name: Constant_22_0	type: float	shape: Shape{3, 3, 80, 192}
// Output:
//	- name: Reshape_504_0	type: float	shape: Shape{192, 80, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_504(float* input0, float* output0)
{
    uint32_t input_strides0 = 15360;
    uint32_t input_strides1 = 192;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 720;
    size_t nx = 192;
    size_t ny = 80;
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
extern void Reshape_float_float_cuda_Reshape_504_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_504<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_512
// Description:	Convolution
// Input:
//	- name: MaxPool_508_0	type: float	shape: Shape{32, 192, 35, 35}
//	- name: Reshape_511_0	type: float	shape: Shape{48, 192, 1, 1}
// Output:
//	- name: Convolution_512_0	type: float	shape: Shape{32, 48, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_512(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 48, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 48, 192, 1, 1));
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
// Node name:	Convolution_520
// Description:	Convolution
// Input:
//	- name: AvgPool_515_0	type: float	shape: Shape{32, 192, 35, 35}
//	- name: Reshape_519_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_520_0	type: float	shape: Shape{32, 32, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_520(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 32, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 192, 1, 1));
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
// Node name:	Convolution_771
// Description:	Convolution
// Input:
//	- name: Relu_767_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Reshape_770_0	type: float	shape: Shape{192, 192, 1, 7}
// Output:
//	- name: Convolution_771_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_771(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 192, 1, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	AvgPool_887
// Description:	AvgPool
// Input:
//	- name: Concat_886_0	type: float	shape: Shape{32, 2048, 8, 8}
// Output:
//	- name: AvgPool_887_0	type: float	shape: Shape{32, 2048, 1, 1}
void AvgPool_float_float_cuda_lib_AvgPool_887(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 8, 8));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 2048, 1, 1));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,8, 8, 0, 0, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Convolution_762
// Description:	Convolution
// Input:
//	- name: Relu_757_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Reshape_761_0	type: float	shape: Shape{192, 192, 7, 1}
// Output:
//	- name: Convolution_762_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_762(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 192, 7, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 3, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_832
// Description:	Convolution
// Input:
//	- name: Relu_825_0	type: float	shape: Shape{32, 448, 8, 8}
//	- name: Reshape_831_0	type: float	shape: Shape{384, 448, 3, 3}
// Output:
//	- name: Convolution_832_0	type: float	shape: Shape{32, 384, 8, 8}
void Convolution_float_float_float_cuda_lib_Convolution_832(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 448, 8, 8));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 384, 8, 8));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 384, 448, 3, 3));
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
// Node name:	Convolution_528
// Description:	Convolution
// Input:
//	- name: Relu_523_0	type: float	shape: Shape{32, 64, 35, 35}
//	- name: Reshape_527_0	type: float	shape: Shape{96, 64, 3, 3}
// Output:
//	- name: Convolution_528_0	type: float	shape: Shape{32, 96, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_528(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 96, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 96, 64, 3, 3));
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
// Node name:	BatchNormInference_798
// Description:	BatchNormInference
// Input:
//	- name: Constant_366_0	type: float	shape: Shape{320}
//	- name: Constant_367_0	type: float	shape: Shape{320}
//	- name: Convolution_795_0	type: float	shape: Shape{32, 320, 8, 8}
//	- name: Constant_368_0	type: float	shape: Shape{320}
//	- name: Constant_369_0	type: float	shape: Shape{320}
// Output:
//	- name: BatchNormInference_798_0	type: float	shape: Shape{32, 320, 8, 8}
extern "C" __launch_bounds__(64) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 8 * 8;
    const int c_id = blockIdx.x % 320;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 8 * 8; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Concat_616
// Description:	Concat
// Input:
//	- name: Relu_606_0	type: float	shape: Shape{32, 384, 17, 17}
//	- name: Relu_615_0	type: float	shape: Shape{32, 96, 17, 17}
//	- name: MaxPool_603_0	type: float	shape: Shape{32, 288, 17, 17}
// Output:
//	- name: Concat_616_0	type: float	shape: Shape{32, 768, 17, 17}
extern "C" __launch_bounds__(512) __global__ void Concat_float_float_float_float_cuda_Concat_616(float* input0, float* input1, float* input2, float* output0)
{
    uint32_t inputs_strides[] = {110976, 27744, 83232};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 7102464)
    {
        uint32_t block_id = tid / 221952;
        uint32_t block_idx = tid % 221952;
        uint32_t output_idx = block_id * 221952 + block_idx;
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
        if(block_idx < inputs_strides[2])
        {
            output0[output_idx] = input2[block_id * inputs_strides[2] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2];
    }

}
extern void Concat_float_float_float_float_cuda_Concat_616_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    Concat_float_float_float_float_cuda_Concat_616<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Convolution_496
// Description:	Convolution
// Input:
//	- name: Relu_494_0	type: float	shape: Shape{32, 32, 147, 147}
//	- name: Reshape_495_0	type: float	shape: Shape{64, 32, 3, 3}
// Output:
//	- name: Convolution_496_0	type: float	shape: Shape{32, 64, 147, 147}
void Convolution_float_float_float_cuda_lib_Convolution_496(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 32, 147, 147));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 147, 147));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
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
// Node name:	Reshape_549
// Description:	Reshape
// Input:
//	- name: Constant_93_0	type: float	shape: Shape{1, 1, 256, 64}
// Output:
//	- name: Reshape_549_0	type: float	shape: Shape{64, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_549(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_549_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_549<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_524
// Description:	BatchNormInference
// Input:
//	- name: Constant_58_0	type: float	shape: Shape{32}
//	- name: Constant_59_0	type: float	shape: Shape{32}
//	- name: Convolution_520_0	type: float	shape: Shape{32, 32, 35, 35}
//	- name: Constant_60_0	type: float	shape: Shape{32}
//	- name: Constant_61_0	type: float	shape: Shape{32}
// Output:
//	- name: BatchNormInference_524_0	type: float	shape: Shape{32, 32, 35, 35}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_524(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 35 * 35;
    const int c_id = blockIdx.x % 32;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 35 * 35; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_524_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_524<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Convolution_550
// Description:	Convolution
// Input:
//	- name: AvgPool_545_0	type: float	shape: Shape{32, 256, 35, 35}
//	- name: Reshape_549_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_550_0	type: float	shape: Shape{32, 64, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_550(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 35, 35));
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
// Node name:	Reshape_541
// Description:	Reshape
// Input:
//	- name: Constant_68_0	type: float	shape: Shape{1, 1, 256, 48}
// Output:
//	- name: Reshape_541_0	type: float	shape: Shape{48, 256, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_541(float* input0, float* output0)
{
    uint32_t input_strides0 = 48;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 256;
    size_t nx = 48;
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
extern void Reshape_float_float_cuda_Reshape_541_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_541<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_542
// Description:	Convolution
// Input:
//	- name: Concat_538_0	type: float	shape: Shape{32, 256, 35, 35}
//	- name: Reshape_541_0	type: float	shape: Shape{48, 256, 1, 1}
// Output:
//	- name: Convolution_542_0	type: float	shape: Shape{32, 48, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_542(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 48, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 48, 256, 1, 1));
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
// Node name:	Reshape_821
// Description:	Reshape
// Input:
//	- name: Constant_431_0	type: float	shape: Shape{1, 1, 1280, 192}
// Output:
//	- name: Reshape_821_0	type: float	shape: Shape{192, 1280, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_821(float* input0, float* output0)
{
    uint32_t input_strides0 = 192;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1280;
    size_t nx = 192;
    size_t ny = 1280;
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
extern void Reshape_float_float_cuda_Reshape_821_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_821<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_501
// Description:	Convolution
// Input:
//	- name: MaxPool_499_0	type: float	shape: Shape{32, 64, 73, 73}
//	- name: Reshape_500_0	type: float	shape: Shape{80, 64, 1, 1}
// Output:
//	- name: Convolution_501_0	type: float	shape: Shape{32, 80, 73, 73}
void Convolution_float_float_float_cuda_lib_Convolution_501(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 73, 73));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 80, 73, 73));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 80, 64, 1, 1));
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
// Node name:	Convolution_535
// Description:	Convolution
// Input:
//	- name: Relu_533_0	type: float	shape: Shape{32, 96, 35, 35}
//	- name: Reshape_534_0	type: float	shape: Shape{96, 96, 3, 3}
// Output:
//	- name: Convolution_535_0	type: float	shape: Shape{32, 96, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_535(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 96, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 96, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 96, 96, 3, 3));
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
// Node name:	Reshape_579
// Description:	Reshape
// Input:
//	- name: Constant_129_0	type: float	shape: Shape{1, 1, 288, 64}
// Output:
//	- name: Reshape_579_0	type: float	shape: Shape{64, 288, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_579(float* input0, float* output0)
{
    uint32_t input_strides0 = 64;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 288;
    size_t nx = 64;
    size_t ny = 288;
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
extern void Reshape_float_float_cuda_Reshape_579_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_579<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_572
// Description:	Convolution
// Input:
//	- name: Concat_568_0	type: float	shape: Shape{32, 288, 35, 35}
//	- name: Reshape_571_0	type: float	shape: Shape{48, 288, 1, 1}
// Output:
//	- name: Convolution_572_0	type: float	shape: Shape{32, 48, 35, 35}
void Convolution_float_float_float_cuda_lib_Convolution_572(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 288, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 48, 35, 35));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 48, 288, 1, 1));
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
// Node name:	BatchNormInference_614
// Description:	BatchNormInference
// Input:
//	- name: Constant_151_0	type: float	shape: Shape{96}
//	- name: Constant_152_0	type: float	shape: Shape{96}
//	- name: Convolution_613_0	type: float	shape: Shape{32, 96, 17, 17}
//	- name: Constant_153_0	type: float	shape: Shape{96}
//	- name: Constant_154_0	type: float	shape: Shape{96}
// Output:
//	- name: BatchNormInference_614_0	type: float	shape: Shape{32, 96, 17, 17}
extern "C" __launch_bounds__(289) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_614(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 17 * 17;
    const int c_id = blockIdx.x % 96;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 17 * 17; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_614_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_614<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_599
// Description:	Reshape
// Input:
//	- name: Constant_135_0	type: float	shape: Shape{3, 3, 288, 384}
// Output:
//	- name: Reshape_599_0	type: float	shape: Shape{384, 288, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_599(float* input0, float* output0)
{
    uint32_t input_strides0 = 110592;
    uint32_t input_strides1 = 384;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 2592;
    size_t nx = 384;
    size_t ny = 288;
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
extern void Reshape_float_float_cuda_Reshape_599_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_599<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_761
// Description:	Reshape
// Input:
//	- name: Constant_334_0	type: float	shape: Shape{7, 1, 192, 192}
// Output:
//	- name: Reshape_761_0	type: float	shape: Shape{192, 192, 7, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_761(float* input0, float* output0)
{
    uint32_t input_strides0 = 36864;
    uint32_t input_strides1 = 192;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 7;
    uint32_t trans_strides2 = 1344;
    size_t nx = 192;
    size_t ny = 192;
    size_t nz = 7;
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
extern void Reshape_float_float_cuda_Reshape_761_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_761<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_813
// Description:	Reshape
// Input:
//	- name: Constant_396_0	type: float	shape: Shape{1, 1, 1280, 384}
// Output:
//	- name: Reshape_813_0	type: float	shape: Shape{384, 1280, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_813(float* input0, float* output0)
{
    uint32_t input_strides0 = 384;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 1280;
    size_t nx = 384;
    size_t ny = 1280;
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
extern void Reshape_float_float_cuda_Reshape_813_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_813<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_613
// Description:	Convolution
// Input:
//	- name: Relu_611_0	type: float	shape: Shape{32, 96, 35, 35}
//	- name: Reshape_612_0	type: float	shape: Shape{96, 96, 3, 3}
// Output:
//	- name: Convolution_613_0	type: float	shape: Shape{32, 96, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_613(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 96, 35, 35));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 96, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 96, 96, 3, 3));
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
// Node name:	BatchNormInference_836
// Description:	BatchNormInference
// Input:
//	- name: Constant_417_0	type: float	shape: Shape{384}
//	- name: Constant_418_0	type: float	shape: Shape{384}
//	- name: Convolution_832_0	type: float	shape: Shape{32, 384, 8, 8}
//	- name: Constant_419_0	type: float	shape: Shape{384}
//	- name: Constant_420_0	type: float	shape: Shape{384}
// Output:
//	- name: BatchNormInference_836_0	type: float	shape: Shape{32, 384, 8, 8}
extern "C" __launch_bounds__(64) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 8 * 8;
    const int c_id = blockIdx.x % 384;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 8 * 8; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	AvgPool_623
// Description:	AvgPool
// Input:
//	- name: Concat_616_0	type: float	shape: Shape{32, 768, 17, 17}
// Output:
//	- name: AvgPool_623_0	type: float	shape: Shape{32, 768, 17, 17}
void AvgPool_float_float_cuda_lib_AvgPool_623(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Broadcast_890
// Description:	Broadcast
// Input:
//	- name: Constant_485_0	type: float	shape: Shape{1001}
// Output:
//	- name: Broadcast_890_0	type: float	shape: Shape{32, 1001}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_890(float* input0, float* output0)
{
    size_t nthreads = 32032;uint32_t strides0 = 1001;
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
extern void Broadcast_float_float_cuda_Broadcast_890_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_890<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_627
// Description:	Reshape
// Input:
//	- name: Constant_201_0	type: float	shape: Shape{1, 1, 768, 192}
// Output:
//	- name: Reshape_627_0	type: float	shape: Shape{192, 768, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_627(float* input0, float* output0)
{
    uint32_t input_strides0 = 192;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 768;
    size_t nx = 192;
    size_t ny = 768;
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
extern void Reshape_float_float_cuda_Reshape_627_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_627<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_632
// Description:	BatchNormInference
// Input:
//	- name: Constant_202_0	type: float	shape: Shape{192}
//	- name: Constant_203_0	type: float	shape: Shape{192}
//	- name: Convolution_628_0	type: float	shape: Shape{32, 192, 17, 17}
//	- name: Constant_204_0	type: float	shape: Shape{192}
//	- name: Constant_205_0	type: float	shape: Shape{192}
// Output:
//	- name: BatchNormInference_632_0	type: float	shape: Shape{32, 192, 17, 17}
extern "C" __launch_bounds__(289) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 17 * 17;
    const int c_id = blockIdx.x % 192;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 17 * 17; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_842
// Description:	Reshape
// Input:
//	- name: Constant_426_0	type: float	shape: Shape{3, 1, 384, 384}
// Output:
//	- name: Reshape_842_0	type: float	shape: Shape{384, 384, 3, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_842(float* input0, float* output0)
{
    uint32_t input_strides0 = 147456;
    uint32_t input_strides1 = 384;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 3;
    uint32_t trans_strides2 = 1152;
    size_t nx = 384;
    size_t ny = 384;
    size_t nz = 3;
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
extern void Reshape_float_float_cuda_Reshape_842_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_842<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_571
// Description:	Reshape
// Input:
//	- name: Constant_104_0	type: float	shape: Shape{1, 1, 288, 48}
// Output:
//	- name: Reshape_571_0	type: float	shape: Shape{48, 288, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_571(float* input0, float* output0)
{
    uint32_t input_strides0 = 48;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 288;
    size_t nx = 48;
    size_t ny = 288;
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
extern void Reshape_float_float_cuda_Reshape_571_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_571<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_663
// Description:	Reshape
// Input:
//	- name: Constant_227_0	type: float	shape: Shape{1, 1, 768, 160}
// Output:
//	- name: Reshape_663_0	type: float	shape: Shape{160, 768, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_663(float* input0, float* output0)
{
    uint32_t input_strides0 = 160;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 768;
    size_t nx = 160;
    size_t ny = 768;
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
extern void Reshape_float_float_cuda_Reshape_663_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_663<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_487
// Description:	Reshape
// Input:
//	- name: Constant_2_0	type: float	shape: Shape{3, 3, 3, 32}
// Output:
//	- name: Reshape_487_0	type: float	shape: Shape{32, 3, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_487(float* input0, float* output0)
{
    uint32_t input_strides0 = 96;
    uint32_t input_strides1 = 32;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 27;
    size_t nx = 32;
    size_t ny = 3;
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
extern void Reshape_float_float_cuda_Reshape_487_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_487<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_621
// Description:	Reshape
// Input:
//	- name: Constant_176_0	type: float	shape: Shape{1, 1, 768, 128}
// Output:
//	- name: Reshape_621_0	type: float	shape: Shape{128, 768, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_621(float* input0, float* output0)
{
    uint32_t input_strides0 = 128;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 768;
    size_t nx = 128;
    size_t ny = 768;
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
extern void Reshape_float_float_cuda_Reshape_621_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_621<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_622
// Description:	Convolution
// Input:
//	- name: Concat_616_0	type: float	shape: Shape{32, 768, 17, 17}
//	- name: Reshape_621_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_622_0	type: float	shape: Shape{32, 128, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_622(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 768, 1, 1));
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
// Node name:	Reshape_635
// Description:	Reshape
// Input:
//	- name: Constant_181_0	type: float	shape: Shape{7, 1, 128, 128}
// Output:
//	- name: Reshape_635_0	type: float	shape: Shape{128, 128, 7, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_635(float* input0, float* output0)
{
    uint32_t input_strides0 = 16384;
    uint32_t input_strides1 = 128;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 7;
    uint32_t trans_strides2 = 896;
    size_t nx = 128;
    size_t ny = 128;
    size_t nz = 7;
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
extern void Reshape_float_float_cuda_Reshape_635_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_635<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_636
// Description:	Convolution
// Input:
//	- name: Relu_631_0	type: float	shape: Shape{32, 128, 17, 17}
//	- name: Reshape_635_0	type: float	shape: Shape{128, 128, 7, 1}
// Output:
//	- name: Convolution_636_0	type: float	shape: Shape{32, 128, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_636(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 7, 1));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 3, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_655
// Description:	Convolution
// Input:
//	- name: Relu_653_0	type: float	shape: Shape{32, 128, 17, 17}
//	- name: Reshape_654_0	type: float	shape: Shape{192, 128, 1, 7}
// Output:
//	- name: Convolution_655_0	type: float	shape: Shape{32, 192, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_655(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 192, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 192, 128, 1, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_645
// Description:	Convolution
// Input:
//	- name: Relu_641_0	type: float	shape: Shape{32, 128, 17, 17}
//	- name: Reshape_644_0	type: float	shape: Shape{128, 128, 1, 7}
// Output:
//	- name: Convolution_645_0	type: float	shape: Shape{32, 128, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_645(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 1, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	BatchNormInference_489
// Description:	BatchNormInference
// Input:
//	- name: Constant_3_0	type: float	shape: Shape{32}
//	- name: Constant_4_0	type: float	shape: Shape{32}
//	- name: Convolution_488_0	type: float	shape: Shape{32, 32, 149, 149}
//	- name: Constant_5_0	type: float	shape: Shape{32}
//	- name: Constant_6_0	type: float	shape: Shape{32}
// Output:
//	- name: BatchNormInference_489_0	type: float	shape: Shape{32, 32, 149, 149}
extern "C" __launch_bounds__(512) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_489(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 149 * 149;
    const int c_id = blockIdx.x % 32;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 149 * 149; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_489_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_489<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Relu_490
// Description:	Relu
// Input:
//	- name: BatchNormInference_489_0	type: float	shape: Shape{32, 32, 149, 149}
// Output:
//	- name: Relu_490_0	type: float	shape: Shape{32, 32, 149, 149}
extern "C" __launch_bounds__(512) __global__ void Relu_float_float_cuda_Relu_490(float* input0, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern void Relu_float_float_cuda_Relu_490_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Relu_float_float_cuda_Relu_490<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_509
// Description:	Reshape
// Input:
//	- name: Constant_27_0	type: float	shape: Shape{1, 1, 192, 64}
// Output:
//	- name: Reshape_509_0	type: float	shape: Shape{64, 192, 1, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_509(float* input0, float* output0)
{
    uint32_t input_strides0 = 64;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 192;
    size_t nx = 64;
    size_t ny = 192;
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
extern void Reshape_float_float_cuda_Reshape_509_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_509<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	BatchNormInference_668
// Description:	BatchNormInference
// Input:
//	- name: Constant_228_0	type: float	shape: Shape{160}
//	- name: Constant_229_0	type: float	shape: Shape{160}
//	- name: Convolution_664_0	type: float	shape: Shape{32, 160, 17, 17}
//	- name: Constant_230_0	type: float	shape: Shape{160}
//	- name: Constant_231_0	type: float	shape: Shape{160}
// Output:
//	- name: BatchNormInference_668_0	type: float	shape: Shape{32, 160, 17, 17}
extern "C" __launch_bounds__(289) __global__ void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0)
{
    const int st = blockIdx.x * 17 * 17;
    const int c_id = blockIdx.x % 160;
    #pragma unroll 1
    for (int i = threadIdx.x; i < 17 * 17; i += blockDim.x)
    {
        output0[st + i] = (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) / sqrtf(0.001 + input4[c_id])));
    }

}
extern void BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0);
}
// Node name:	Reshape_677
// Description:	Reshape
// Input:
//	- name: Constant_232_0	type: float	shape: Shape{7, 1, 160, 160}
// Output:
//	- name: Reshape_677_0	type: float	shape: Shape{160, 160, 7, 1}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_677(float* input0, float* output0)
{
    uint32_t input_strides0 = 25600;
    uint32_t input_strides1 = 160;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 7;
    uint32_t trans_strides2 = 1120;
    size_t nx = 160;
    size_t ny = 160;
    size_t nz = 7;
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
extern void Reshape_float_float_cuda_Reshape_677_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_677<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_687
// Description:	Convolution
// Input:
//	- name: Relu_683_0	type: float	shape: Shape{32, 160, 17, 17}
//	- name: Reshape_686_0	type: float	shape: Shape{160, 160, 1, 7}
// Output:
//	- name: Convolution_687_0	type: float	shape: Shape{32, 160, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_687(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 160, 160, 1, 7));
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
// Node name:	Convolution_664
// Description:	Convolution
// Input:
//	- name: Concat_658_0	type: float	shape: Shape{32, 768, 17, 17}
//	- name: Reshape_663_0	type: float	shape: Shape{160, 768, 1, 1}
// Output:
//	- name: Convolution_664_0	type: float	shape: Shape{32, 160, 17, 17}
void Convolution_float_float_float_cuda_lib_Convolution_664(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 768, 17, 17));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 160, 17, 17));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 160, 768, 1, 1));
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

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {32, 299, 299, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {32, 1001}
#endif


extern "C" int kernel_entry(float* Parameter_0_0, float** Result_892_0)
{
// kernel_entry_init
 // name=transpose
Reshape_float_float_cuda_Reshape_486_Call(dim3(1, 5588, 32), dim3(16, 16, 1), 0, 0, Parameter_0_0, Reshape_486_0);
 // name=Reshape_487
Reshape_float_float_cuda_Reshape_487_Call(dim3(2, 3, 1), dim3(16, 1, 16), 0, 0, Constant_2_0, Reshape_487_0);
 // name=cg/conv0/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_488(cudnn_handle_0, Reshape_486_0, Reshape_487_0, Convolution_488_0);
 // name=cg/conv0/batchnorm0/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_489_Call(dim3(1024, 1, 1), dim3(512, 1, 1), 0, 0, Constant_3_0, Constant_4_0, Convolution_488_0, Constant_5_0, Constant_6_0, BatchNormInference_489_0);
 // name=cg/conv0/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(44402, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_489_0, Relu_490_0);
 // name=Reshape_491
Reshape_float_float_cuda_Reshape_491_Call(dim3(2, 32, 1), dim3(16, 1, 16), 0, 0, Constant_7_0, Reshape_491_0);
 // name=cg/conv1/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_492(cudnn_handle_0, Relu_490_0, Reshape_491_0, Convolution_492_0);
 // name=cg/conv1/batchnorm1/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_493_Call(dim3(1024, 1, 1), dim3(512, 1, 1), 0, 0, Constant_8_0, Constant_9_0, Convolution_492_0, Constant_10_0, Constant_11_0, BatchNormInference_493_0);
 // name=cg/conv1/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(43218, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_493_0, Relu_494_0);
 // name=Reshape_495
Reshape_float_float_cuda_Reshape_495_Call(dim3(4, 32, 1), dim3(16, 1, 16), 0, 0, Constant_12_0, Reshape_495_0);
 // name=cg/conv2/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_496(cudnn_handle_0, Relu_494_0, Reshape_495_0, Convolution_496_0);
 // name=cg/conv2/batchnorm2/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_497_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_13_0, Constant_14_0, Convolution_496_0, Constant_15_0, Constant_16_0, BatchNormInference_497_0);
 // name=cg/conv2/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(86436, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_497_0, Relu_498_0);
 // name=cg/mpool0/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_499(cudnn_handle_0, Relu_498_0, MaxPool_499_0);
 // name=Reshape_500
Reshape_float_float_cuda_Reshape_500_Call(dim3(5, 4, 1), dim3(16, 16, 1), 0, 0, Constant_17_0, Reshape_500_0);
 // name=cg/conv3/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_501(cudnn_handle_0, MaxPool_499_0, Reshape_500_0, Convolution_501_0);
 // name=cg/conv3/batchnorm3/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_502_Call(dim3(2560, 1, 1), dim3(512, 1, 1), 0, 0, Constant_18_0, Constant_19_0, Convolution_501_0, Constant_20_0, Constant_21_0, BatchNormInference_502_0);
 // name=cg/conv3/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(26645, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_502_0, Relu_503_0);
 // name=Reshape_504
Reshape_float_float_cuda_Reshape_504_Call(dim3(12, 80, 1), dim3(16, 1, 16), 0, 0, Constant_22_0, Reshape_504_0);
 // name=cg/conv4/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_505(cudnn_handle_0, Relu_503_0, Reshape_504_0, Convolution_505_0);
 // name=cg/conv4/batchnorm4/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_506_Call(dim3(6144, 1, 1), dim3(512, 1, 1), 0, 0, Constant_23_0, Constant_24_0, Convolution_505_0, Constant_25_0, Constant_26_0, BatchNormInference_506_0);
 // name=cg/conv4/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(60492, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_506_0, Relu_507_0);
 // name=cg/mpool1/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_508(cudnn_handle_0, Relu_507_0, MaxPool_508_0);
 // name=Reshape_511
Reshape_float_float_cuda_Reshape_511_Call(dim3(3, 12, 1), dim3(16, 16, 1), 0, 0, Constant_32_0, Reshape_511_0);
 // name=cg/incept_v3_a0/conv6/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_512(cudnn_handle_0, MaxPool_508_0, Reshape_511_0, Convolution_512_0);
 // name=cg/incept_v3_a0/conv6/batchnorm6/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, Constant_33_0, Constant_34_0, Convolution_512_0, Constant_35_0, Constant_36_0, BatchNormInference_517_0);
 // name=cg/incept_v3_a0/conv6/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3675, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_517_0, Relu_522_0);
 // name=Reshape_525
Reshape_float_float_cuda_Reshape_525_Call(dim3(4, 48, 2), dim3(16, 1, 16), 0, 0, Constant_37_0, Reshape_525_0);
 // name=cg/incept_v3_a0/conv7/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_526(cudnn_handle_0, Relu_522_0, Reshape_525_0, Convolution_526_0);
 // name=cg/incept_v3_a0/conv7/batchnorm7/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_38_0, Constant_39_0, Convolution_526_0, Constant_40_0, Constant_41_0, BatchNormInference_530_0);
 // name=cg/incept_v3_a0/conv7/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_530_0, Relu_532_0);
 // name=Reshape_509
Reshape_float_float_cuda_Reshape_509_Call(dim3(4, 12, 1), dim3(16, 16, 1), 0, 0, Constant_27_0, Reshape_509_0);
 // name=cg/incept_v3_a0/conv5/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_510(cudnn_handle_0, MaxPool_508_0, Reshape_509_0, Convolution_510_0);
 // name=cg/incept_v3_a0/conv5/batchnorm5/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_28_0, Constant_29_0, Convolution_510_0, Constant_30_0, Constant_31_0, BatchNormInference_516_0);
 // name=cg/incept_v3_a0/conv5/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_516_0, Relu_521_0);
 // name=cg/incept_v3_a0/apool0/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_515(cudnn_handle_0, MaxPool_508_0, AvgPool_515_0);
 // name=Reshape_519
Reshape_float_float_cuda_Reshape_519_Call(dim3(2, 12, 1), dim3(16, 16, 1), 0, 0, Constant_57_0, Reshape_519_0);
 // name=cg/incept_v3_a0/conv11/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_520(cudnn_handle_0, AvgPool_515_0, Reshape_519_0, Convolution_520_0);
 // name=cg/incept_v3_a0/conv11/batchnorm11/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_524_Call(dim3(1024, 1, 1), dim3(512, 1, 1), 0, 0, Constant_58_0, Constant_59_0, Convolution_520_0, Constant_60_0, Constant_61_0, BatchNormInference_524_0);
 // name=cg/incept_v3_a0/conv11/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2450, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_524_0, Relu_529_0);
 // name=Reshape_513
Reshape_float_float_cuda_Reshape_509_Call(dim3(4, 12, 1), dim3(16, 16, 1), 0, 0, Constant_42_0, Reshape_513_0);
 // name=cg/incept_v3_a0/conv8/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_510(cudnn_handle_0, MaxPool_508_0, Reshape_513_0, Convolution_514_0);
 // name=cg/incept_v3_a0/conv8/batchnorm8/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_43_0, Constant_44_0, Convolution_514_0, Constant_45_0, Constant_46_0, BatchNormInference_518_0);
 // name=cg/incept_v3_a0/conv8/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_518_0, Relu_523_0);
 // name=Reshape_527
Reshape_float_float_cuda_Reshape_527_Call(dim3(6, 64, 1), dim3(16, 1, 16), 0, 0, Constant_47_0, Reshape_527_0);
 // name=cg/incept_v3_a0/conv9/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_528(cudnn_handle_0, Relu_523_0, Reshape_527_0, Convolution_528_0);
 // name=cg/incept_v3_a0/conv9/batchnorm9/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_48_0, Constant_49_0, Convolution_528_0, Constant_50_0, Constant_51_0, BatchNormInference_531_0);
 // name=cg/incept_v3_a0/conv9/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_531_0, Relu_533_0);
 // name=Reshape_534
Reshape_float_float_cuda_Reshape_534_Call(dim3(6, 96, 1), dim3(16, 1, 16), 0, 0, Constant_52_0, Reshape_534_0);
 // name=cg/incept_v3_a0/conv10/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_535(cudnn_handle_0, Relu_533_0, Reshape_534_0, Convolution_535_0);
 // name=cg/incept_v3_a0/conv10/batchnorm10/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_53_0, Constant_54_0, Convolution_535_0, Constant_55_0, Constant_56_0, BatchNormInference_536_0);
 // name=cg/incept_v3_a0/conv10/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_536_0, Relu_537_0);
 // name=cg/incept_v3_a0/concat_0
Concat_float_float_float_float_float_cuda_Concat_538_Call(dim3(19600, 1, 1), dim3(512, 1, 1), 0, 0, Relu_521_0, Relu_532_0, Relu_537_0, Relu_529_0, Concat_538_0);
 // name=cg/incept_v3_a0_1/apool1/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_545(cudnn_handle_0, Concat_538_0, AvgPool_545_0);
 // name=Reshape_549
Reshape_float_float_cuda_Reshape_549_Call(dim3(4, 16, 1), dim3(16, 16, 1), 0, 0, Constant_93_0, Reshape_549_0);
 // name=cg/incept_v3_a0_1/conv18/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_550(cudnn_handle_0, AvgPool_545_0, Reshape_549_0, Convolution_550_0);
 // name=cg/incept_v3_a0_1/conv18/batchnorm18/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_94_0, Constant_95_0, Convolution_550_0, Constant_96_0, Constant_97_0, BatchNormInference_554_0);
 // name=cg/incept_v3_a0_1/conv18/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_554_0, Relu_559_0);
 // name=Reshape_543
Reshape_float_float_cuda_Reshape_549_Call(dim3(4, 16, 1), dim3(16, 16, 1), 0, 0, Constant_78_0, Reshape_543_0);
 // name=cg/incept_v3_a0_1/conv15/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_550(cudnn_handle_0, Concat_538_0, Reshape_543_0, Convolution_544_0);
 // name=cg/incept_v3_a0_1/conv15/batchnorm15/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_79_0, Constant_80_0, Convolution_544_0, Constant_81_0, Constant_82_0, BatchNormInference_548_0);
 // name=cg/incept_v3_a0_1/conv15/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_548_0, Relu_553_0);
 // name=Reshape_557
Reshape_float_float_cuda_Reshape_527_Call(dim3(6, 64, 1), dim3(16, 1, 16), 0, 0, Constant_83_0, Reshape_557_0);
 // name=cg/incept_v3_a0_1/conv16/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_528(cudnn_handle_0, Relu_553_0, Reshape_557_0, Convolution_558_0);
 // name=cg/incept_v3_a0_1/conv16/batchnorm16/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_84_0, Constant_85_0, Convolution_558_0, Constant_86_0, Constant_87_0, BatchNormInference_561_0);
 // name=cg/incept_v3_a0_1/conv16/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_561_0, Relu_563_0);
 // name=Reshape_564
Reshape_float_float_cuda_Reshape_534_Call(dim3(6, 96, 1), dim3(16, 1, 16), 0, 0, Constant_88_0, Reshape_564_0);
 // name=cg/incept_v3_a0_1/conv17/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_535(cudnn_handle_0, Relu_563_0, Reshape_564_0, Convolution_565_0);
 // name=cg/incept_v3_a0_1/conv17/batchnorm17/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_89_0, Constant_90_0, Convolution_565_0, Constant_91_0, Constant_92_0, BatchNormInference_566_0);
 // name=cg/incept_v3_a0_1/conv17/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_566_0, Relu_567_0);
 // name=Reshape_541
Reshape_float_float_cuda_Reshape_541_Call(dim3(3, 16, 1), dim3(16, 16, 1), 0, 0, Constant_68_0, Reshape_541_0);
 // name=cg/incept_v3_a0_1/conv13/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_542(cudnn_handle_0, Concat_538_0, Reshape_541_0, Convolution_542_0);
 // name=cg/incept_v3_a0_1/conv13/batchnorm13/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, Constant_69_0, Constant_70_0, Convolution_542_0, Constant_71_0, Constant_72_0, BatchNormInference_547_0);
 // name=cg/incept_v3_a0_1/conv13/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3675, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_547_0, Relu_552_0);
 // name=Reshape_555
Reshape_float_float_cuda_Reshape_525_Call(dim3(4, 48, 2), dim3(16, 1, 16), 0, 0, Constant_73_0, Reshape_555_0);
 // name=cg/incept_v3_a0_1/conv14/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_526(cudnn_handle_0, Relu_552_0, Reshape_555_0, Convolution_556_0);
 // name=cg/incept_v3_a0_1/conv14/batchnorm14/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_74_0, Constant_75_0, Convolution_556_0, Constant_76_0, Constant_77_0, BatchNormInference_560_0);
 // name=cg/incept_v3_a0_1/conv14/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_560_0, Relu_562_0);
 // name=Reshape_539
Reshape_float_float_cuda_Reshape_549_Call(dim3(4, 16, 1), dim3(16, 16, 1), 0, 0, Constant_63_0, Reshape_539_0);
 // name=cg/incept_v3_a0_1/conv12/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_550(cudnn_handle_0, Concat_538_0, Reshape_539_0, Convolution_540_0);
 // name=cg/incept_v3_a0_1/conv12/batchnorm12/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_64_0, Constant_65_0, Convolution_540_0, Constant_66_0, Constant_67_0, BatchNormInference_546_0);
 // name=cg/incept_v3_a0_1/conv12/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_546_0, Relu_551_0);
 // name=cg/incept_v3_a0_1/concat_0
Concat_float_float_float_float_float_cuda_Concat_568_Call(dim3(22050, 1, 1), dim3(512, 1, 1), 0, 0, Relu_551_0, Relu_562_0, Relu_567_0, Relu_559_0, Concat_568_0);
 // name=cg/incept_v3_a0_2/apool2/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_575(cudnn_handle_0, Concat_568_0, AvgPool_575_0);
 // name=Reshape_579
Reshape_float_float_cuda_Reshape_579_Call(dim3(4, 18, 1), dim3(16, 16, 1), 0, 0, Constant_129_0, Reshape_579_0);
 // name=cg/incept_v3_a0_2/conv25/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_580(cudnn_handle_0, AvgPool_575_0, Reshape_579_0, Convolution_580_0);
 // name=cg/incept_v3_a0_2/conv25/batchnorm25/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_130_0, Constant_131_0, Convolution_580_0, Constant_132_0, Constant_133_0, BatchNormInference_584_0);
 // name=cg/incept_v3_a0_2/conv25/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_584_0, Relu_589_0);
 // name=Reshape_573
Reshape_float_float_cuda_Reshape_579_Call(dim3(4, 18, 1), dim3(16, 16, 1), 0, 0, Constant_114_0, Reshape_573_0);
 // name=cg/incept_v3_a0_2/conv22/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_580(cudnn_handle_0, Concat_568_0, Reshape_573_0, Convolution_574_0);
 // name=cg/incept_v3_a0_2/conv22/batchnorm22/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_115_0, Constant_116_0, Convolution_574_0, Constant_117_0, Constant_118_0, BatchNormInference_578_0);
 // name=cg/incept_v3_a0_2/conv22/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_578_0, Relu_583_0);
 // name=Reshape_587
Reshape_float_float_cuda_Reshape_527_Call(dim3(6, 64, 1), dim3(16, 1, 16), 0, 0, Constant_119_0, Reshape_587_0);
 // name=cg/incept_v3_a0_2/conv23/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_528(cudnn_handle_0, Relu_583_0, Reshape_587_0, Convolution_588_0);
 // name=cg/incept_v3_a0_2/conv23/batchnorm23/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_120_0, Constant_121_0, Convolution_588_0, Constant_122_0, Constant_123_0, BatchNormInference_591_0);
 // name=cg/incept_v3_a0_2/conv23/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_591_0, Relu_593_0);
 // name=Reshape_594
Reshape_float_float_cuda_Reshape_534_Call(dim3(6, 96, 1), dim3(16, 1, 16), 0, 0, Constant_124_0, Reshape_594_0);
 // name=cg/incept_v3_a0_2/conv24/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_535(cudnn_handle_0, Relu_593_0, Reshape_594_0, Convolution_595_0);
 // name=cg/incept_v3_a0_2/conv24/batchnorm24/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_125_0, Constant_126_0, Convolution_595_0, Constant_127_0, Constant_128_0, BatchNormInference_596_0);
 // name=cg/incept_v3_a0_2/conv24/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_596_0, Relu_597_0);
 // name=Reshape_571
Reshape_float_float_cuda_Reshape_571_Call(dim3(3, 18, 1), dim3(16, 16, 1), 0, 0, Constant_104_0, Reshape_571_0);
 // name=cg/incept_v3_a0_2/conv20/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_572(cudnn_handle_0, Concat_568_0, Reshape_571_0, Convolution_572_0);
 // name=cg/incept_v3_a0_2/conv20/batchnorm20/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_517_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, Constant_105_0, Constant_106_0, Convolution_572_0, Constant_107_0, Constant_108_0, BatchNormInference_577_0);
 // name=cg/incept_v3_a0_2/conv20/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3675, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_577_0, Relu_582_0);
 // name=Reshape_585
Reshape_float_float_cuda_Reshape_525_Call(dim3(4, 48, 2), dim3(16, 1, 16), 0, 0, Constant_109_0, Reshape_585_0);
 // name=cg/incept_v3_a0_2/conv21/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_526(cudnn_handle_0, Relu_582_0, Reshape_585_0, Convolution_586_0);
 // name=cg/incept_v3_a0_2/conv21/batchnorm21/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_110_0, Constant_111_0, Convolution_586_0, Constant_112_0, Constant_113_0, BatchNormInference_590_0);
 // name=cg/incept_v3_a0_2/conv21/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_590_0, Relu_592_0);
 // name=Reshape_569
Reshape_float_float_cuda_Reshape_579_Call(dim3(4, 18, 1), dim3(16, 16, 1), 0, 0, Constant_99_0, Reshape_569_0);
 // name=cg/incept_v3_a0_2/conv19/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_580(cudnn_handle_0, Concat_568_0, Reshape_569_0, Convolution_570_0);
 // name=cg/incept_v3_a0_2/conv19/batchnorm19/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_100_0, Constant_101_0, Convolution_570_0, Constant_102_0, Constant_103_0, BatchNormInference_576_0);
 // name=cg/incept_v3_a0_2/conv19/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_576_0, Relu_581_0);
 // name=cg/incept_v3_a0_2/concat_0
Concat_float_float_float_float_float_cuda_Concat_568_Call(dim3(22050, 1, 1), dim3(512, 1, 1), 0, 0, Relu_581_0, Relu_592_0, Relu_597_0, Relu_589_0, Concat_598_0);
 // name=cg/incept_v3_b0/mpool2/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_603(cudnn_handle_0, Concat_598_0, MaxPool_603_0);
 // name=Reshape_601
Reshape_float_float_cuda_Reshape_579_Call(dim3(4, 18, 1), dim3(16, 16, 1), 0, 0, Constant_140_0, Reshape_601_0);
 // name=cg/incept_v3_b0/conv27/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_580(cudnn_handle_0, Concat_598_0, Reshape_601_0, Convolution_602_0);
 // name=cg/incept_v3_b0/conv27/batchnorm27/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_530_Call(dim3(2048, 1, 1), dim3(512, 1, 1), 0, 0, Constant_141_0, Constant_142_0, Convolution_602_0, Constant_143_0, Constant_144_0, BatchNormInference_605_0);
 // name=cg/incept_v3_b0/conv27/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(4900, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_605_0, Relu_607_0);
 // name=Reshape_608
Reshape_float_float_cuda_Reshape_527_Call(dim3(6, 64, 1), dim3(16, 1, 16), 0, 0, Constant_145_0, Reshape_608_0);
 // name=cg/incept_v3_b0/conv28/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_528(cudnn_handle_0, Relu_607_0, Reshape_608_0, Convolution_609_0);
 // name=cg/incept_v3_b0/conv28/batchnorm28/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_531_Call(dim3(3072, 1, 1), dim3(512, 1, 1), 0, 0, Constant_146_0, Constant_147_0, Convolution_609_0, Constant_148_0, Constant_149_0, BatchNormInference_610_0);
 // name=cg/incept_v3_b0/conv28/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(7350, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_610_0, Relu_611_0);
 // name=Reshape_612
Reshape_float_float_cuda_Reshape_534_Call(dim3(6, 96, 1), dim3(16, 1, 16), 0, 0, Constant_150_0, Reshape_612_0);
 // name=cg/incept_v3_b0/conv29/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_613(cudnn_handle_0, Relu_611_0, Reshape_612_0, Convolution_613_0);
 // name=cg/incept_v3_b0/conv29/batchnorm29/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_614_Call(dim3(3072, 1, 1), dim3(289, 1, 1), 0, 0, Constant_151_0, Constant_152_0, Convolution_613_0, Constant_153_0, Constant_154_0, BatchNormInference_614_0);
 // name=cg/incept_v3_b0/conv29/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1734, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_614_0, Relu_615_0);
 // name=Reshape_599
Reshape_float_float_cuda_Reshape_599_Call(dim3(24, 288, 1), dim3(16, 1, 16), 0, 0, Constant_135_0, Reshape_599_0);
 // name=cg/incept_v3_b0/conv26/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_600(cudnn_handle_0, Concat_598_0, Reshape_599_0, Convolution_600_0);
 // name=cg/incept_v3_b0/conv26/batchnorm26/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_604_Call(dim3(12288, 1, 1), dim3(289, 1, 1), 0, 0, Constant_136_0, Constant_137_0, Convolution_600_0, Constant_138_0, Constant_139_0, BatchNormInference_604_0);
 // name=cg/incept_v3_b0/conv26/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(6936, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_604_0, Relu_606_0);
 // name=cg/incept_v3_b0/concat_0
Concat_float_float_float_float_cuda_Concat_616_Call(dim3(13872, 1, 1), dim3(512, 1, 1), 0, 0, Relu_606_0, Relu_615_0, MaxPool_603_0, Concat_616_0);
 // name=cg/incept_v3_c0/apool3/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_623(cudnn_handle_0, Concat_616_0, AvgPool_623_0);
 // name=Reshape_627
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_201_0, Reshape_627_0);
 // name=cg/incept_v3_c0/conv39/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, AvgPool_623_0, Reshape_627_0, Convolution_628_0);
 // name=cg/incept_v3_c0/conv39/batchnorm39/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_202_0, Constant_203_0, Convolution_628_0, Constant_204_0, Constant_205_0, BatchNormInference_632_0);
 // name=cg/incept_v3_c0/conv39/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_632_0, Relu_637_0);
 // name=Reshape_621
Reshape_float_float_cuda_Reshape_621_Call(dim3(8, 48, 1), dim3(16, 16, 1), 0, 0, Constant_176_0, Reshape_621_0);
 // name=cg/incept_v3_c0/conv34/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_622(cudnn_handle_0, Concat_616_0, Reshape_621_0, Convolution_622_0);
 // name=cg/incept_v3_c0/conv34/batchnorm34/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_177_0, Constant_178_0, Convolution_622_0, Constant_179_0, Constant_180_0, BatchNormInference_626_0);
 // name=cg/incept_v3_c0/conv34/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_626_0, Relu_631_0);
 // name=Reshape_635
Reshape_float_float_cuda_Reshape_635_Call(dim3(8, 128, 1), dim3(16, 1, 16), 0, 0, Constant_181_0, Reshape_635_0);
 // name=cg/incept_v3_c0/conv35/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_636(cudnn_handle_0, Relu_631_0, Reshape_635_0, Convolution_636_0);
 // name=cg/incept_v3_c0/conv35/batchnorm35/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_182_0, Constant_183_0, Convolution_636_0, Constant_184_0, Constant_185_0, BatchNormInference_639_0);
 // name=cg/incept_v3_c0/conv35/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_639_0, Relu_641_0);
 // name=Reshape_644
Reshape_float_float_cuda_Reshape_635_Call(dim3(8, 128, 1), dim3(16, 1, 16), 0, 0, Constant_186_0, Reshape_644_0);
 // name=cg/incept_v3_c0/conv36/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_645(cudnn_handle_0, Relu_641_0, Reshape_644_0, Convolution_645_0);
 // name=cg/incept_v3_c0/conv36/batchnorm36/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_187_0, Constant_188_0, Convolution_645_0, Constant_189_0, Constant_190_0, BatchNormInference_647_0);
 // name=cg/incept_v3_c0/conv36/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_647_0, Relu_649_0);
 // name=Reshape_650
Reshape_float_float_cuda_Reshape_635_Call(dim3(8, 128, 1), dim3(16, 1, 16), 0, 0, Constant_191_0, Reshape_650_0);
 // name=cg/incept_v3_c0/conv37/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_636(cudnn_handle_0, Relu_649_0, Reshape_650_0, Convolution_651_0);
 // name=cg/incept_v3_c0/conv37/batchnorm37/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_192_0, Constant_193_0, Convolution_651_0, Constant_194_0, Constant_195_0, BatchNormInference_652_0);
 // name=cg/incept_v3_c0/conv37/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_652_0, Relu_653_0);
 // name=Reshape_654
Reshape_float_float_cuda_Reshape_654_Call(dim3(12, 128, 1), dim3(16, 1, 16), 0, 0, Constant_196_0, Reshape_654_0);
 // name=cg/incept_v3_c0/conv38/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_655(cudnn_handle_0, Relu_653_0, Reshape_654_0, Convolution_655_0);
 // name=cg/incept_v3_c0/conv38/batchnorm38/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_197_0, Constant_198_0, Convolution_655_0, Constant_199_0, Constant_200_0, BatchNormInference_656_0);
 // name=cg/incept_v3_c0/conv38/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_656_0, Relu_657_0);
 // name=Reshape_619
Reshape_float_float_cuda_Reshape_621_Call(dim3(8, 48, 1), dim3(16, 16, 1), 0, 0, Constant_161_0, Reshape_619_0);
 // name=cg/incept_v3_c0/conv31/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_622(cudnn_handle_0, Concat_616_0, Reshape_619_0, Convolution_620_0);
 // name=cg/incept_v3_c0/conv31/batchnorm31/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_162_0, Constant_163_0, Convolution_620_0, Constant_164_0, Constant_165_0, BatchNormInference_625_0);
 // name=cg/incept_v3_c0/conv31/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_625_0, Relu_630_0);
 // name=Reshape_633
Reshape_float_float_cuda_Reshape_635_Call(dim3(8, 128, 1), dim3(16, 1, 16), 0, 0, Constant_166_0, Reshape_633_0);
 // name=cg/incept_v3_c0/conv32/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_645(cudnn_handle_0, Relu_630_0, Reshape_633_0, Convolution_634_0);
 // name=cg/incept_v3_c0/conv32/batchnorm32/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_626_Call(dim3(4096, 1, 1), dim3(289, 1, 1), 0, 0, Constant_167_0, Constant_168_0, Convolution_634_0, Constant_169_0, Constant_170_0, BatchNormInference_638_0);
 // name=cg/incept_v3_c0/conv32/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2312, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_638_0, Relu_640_0);
 // name=Reshape_642
Reshape_float_float_cuda_Reshape_654_Call(dim3(12, 128, 1), dim3(16, 1, 16), 0, 0, Constant_171_0, Reshape_642_0);
 // name=cg/incept_v3_c0/conv33/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_643(cudnn_handle_0, Relu_640_0, Reshape_642_0, Convolution_643_0);
 // name=cg/incept_v3_c0/conv33/batchnorm33/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_172_0, Constant_173_0, Convolution_643_0, Constant_174_0, Constant_175_0, BatchNormInference_646_0);
 // name=cg/incept_v3_c0/conv33/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_646_0, Relu_648_0);
 // name=Reshape_617
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_156_0, Reshape_617_0);
 // name=cg/incept_v3_c0/conv30/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_616_0, Reshape_617_0, Convolution_618_0);
 // name=cg/incept_v3_c0/conv30/batchnorm30/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_157_0, Constant_158_0, Convolution_618_0, Constant_159_0, Constant_160_0, BatchNormInference_624_0);
 // name=cg/incept_v3_c0/conv30/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_624_0, Relu_629_0);
 // name=cg/incept_v3_c0/concat_0
Concat_float_float_float_float_float_cuda_Concat_658_Call(dim3(13872, 1, 1), dim3(512, 1, 1), 0, 0, Relu_629_0, Relu_648_0, Relu_657_0, Relu_637_0, Concat_658_0);
 // name=cg/incept_v3_c0_1/apool4/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_623(cudnn_handle_0, Concat_658_0, AvgPool_665_0);
 // name=Reshape_669
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_252_0, Reshape_669_0);
 // name=cg/incept_v3_c0_1/conv49/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, AvgPool_665_0, Reshape_669_0, Convolution_670_0);
 // name=cg/incept_v3_c0_1/conv49/batchnorm49/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_253_0, Constant_254_0, Convolution_670_0, Constant_255_0, Constant_256_0, BatchNormInference_674_0);
 // name=cg/incept_v3_c0_1/conv49/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_674_0, Relu_679_0);
 // name=Reshape_663
Reshape_float_float_cuda_Reshape_663_Call(dim3(10, 48, 1), dim3(16, 16, 1), 0, 0, Constant_227_0, Reshape_663_0);
 // name=cg/incept_v3_c0_1/conv44/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_664(cudnn_handle_0, Concat_658_0, Reshape_663_0, Convolution_664_0);
 // name=cg/incept_v3_c0_1/conv44/batchnorm44/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_228_0, Constant_229_0, Convolution_664_0, Constant_230_0, Constant_231_0, BatchNormInference_668_0);
 // name=cg/incept_v3_c0_1/conv44/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_668_0, Relu_673_0);
 // name=Reshape_677
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_232_0, Reshape_677_0);
 // name=cg/incept_v3_c0_1/conv45/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_678(cudnn_handle_0, Relu_673_0, Reshape_677_0, Convolution_678_0);
 // name=cg/incept_v3_c0_1/conv45/batchnorm45/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_233_0, Constant_234_0, Convolution_678_0, Constant_235_0, Constant_236_0, BatchNormInference_681_0);
 // name=cg/incept_v3_c0_1/conv45/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_681_0, Relu_683_0);
 // name=Reshape_686
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_237_0, Reshape_686_0);
 // name=cg/incept_v3_c0_1/conv46/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_687(cudnn_handle_0, Relu_683_0, Reshape_686_0, Convolution_687_0);
 // name=cg/incept_v3_c0_1/conv46/batchnorm46/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_238_0, Constant_239_0, Convolution_687_0, Constant_240_0, Constant_241_0, BatchNormInference_689_0);
 // name=cg/incept_v3_c0_1/conv46/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_689_0, Relu_691_0);
 // name=Reshape_692
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_242_0, Reshape_692_0);
 // name=cg/incept_v3_c0_1/conv47/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_678(cudnn_handle_0, Relu_691_0, Reshape_692_0, Convolution_693_0);
 // name=cg/incept_v3_c0_1/conv47/batchnorm47/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_243_0, Constant_244_0, Convolution_693_0, Constant_245_0, Constant_246_0, BatchNormInference_694_0);
 // name=cg/incept_v3_c0_1/conv47/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_694_0, Relu_695_0);
 // name=Reshape_696
Reshape_float_float_cuda_Reshape_696_Call(dim3(12, 160, 1), dim3(16, 1, 16), 0, 0, Constant_247_0, Reshape_696_0);
 // name=cg/incept_v3_c0_1/conv48/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_697(cudnn_handle_0, Relu_695_0, Reshape_696_0, Convolution_697_0);
 // name=cg/incept_v3_c0_1/conv48/batchnorm48/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_248_0, Constant_249_0, Convolution_697_0, Constant_250_0, Constant_251_0, BatchNormInference_698_0);
 // name=cg/incept_v3_c0_1/conv48/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_698_0, Relu_699_0);
 // name=Reshape_661
Reshape_float_float_cuda_Reshape_663_Call(dim3(10, 48, 1), dim3(16, 16, 1), 0, 0, Constant_212_0, Reshape_661_0);
 // name=cg/incept_v3_c0_1/conv41/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_664(cudnn_handle_0, Concat_658_0, Reshape_661_0, Convolution_662_0);
 // name=cg/incept_v3_c0_1/conv41/batchnorm41/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_213_0, Constant_214_0, Convolution_662_0, Constant_215_0, Constant_216_0, BatchNormInference_667_0);
 // name=cg/incept_v3_c0_1/conv41/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_667_0, Relu_672_0);
 // name=Reshape_675
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_217_0, Reshape_675_0);
 // name=cg/incept_v3_c0_1/conv42/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_687(cudnn_handle_0, Relu_672_0, Reshape_675_0, Convolution_676_0);
 // name=cg/incept_v3_c0_1/conv42/batchnorm42/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_218_0, Constant_219_0, Convolution_676_0, Constant_220_0, Constant_221_0, BatchNormInference_680_0);
 // name=cg/incept_v3_c0_1/conv42/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_680_0, Relu_682_0);
 // name=Reshape_684
Reshape_float_float_cuda_Reshape_696_Call(dim3(12, 160, 1), dim3(16, 1, 16), 0, 0, Constant_222_0, Reshape_684_0);
 // name=cg/incept_v3_c0_1/conv43/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_685(cudnn_handle_0, Relu_682_0, Reshape_684_0, Convolution_685_0);
 // name=cg/incept_v3_c0_1/conv43/batchnorm43/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_223_0, Constant_224_0, Convolution_685_0, Constant_225_0, Constant_226_0, BatchNormInference_688_0);
 // name=cg/incept_v3_c0_1/conv43/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_688_0, Relu_690_0);
 // name=Reshape_659
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_207_0, Reshape_659_0);
 // name=cg/incept_v3_c0_1/conv40/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_658_0, Reshape_659_0, Convolution_660_0);
 // name=cg/incept_v3_c0_1/conv40/batchnorm40/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_208_0, Constant_209_0, Convolution_660_0, Constant_210_0, Constant_211_0, BatchNormInference_666_0);
 // name=cg/incept_v3_c0_1/conv40/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_666_0, Relu_671_0);
 // name=cg/incept_v3_c0_1/concat_0
Concat_float_float_float_float_float_cuda_Concat_658_Call(dim3(13872, 1, 1), dim3(512, 1, 1), 0, 0, Relu_671_0, Relu_690_0, Relu_699_0, Relu_679_0, Concat_700_0);
 // name=cg/incept_v3_c0_2/apool5/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_623(cudnn_handle_0, Concat_700_0, AvgPool_707_0);
 // name=Reshape_711
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_303_0, Reshape_711_0);
 // name=cg/incept_v3_c0_2/conv59/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, AvgPool_707_0, Reshape_711_0, Convolution_712_0);
 // name=cg/incept_v3_c0_2/conv59/batchnorm59/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_304_0, Constant_305_0, Convolution_712_0, Constant_306_0, Constant_307_0, BatchNormInference_716_0);
 // name=cg/incept_v3_c0_2/conv59/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_716_0, Relu_721_0);
 // name=Reshape_705
Reshape_float_float_cuda_Reshape_663_Call(dim3(10, 48, 1), dim3(16, 16, 1), 0, 0, Constant_278_0, Reshape_705_0);
 // name=cg/incept_v3_c0_2/conv54/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_664(cudnn_handle_0, Concat_700_0, Reshape_705_0, Convolution_706_0);
 // name=cg/incept_v3_c0_2/conv54/batchnorm54/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_279_0, Constant_280_0, Convolution_706_0, Constant_281_0, Constant_282_0, BatchNormInference_710_0);
 // name=cg/incept_v3_c0_2/conv54/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_710_0, Relu_715_0);
 // name=Reshape_719
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_283_0, Reshape_719_0);
 // name=cg/incept_v3_c0_2/conv55/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_678(cudnn_handle_0, Relu_715_0, Reshape_719_0, Convolution_720_0);
 // name=cg/incept_v3_c0_2/conv55/batchnorm55/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_284_0, Constant_285_0, Convolution_720_0, Constant_286_0, Constant_287_0, BatchNormInference_723_0);
 // name=cg/incept_v3_c0_2/conv55/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_723_0, Relu_725_0);
 // name=Reshape_728
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_288_0, Reshape_728_0);
 // name=cg/incept_v3_c0_2/conv56/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_687(cudnn_handle_0, Relu_725_0, Reshape_728_0, Convolution_729_0);
 // name=cg/incept_v3_c0_2/conv56/batchnorm56/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_289_0, Constant_290_0, Convolution_729_0, Constant_291_0, Constant_292_0, BatchNormInference_731_0);
 // name=cg/incept_v3_c0_2/conv56/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_731_0, Relu_733_0);
 // name=Reshape_734
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_293_0, Reshape_734_0);
 // name=cg/incept_v3_c0_2/conv57/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_678(cudnn_handle_0, Relu_733_0, Reshape_734_0, Convolution_735_0);
 // name=cg/incept_v3_c0_2/conv57/batchnorm57/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_294_0, Constant_295_0, Convolution_735_0, Constant_296_0, Constant_297_0, BatchNormInference_736_0);
 // name=cg/incept_v3_c0_2/conv57/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_736_0, Relu_737_0);
 // name=Reshape_738
Reshape_float_float_cuda_Reshape_696_Call(dim3(12, 160, 1), dim3(16, 1, 16), 0, 0, Constant_298_0, Reshape_738_0);
 // name=cg/incept_v3_c0_2/conv58/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_697(cudnn_handle_0, Relu_737_0, Reshape_738_0, Convolution_739_0);
 // name=cg/incept_v3_c0_2/conv58/batchnorm58/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_299_0, Constant_300_0, Convolution_739_0, Constant_301_0, Constant_302_0, BatchNormInference_740_0);
 // name=cg/incept_v3_c0_2/conv58/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_740_0, Relu_741_0);
 // name=Reshape_703
Reshape_float_float_cuda_Reshape_663_Call(dim3(10, 48, 1), dim3(16, 16, 1), 0, 0, Constant_263_0, Reshape_703_0);
 // name=cg/incept_v3_c0_2/conv51/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_664(cudnn_handle_0, Concat_700_0, Reshape_703_0, Convolution_704_0);
 // name=cg/incept_v3_c0_2/conv51/batchnorm51/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_264_0, Constant_265_0, Convolution_704_0, Constant_266_0, Constant_267_0, BatchNormInference_709_0);
 // name=cg/incept_v3_c0_2/conv51/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_709_0, Relu_714_0);
 // name=Reshape_717
Reshape_float_float_cuda_Reshape_677_Call(dim3(10, 160, 1), dim3(16, 1, 16), 0, 0, Constant_268_0, Reshape_717_0);
 // name=cg/incept_v3_c0_2/conv52/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_687(cudnn_handle_0, Relu_714_0, Reshape_717_0, Convolution_718_0);
 // name=cg/incept_v3_c0_2/conv52/batchnorm52/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_668_Call(dim3(5120, 1, 1), dim3(289, 1, 1), 0, 0, Constant_269_0, Constant_270_0, Convolution_718_0, Constant_271_0, Constant_272_0, BatchNormInference_722_0);
 // name=cg/incept_v3_c0_2/conv52/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(2890, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_722_0, Relu_724_0);
 // name=Reshape_726
Reshape_float_float_cuda_Reshape_696_Call(dim3(12, 160, 1), dim3(16, 1, 16), 0, 0, Constant_273_0, Reshape_726_0);
 // name=cg/incept_v3_c0_2/conv53/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_685(cudnn_handle_0, Relu_724_0, Reshape_726_0, Convolution_727_0);
 // name=cg/incept_v3_c0_2/conv53/batchnorm53/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_274_0, Constant_275_0, Convolution_727_0, Constant_276_0, Constant_277_0, BatchNormInference_730_0);
 // name=cg/incept_v3_c0_2/conv53/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_730_0, Relu_732_0);
 // name=Reshape_701
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_258_0, Reshape_701_0);
 // name=cg/incept_v3_c0_2/conv50/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_700_0, Reshape_701_0, Convolution_702_0);
 // name=cg/incept_v3_c0_2/conv50/batchnorm50/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_259_0, Constant_260_0, Convolution_702_0, Constant_261_0, Constant_262_0, BatchNormInference_708_0);
 // name=cg/incept_v3_c0_2/conv50/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_708_0, Relu_713_0);
 // name=cg/incept_v3_c0_2/concat_0
Concat_float_float_float_float_float_cuda_Concat_658_Call(dim3(13872, 1, 1), dim3(512, 1, 1), 0, 0, Relu_713_0, Relu_732_0, Relu_741_0, Relu_721_0, Concat_742_0);
 // name=cg/incept_v3_c0_3/apool6/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_623(cudnn_handle_0, Concat_742_0, AvgPool_749_0);
 // name=Reshape_753
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_354_0, Reshape_753_0);
 // name=cg/incept_v3_c0_3/conv69/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, AvgPool_749_0, Reshape_753_0, Convolution_754_0);
 // name=cg/incept_v3_c0_3/conv69/batchnorm69/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_355_0, Constant_356_0, Convolution_754_0, Constant_357_0, Constant_358_0, BatchNormInference_758_0);
 // name=cg/incept_v3_c0_3/conv69/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_758_0, Relu_763_0);
 // name=Reshape_747
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_329_0, Reshape_747_0);
 // name=cg/incept_v3_c0_3/conv64/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_742_0, Reshape_747_0, Convolution_748_0);
 // name=cg/incept_v3_c0_3/conv64/batchnorm64/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_330_0, Constant_331_0, Convolution_748_0, Constant_332_0, Constant_333_0, BatchNormInference_752_0);
 // name=cg/incept_v3_c0_3/conv64/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_752_0, Relu_757_0);
 // name=Reshape_761
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_334_0, Reshape_761_0);
 // name=cg/incept_v3_c0_3/conv65/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_762(cudnn_handle_0, Relu_757_0, Reshape_761_0, Convolution_762_0);
 // name=cg/incept_v3_c0_3/conv65/batchnorm65/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_335_0, Constant_336_0, Convolution_762_0, Constant_337_0, Constant_338_0, BatchNormInference_765_0);
 // name=cg/incept_v3_c0_3/conv65/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_765_0, Relu_767_0);
 // name=Reshape_770
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_339_0, Reshape_770_0);
 // name=cg/incept_v3_c0_3/conv66/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_771(cudnn_handle_0, Relu_767_0, Reshape_770_0, Convolution_771_0);
 // name=cg/incept_v3_c0_3/conv66/batchnorm66/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_340_0, Constant_341_0, Convolution_771_0, Constant_342_0, Constant_343_0, BatchNormInference_773_0);
 // name=cg/incept_v3_c0_3/conv66/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_773_0, Relu_775_0);
 // name=Reshape_776
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_344_0, Reshape_776_0);
 // name=cg/incept_v3_c0_3/conv67/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_762(cudnn_handle_0, Relu_775_0, Reshape_776_0, Convolution_777_0);
 // name=cg/incept_v3_c0_3/conv67/batchnorm67/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_345_0, Constant_346_0, Convolution_777_0, Constant_347_0, Constant_348_0, BatchNormInference_778_0);
 // name=cg/incept_v3_c0_3/conv67/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_778_0, Relu_779_0);
 // name=Reshape_780
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_349_0, Reshape_780_0);
 // name=cg/incept_v3_c0_3/conv68/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_771(cudnn_handle_0, Relu_779_0, Reshape_780_0, Convolution_781_0);
 // name=cg/incept_v3_c0_3/conv68/batchnorm68/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_350_0, Constant_351_0, Convolution_781_0, Constant_352_0, Constant_353_0, BatchNormInference_782_0);
 // name=cg/incept_v3_c0_3/conv68/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_782_0, Relu_783_0);
 // name=Reshape_745
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_314_0, Reshape_745_0);
 // name=cg/incept_v3_c0_3/conv61/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_742_0, Reshape_745_0, Convolution_746_0);
 // name=cg/incept_v3_c0_3/conv61/batchnorm61/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_315_0, Constant_316_0, Convolution_746_0, Constant_317_0, Constant_318_0, BatchNormInference_751_0);
 // name=cg/incept_v3_c0_3/conv61/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_751_0, Relu_756_0);
 // name=Reshape_759
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_319_0, Reshape_759_0);
 // name=cg/incept_v3_c0_3/conv62/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_771(cudnn_handle_0, Relu_756_0, Reshape_759_0, Convolution_760_0);
 // name=cg/incept_v3_c0_3/conv62/batchnorm62/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_320_0, Constant_321_0, Convolution_760_0, Constant_322_0, Constant_323_0, BatchNormInference_764_0);
 // name=cg/incept_v3_c0_3/conv62/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_764_0, Relu_766_0);
 // name=Reshape_768
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_324_0, Reshape_768_0);
 // name=cg/incept_v3_c0_3/conv63/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_762(cudnn_handle_0, Relu_766_0, Reshape_768_0, Convolution_769_0);
 // name=cg/incept_v3_c0_3/conv63/batchnorm63/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_325_0, Constant_326_0, Convolution_769_0, Constant_327_0, Constant_328_0, BatchNormInference_772_0);
 // name=cg/incept_v3_c0_3/conv63/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_772_0, Relu_774_0);
 // name=Reshape_743
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_309_0, Reshape_743_0);
 // name=cg/incept_v3_c0_3/conv60/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_742_0, Reshape_743_0, Convolution_744_0);
 // name=cg/incept_v3_c0_3/conv60/batchnorm60/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_310_0, Constant_311_0, Convolution_744_0, Constant_312_0, Constant_313_0, BatchNormInference_750_0);
 // name=cg/incept_v3_c0_3/conv60/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_750_0, Relu_755_0);
 // name=cg/incept_v3_c0_3/concat_0
Concat_float_float_float_float_float_cuda_Concat_658_Call(dim3(13872, 1, 1), dim3(512, 1, 1), 0, 0, Relu_755_0, Relu_774_0, Relu_783_0, Relu_763_0, Concat_784_0);
 // name=cg/incept_v3_d0/mpool3/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_789(cudnn_handle_0, Concat_784_0, MaxPool_789_0);
 // name=Reshape_787
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_370_0, Reshape_787_0);
 // name=cg/incept_v3_d0/conv72/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_784_0, Reshape_787_0, Convolution_788_0);
 // name=cg/incept_v3_d0/conv72/batchnorm72/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_371_0, Constant_372_0, Convolution_788_0, Constant_373_0, Constant_374_0, BatchNormInference_791_0);
 // name=cg/incept_v3_d0/conv72/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_791_0, Relu_793_0);
 // name=Reshape_796
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_375_0, Reshape_796_0);
 // name=cg/incept_v3_d0/conv73/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_771(cudnn_handle_0, Relu_793_0, Reshape_796_0, Convolution_797_0);
 // name=cg/incept_v3_d0/conv73/batchnorm73/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_376_0, Constant_377_0, Convolution_797_0, Constant_378_0, Constant_379_0, BatchNormInference_799_0);
 // name=cg/incept_v3_d0/conv73/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_799_0, Relu_801_0);
 // name=Reshape_802
Reshape_float_float_cuda_Reshape_761_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_380_0, Reshape_802_0);
 // name=cg/incept_v3_d0/conv74/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_762(cudnn_handle_0, Relu_801_0, Reshape_802_0, Convolution_803_0);
 // name=cg/incept_v3_d0/conv74/batchnorm74/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_381_0, Constant_382_0, Convolution_803_0, Constant_383_0, Constant_384_0, BatchNormInference_804_0);
 // name=cg/incept_v3_d0/conv74/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_804_0, Relu_805_0);
 // name=Reshape_806
Reshape_float_float_cuda_Reshape_806_Call(dim3(12, 192, 1), dim3(16, 1, 16), 0, 0, Constant_385_0, Reshape_806_0);
 // name=cg/incept_v3_d0/conv75/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_807(cudnn_handle_0, Relu_805_0, Reshape_806_0, Convolution_807_0);
 // name=cg/incept_v3_d0/conv75/batchnorm75/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808_Call(dim3(6144, 1, 1), dim3(64, 1, 1), 0, 0, Constant_386_0, Constant_387_0, Convolution_807_0, Constant_388_0, Constant_389_0, BatchNormInference_808_0);
 // name=cg/incept_v3_d0/conv75/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(768, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_808_0, Relu_809_0);
 // name=Reshape_785
Reshape_float_float_cuda_Reshape_627_Call(dim3(12, 48, 1), dim3(16, 16, 1), 0, 0, Constant_360_0, Reshape_785_0);
 // name=cg/incept_v3_d0/conv70/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_628(cudnn_handle_0, Concat_784_0, Reshape_785_0, Convolution_786_0);
 // name=cg/incept_v3_d0/conv70/batchnorm70/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_632_Call(dim3(6144, 1, 1), dim3(289, 1, 1), 0, 0, Constant_361_0, Constant_362_0, Convolution_786_0, Constant_363_0, Constant_364_0, BatchNormInference_790_0);
 // name=cg/incept_v3_d0/conv70/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(3468, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_790_0, Relu_792_0);
 // name=Reshape_794
Reshape_float_float_cuda_Reshape_794_Call(dim3(20, 192, 1), dim3(16, 1, 16), 0, 0, Constant_365_0, Reshape_794_0);
 // name=cg/incept_v3_d0/conv71/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_795(cudnn_handle_0, Relu_792_0, Reshape_794_0, Convolution_795_0);
 // name=cg/incept_v3_d0/conv71/batchnorm71/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798_Call(dim3(10240, 1, 1), dim3(64, 1, 1), 0, 0, Constant_366_0, Constant_367_0, Convolution_795_0, Constant_368_0, Constant_369_0, BatchNormInference_798_0);
 // name=cg/incept_v3_d0/conv71/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1280, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_798_0, Relu_800_0);
 // name=cg/incept_v3_d0/concat_0
Concat_float_float_float_float_cuda_Concat_810_Call(dim3(5120, 1, 1), dim3(512, 1, 1), 0, 0, Relu_800_0, Relu_809_0, MaxPool_789_0, Concat_810_0);
 // name=cg/incept_v3_e0/apool7/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_817(cudnn_handle_0, Concat_810_0, AvgPool_817_0);
 // name=Reshape_821
Reshape_float_float_cuda_Reshape_821_Call(dim3(12, 80, 1), dim3(16, 16, 1), 0, 0, Constant_431_0, Reshape_821_0);
 // name=cg/incept_v3_e0/conv84/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_822(cudnn_handle_0, AvgPool_817_0, Reshape_821_0, Convolution_822_0);
 // name=cg/incept_v3_e0/conv84/batchnorm84/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808_Call(dim3(6144, 1, 1), dim3(64, 1, 1), 0, 0, Constant_432_0, Constant_433_0, Convolution_822_0, Constant_434_0, Constant_435_0, BatchNormInference_826_0);
 // name=cg/incept_v3_e0/conv84/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(768, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_826_0, Relu_833_0);
 // name=Reshape_815
Reshape_float_float_cuda_Reshape_815_Call(dim3(28, 80, 1), dim3(16, 16, 1), 0, 0, Constant_411_0, Reshape_815_0);
 // name=cg/incept_v3_e0/conv80/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_816(cudnn_handle_0, Concat_810_0, Reshape_815_0, Convolution_816_0);
 // name=cg/incept_v3_e0/conv80/batchnorm80/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_820_Call(dim3(14336, 1, 1), dim3(64, 1, 1), 0, 0, Constant_412_0, Constant_413_0, Convolution_816_0, Constant_414_0, Constant_415_0, BatchNormInference_820_0);
 // name=cg/incept_v3_e0/conv80/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1792, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_820_0, Relu_825_0);
 // name=Reshape_831
Reshape_float_float_cuda_Reshape_831_Call(dim3(24, 448, 1), dim3(16, 1, 16), 0, 0, Constant_416_0, Reshape_831_0);
 // name=cg/incept_v3_e0/conv81/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_832(cudnn_handle_0, Relu_825_0, Reshape_831_0, Convolution_832_0);
 // name=cg/incept_v3_e0/conv81/batchnorm81/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_417_0, Constant_418_0, Convolution_832_0, Constant_419_0, Constant_420_0, BatchNormInference_836_0);
 // name=cg/incept_v3_e0/conv81/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_836_0, Relu_839_0);
 // name=Reshape_842
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_426_0, Reshape_842_0);
 // name=cg/incept_v3_e0/conv83/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_843(cudnn_handle_0, Relu_839_0, Reshape_842_0, Convolution_843_0);
 // name=cg/incept_v3_e0/conv83/batchnorm83/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_427_0, Constant_428_0, Convolution_843_0, Constant_429_0, Constant_430_0, BatchNormInference_845_0);
 // name=cg/incept_v3_e0/conv83/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_845_0, Relu_847_0);
 // name=Reshape_840
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_421_0, Reshape_840_0);
 // name=cg/incept_v3_e0/conv82/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_841(cudnn_handle_0, Relu_839_0, Reshape_840_0, Convolution_841_0);
 // name=cg/incept_v3_e0/conv82/batchnorm82/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_422_0, Constant_423_0, Convolution_841_0, Constant_424_0, Constant_425_0, BatchNormInference_844_0);
 // name=cg/incept_v3_e0/conv82/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_844_0, Relu_846_0);
 // name=Reshape_813
Reshape_float_float_cuda_Reshape_813_Call(dim3(24, 80, 1), dim3(16, 16, 1), 0, 0, Constant_396_0, Reshape_813_0);
 // name=cg/incept_v3_e0/conv77/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_814(cudnn_handle_0, Concat_810_0, Reshape_813_0, Convolution_814_0);
 // name=cg/incept_v3_e0/conv77/batchnorm77/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_397_0, Constant_398_0, Convolution_814_0, Constant_399_0, Constant_400_0, BatchNormInference_819_0);
 // name=cg/incept_v3_e0/conv77/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_819_0, Relu_824_0);
 // name=Reshape_829
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_406_0, Reshape_829_0);
 // name=cg/incept_v3_e0/conv79/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_843(cudnn_handle_0, Relu_824_0, Reshape_829_0, Convolution_830_0);
 // name=cg/incept_v3_e0/conv79/batchnorm79/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_407_0, Constant_408_0, Convolution_830_0, Constant_409_0, Constant_410_0, BatchNormInference_835_0);
 // name=cg/incept_v3_e0/conv79/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_835_0, Relu_838_0);
 // name=Reshape_827
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_401_0, Reshape_827_0);
 // name=cg/incept_v3_e0/conv78/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_841(cudnn_handle_0, Relu_824_0, Reshape_827_0, Convolution_828_0);
 // name=cg/incept_v3_e0/conv78/batchnorm78/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_402_0, Constant_403_0, Convolution_828_0, Constant_404_0, Constant_405_0, BatchNormInference_834_0);
 // name=cg/incept_v3_e0/conv78/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_834_0, Relu_837_0);
 // name=Reshape_811
Reshape_float_float_cuda_Reshape_811_Call(dim3(20, 80, 1), dim3(16, 16, 1), 0, 0, Constant_391_0, Reshape_811_0);
 // name=cg/incept_v3_e0/conv76/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_812(cudnn_handle_0, Concat_810_0, Reshape_811_0, Convolution_812_0);
 // name=cg/incept_v3_e0/conv76/batchnorm76/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798_Call(dim3(10240, 1, 1), dim3(64, 1, 1), 0, 0, Constant_392_0, Constant_393_0, Convolution_812_0, Constant_394_0, Constant_395_0, BatchNormInference_818_0);
 // name=cg/incept_v3_e0/conv76/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1280, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_818_0, Relu_823_0);
 // name=cg/incept_v3_e0/concat_0
Concat_float_float_float_float_float_float_float_cuda_Concat_848_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Relu_823_0, Relu_837_0, Relu_838_0, Relu_846_0, Relu_847_0, Relu_833_0, Concat_848_0);
 // name=cg/incept_v3_e0_1/mpool4/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_855(cudnn_handle_0, Concat_848_0, MaxPool_855_0);
 // name=Reshape_859
Reshape_float_float_cuda_Reshape_859_Call(dim3(12, 128, 1), dim3(16, 16, 1), 0, 0, Constant_477_0, Reshape_859_0);
 // name=cg/incept_v3_e0_1/conv93/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_860(cudnn_handle_0, MaxPool_855_0, Reshape_859_0, Convolution_860_0);
 // name=cg/incept_v3_e0_1/conv93/batchnorm93/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_808_Call(dim3(6144, 1, 1), dim3(64, 1, 1), 0, 0, Constant_478_0, Constant_479_0, Convolution_860_0, Constant_480_0, Constant_481_0, BatchNormInference_864_0);
 // name=cg/incept_v3_e0_1/conv93/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(768, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_864_0, Relu_871_0);
 // name=Reshape_853
Reshape_float_float_cuda_Reshape_853_Call(dim3(28, 128, 1), dim3(16, 16, 1), 0, 0, Constant_457_0, Reshape_853_0);
 // name=cg/incept_v3_e0_1/conv89/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_854(cudnn_handle_0, Concat_848_0, Reshape_853_0, Convolution_854_0);
 // name=cg/incept_v3_e0_1/conv89/batchnorm89/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_820_Call(dim3(14336, 1, 1), dim3(64, 1, 1), 0, 0, Constant_458_0, Constant_459_0, Convolution_854_0, Constant_460_0, Constant_461_0, BatchNormInference_858_0);
 // name=cg/incept_v3_e0_1/conv89/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1792, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_858_0, Relu_863_0);
 // name=Reshape_869
Reshape_float_float_cuda_Reshape_831_Call(dim3(24, 448, 1), dim3(16, 1, 16), 0, 0, Constant_462_0, Reshape_869_0);
 // name=cg/incept_v3_e0_1/conv90/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_832(cudnn_handle_0, Relu_863_0, Reshape_869_0, Convolution_870_0);
 // name=cg/incept_v3_e0_1/conv90/batchnorm90/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_463_0, Constant_464_0, Convolution_870_0, Constant_465_0, Constant_466_0, BatchNormInference_874_0);
 // name=cg/incept_v3_e0_1/conv90/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_874_0, Relu_877_0);
 // name=Reshape_880
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_472_0, Reshape_880_0);
 // name=cg/incept_v3_e0_1/conv92/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_843(cudnn_handle_0, Relu_877_0, Reshape_880_0, Convolution_881_0);
 // name=cg/incept_v3_e0_1/conv92/batchnorm92/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_473_0, Constant_474_0, Convolution_881_0, Constant_475_0, Constant_476_0, BatchNormInference_883_0);
 // name=cg/incept_v3_e0_1/conv92/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_883_0, Relu_885_0);
 // name=Reshape_878
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_467_0, Reshape_878_0);
 // name=cg/incept_v3_e0_1/conv91/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_841(cudnn_handle_0, Relu_877_0, Reshape_878_0, Convolution_879_0);
 // name=cg/incept_v3_e0_1/conv91/batchnorm91/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_468_0, Constant_469_0, Convolution_879_0, Constant_470_0, Constant_471_0, BatchNormInference_882_0);
 // name=cg/incept_v3_e0_1/conv91/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_882_0, Relu_884_0);
 // name=Reshape_851
Reshape_float_float_cuda_Reshape_851_Call(dim3(24, 128, 1), dim3(16, 16, 1), 0, 0, Constant_442_0, Reshape_851_0);
 // name=cg/incept_v3_e0_1/conv86/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_852(cudnn_handle_0, Concat_848_0, Reshape_851_0, Convolution_852_0);
 // name=cg/incept_v3_e0_1/conv86/batchnorm86/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_443_0, Constant_444_0, Convolution_852_0, Constant_445_0, Constant_446_0, BatchNormInference_857_0);
 // name=cg/incept_v3_e0_1/conv86/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_857_0, Relu_862_0);
 // name=Reshape_867
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_452_0, Reshape_867_0);
 // name=cg/incept_v3_e0_1/conv88/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_843(cudnn_handle_0, Relu_862_0, Reshape_867_0, Convolution_868_0);
 // name=cg/incept_v3_e0_1/conv88/batchnorm88/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_453_0, Constant_454_0, Convolution_868_0, Constant_455_0, Constant_456_0, BatchNormInference_873_0);
 // name=cg/incept_v3_e0_1/conv88/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_873_0, Relu_876_0);
 // name=Reshape_865
Reshape_float_float_cuda_Reshape_842_Call(dim3(24, 384, 1), dim3(16, 1, 16), 0, 0, Constant_447_0, Reshape_865_0);
 // name=cg/incept_v3_e0_1/conv87/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_841(cudnn_handle_0, Relu_862_0, Reshape_865_0, Convolution_866_0);
 // name=cg/incept_v3_e0_1/conv87/batchnorm87/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_836_Call(dim3(12288, 1, 1), dim3(64, 1, 1), 0, 0, Constant_448_0, Constant_449_0, Convolution_866_0, Constant_450_0, Constant_451_0, BatchNormInference_872_0);
 // name=cg/incept_v3_e0_1/conv87/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1536, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_872_0, Relu_875_0);
 // name=Reshape_849
Reshape_float_float_cuda_Reshape_849_Call(dim3(20, 128, 1), dim3(16, 16, 1), 0, 0, Constant_437_0, Reshape_849_0);
 // name=cg/incept_v3_e0_1/conv85/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_850(cudnn_handle_0, Concat_848_0, Reshape_849_0, Convolution_850_0);
 // name=cg/incept_v3_e0_1/conv85/batchnorm85/FusedBatchNorm
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_798_Call(dim3(10240, 1, 1), dim3(64, 1, 1), 0, 0, Constant_438_0, Constant_439_0, Convolution_850_0, Constant_440_0, Constant_441_0, BatchNormInference_856_0);
 // name=cg/incept_v3_e0_1/conv85/Relu
Relu_float_float_cuda_Relu_490_Call(dim3(1280, 1, 1), dim3(512, 1, 1), 0, 0, BatchNormInference_856_0, Relu_861_0);
 // name=cg/incept_v3_e0_1/concat_0
Concat_float_float_float_float_float_float_float_cuda_Concat_848_Call(dim3(8192, 1, 1), dim3(512, 1, 1), 0, 0, Relu_861_0, Relu_875_0, Relu_876_0, Relu_884_0, Relu_885_0, Relu_871_0, Concat_886_0);
 // name=cg/apool8/AvgPool
AvgPool_float_float_cuda_lib_AvgPool_887(cudnn_handle_0, Concat_886_0, AvgPool_887_0);
 // name=cg/Reshape
// eliminated
 // name=cg/affine0/xw_plus_b/MatMul
Dot_float_float_float_cuda_lib_Dot_889(cublas_handle_0, Reshape_888_0, Constant_484_0, Dot_889_0);
 // name=Broadcast_890
Broadcast_float_float_cuda_Broadcast_890_Call(dim3(501, 1, 1), dim3(64, 1, 1), 0, 0, Constant_485_0, Broadcast_890_0);
 // name=cg/affine0/xw_plus_b
Add_float_float_float_cuda_Add_891_Call(dim3(77, 1, 1), dim3(416, 1, 1), 0, 0, Dot_889_0, Broadcast_890_0, Add_891_0);
 // name=Result_892
Result_float_float_cuda_lib_Result_892(Add_891_0, Result_892_0);
return 0;
}


extern "C" void cuda_free()
{
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));
}

