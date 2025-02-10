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
char* vgg16_group_0_CUDA_GPU0_allocator_memory_pool;
float* vgg16_Reshape_48_0;
float* vgg16_Reshape_49_0;
float* vgg16_Convolution_50_0;
float* vgg16_Broadcast_51_0;
float* vgg16_Relu_54_0;
float* vgg16_Reshape_55_0;
float* vgg16_Convolution_56_0;
float* vgg16_Broadcast_57_0;
float* vgg16_Relu_60_0;
float* vgg16_MaxPool_61_0;
float* vgg16_Reshape_62_0;
float* vgg16_Convolution_63_0;
float* vgg16_Broadcast_64_0;
float* vgg16_Relu_67_0;
float* vgg16_Reshape_68_0;
float* vgg16_Convolution_69_0;
float* vgg16_Broadcast_70_0;
float* vgg16_Relu_73_0;
float* vgg16_MaxPool_74_0;
float* vgg16_Reshape_75_0;
float* vgg16_Convolution_76_0;
float* vgg16_Broadcast_77_0;
float* vgg16_Relu_80_0;
float* vgg16_Reshape_81_0;
float* vgg16_Convolution_82_0;
float* vgg16_Broadcast_83_0;
float* vgg16_Relu_86_0;
float* vgg16_Reshape_87_0;
float* vgg16_Convolution_88_0;
float* vgg16_Broadcast_89_0;
float* vgg16_Relu_92_0;
float* vgg16_MaxPool_93_0;
float* vgg16_Reshape_94_0;
float* vgg16_Convolution_95_0;
float* vgg16_Broadcast_96_0;
float* vgg16_Relu_99_0;
float* vgg16_Reshape_100_0;
float* vgg16_Convolution_101_0;
float* vgg16_Broadcast_102_0;
float* vgg16_Relu_105_0;
float* vgg16_Reshape_106_0;
float* vgg16_Convolution_107_0;
float* vgg16_Broadcast_108_0;
float* vgg16_Relu_111_0;
float* vgg16_MaxPool_112_0;
float* vgg16_Reshape_113_0;
float* vgg16_Convolution_114_0;
float* vgg16_Broadcast_115_0;
float* vgg16_Relu_118_0;
float* vgg16_Reshape_119_0;
float* vgg16_Convolution_120_0;
float* vgg16_Broadcast_121_0;
float* vgg16_Relu_124_0;
float* vgg16_Reshape_125_0;
float* vgg16_Convolution_126_0;
float* vgg16_Broadcast_127_0;
float* vgg16_Relu_130_0;
float* vgg16_MaxPool_131_0;
float* vgg16_Reshape_132_0;
float* vgg16_Dot_133_0;
float* vgg16_Relu_136_0;
float* vgg16_Dot_137_0;
float* vgg16_Relu_140_0;
float* vgg16_Dot_141_0;
float* vgg16_Broadcast_142_0;
float* vgg16_Add_143_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
char* vgg16_group_persist_CUDA_GPU0_allocator_memory_pool;
float* vgg16_Constant_42_0;
float* vgg16_Constant_2_0;
float* vgg16_Constant_3_0;
float* vgg16_Constant_5_0;
float* vgg16_Constant_6_0;
float* vgg16_Constant_8_0;
float* vgg16_Constant_9_0;
float* vgg16_Constant_11_0;
float* vgg16_Constant_12_0;
float* vgg16_Constant_14_0;
float* vgg16_Constant_15_0;
float* vgg16_Constant_17_0;
float* vgg16_Constant_18_0;
float* vgg16_Constant_20_0;
float* vgg16_Constant_21_0;
float* vgg16_Constant_23_0;
float* vgg16_Constant_24_0;
float* vgg16_Constant_26_0;
float* vgg16_Constant_27_0;
float* vgg16_Constant_29_0;
float* vgg16_Constant_30_0;
float* vgg16_Constant_32_0;
float* vgg16_Constant_33_0;
float* vgg16_Constant_35_0;
float* vgg16_Constant_36_0;
float* vgg16_Constant_38_0;
float* vgg16_Constant_39_0;
float* vgg16_Constant_43_0;
float* vgg16_Constant_44_0;
float* vgg16_Constant_45_0;
float* vgg16_Constant_46_0;
float* vgg16_Constant_47_0;
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
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

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
cublasHandle_t vgg16_cublas_handle_0;
cudnnHandle_t vgg16_cudnn_handle_0;
// Node name:	MaxPool_74
// Description:	MaxPool
// Input:
//	- name: vgg16_Relu_73_0	type: float	shape: Shape{32, 128, 112, 112}
// Output:
//	- name: vgg16_MaxPool_74_0	type: float	shape: Shape{32, 128, 56, 56}
void MaxPool_float_float_cuda_lib_MaxPool_74(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 56, 56));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Broadcast_51
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_3_0	type: float	shape: Shape{64}
// Output:
//	- name: vgg16_Broadcast_51_0	type: float	shape: Shape{32, 64, 224, 224}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_51(float* input0, float* output0)
{
    size_t nthreads = 102760448;uint32_t strides0 = 3211264;
    uint32_t strides1 = 50176;
    uint32_t strides2 = 224;
    uint32_t strides3 = 1;
    int stride_magic0 = 1402438301;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 20;
    int stride_shift1 = 14;
    int stride_shift2 = 7;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_51_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_51<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_56
// Description:	Convolution
// Input:
//	- name: vgg16_Relu_54_0	type: float	shape: Shape{32, 64, 224, 224}
//	- name: vgg16_Reshape_55_0	type: float	shape: Shape{64, 64, 3, 3}
// Output:
//	- name: vgg16_Convolution_56_0	type: float	shape: Shape{32, 64, 224, 224}
void Convolution_float_float_float_cuda_lib_Convolution_56(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
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
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_29_0	type: float	shape: Shape{3, 3, 512, 512}
void vgg16_Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_47_0	type: float	shape: Shape{1001}
void vgg16_Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_24_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_38_0	type: float	shape: Shape{3, 3, 512, 512}
void vgg16_Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_26_0	type: float	shape: Shape{3, 3, 512, 512}
void vgg16_Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_26_0 failed.\n");
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
//	- name: vgg16_Constant_39_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_39_0 failed.\n");
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
//	- name: vgg16_Constant_18_0	type: float	shape: Shape{256}
void vgg16_Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_18_0 failed.\n");
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
//	- name: vgg16_Constant_27_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_30_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_36_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_36_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_35_0	type: float	shape: Shape{3, 3, 512, 512}
void vgg16_Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_23_0	type: float	shape: Shape{3, 3, 256, 512}
void vgg16_Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4718592];
    bin_file.read(tmp_mem, 4718592);
    cudaMemcpyAsync(output0, tmp_mem, 4718592, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_33_0	type: float	shape: Shape{512}
void vgg16_Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_33_0 failed.\n");
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
//	- name: vgg16_Constant_17_0	type: float	shape: Shape{3, 3, 256, 256}
void vgg16_Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_8_0	type: float	shape: Shape{3, 3, 64, 128}
void vgg16_Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[294912];
    bin_file.read(tmp_mem, 294912);
    cudaMemcpyAsync(output0, tmp_mem, 294912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_32_0	type: float	shape: Shape{3, 3, 512, 512}
void vgg16_Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_6_0	type: float	shape: Shape{64}
void vgg16_Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_6_0 failed.\n");
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
//	- name: vgg16_Constant_5_0	type: float	shape: Shape{3, 3, 64, 64}
void vgg16_Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[147456];
    bin_file.read(tmp_mem, 147456);
    cudaMemcpyAsync(output0, tmp_mem, 147456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_44_0	type: float	shape: Shape{4096, 4096}
void vgg16_Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[67108864];
    bin_file.read(tmp_mem, 67108864);
    cudaMemcpyAsync(output0, tmp_mem, 67108864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_42_0	type: float	shape: Shape{25088, 4096}
void vgg16_Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[411041792];
    bin_file.read(tmp_mem, 411041792);
    cudaMemcpyAsync(output0, tmp_mem, 411041792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_14_0	type: float	shape: Shape{3, 3, 128, 256}
void vgg16_Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1179648];
    bin_file.read(tmp_mem, 1179648);
    cudaMemcpyAsync(output0, tmp_mem, 1179648, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_45_0	type: float	shape: Shape{4096}
void vgg16_Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_2_0	type: float	shape: Shape{3, 3, 3, 64}
void vgg16_Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6912];
    bin_file.read(tmp_mem, 6912);
    cudaMemcpyAsync(output0, tmp_mem, 6912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_11_0	type: float	shape: Shape{3, 3, 128, 128}
void vgg16_Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[589824];
    bin_file.read(tmp_mem, 589824);
    cudaMemcpyAsync(output0, tmp_mem, 589824, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_43_0	type: float	shape: Shape{4096}
void vgg16_Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_12_0	type: float	shape: Shape{128}
void vgg16_Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_46_0	type: float	shape: Shape{4096, 1001}
void vgg16_Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16400384];
    bin_file.read(tmp_mem, 16400384);
    cudaMemcpyAsync(output0, tmp_mem, 16400384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_21_0	type: float	shape: Shape{256}
void vgg16_Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_21_0 failed.\n");
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
//	- name: vgg16_Constant_9_0	type: float	shape: Shape{128}
void vgg16_Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_15_0	type: float	shape: Shape{256}
void vgg16_Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: vgg16_Constant_3_0	type: float	shape: Shape{64}
void vgg16_Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_3_0 failed.\n");
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
//	- name: vgg16_Constant_20_0	type: float	shape: Shape{3, 3, 256, 256}
void vgg16_Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("../dnn/vgg16/Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load vgg16_Constant_20_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Convolution_69
// Description:	Convolution
// Input:
//	- name: vgg16_Relu_67_0	type: float	shape: Shape{32, 128, 112, 112}
//	- name: vgg16_Reshape_68_0	type: float	shape: Shape{128, 128, 3, 3}
// Output:
//	- name: vgg16_Convolution_69_0	type: float	shape: Shape{32, 128, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_69(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
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
// Node name:	Reshape_100
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_26_0	type: float	shape: Shape{3, 3, 512, 512}
// Output:
//	- name: vgg16_Reshape_100_0	type: float	shape: Shape{512, 512, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_100(float* input0, float* output0)
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
extern void vgg16_Reshape_float_float_cuda_Reshape_100_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_100<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_76
// Description:	Convolution
// Input:
//	- name: vgg16_MaxPool_74_0	type: float	shape: Shape{32, 128, 56, 56}
//	- name: vgg16_Reshape_75_0	type: float	shape: Shape{256, 128, 3, 3}
// Output:
//	- name: vgg16_Convolution_76_0	type: float	shape: Shape{32, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_76(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 128, 3, 3));
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
// Node name:	Reshape_68
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_11_0	type: float	shape: Shape{3, 3, 128, 128}
// Output:
//	- name: vgg16_Reshape_68_0	type: float	shape: Shape{128, 128, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_68(float* input0, float* output0)
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
extern void vgg16_Reshape_float_float_cuda_Reshape_68_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_68<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_62
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_8_0	type: float	shape: Shape{3, 3, 64, 128}
// Output:
//	- name: vgg16_Reshape_62_0	type: float	shape: Shape{128, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_62(float* input0, float* output0)
{
    uint32_t input_strides0 = 8192;
    uint32_t input_strides1 = 128;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 576;
    size_t nx = 128;
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
extern void vgg16_Reshape_float_float_cuda_Reshape_62_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_62<<<grids, blocks, mem, stream>>>(input0, output0);
}

extern "C" void vgg16_cuda_init()
{
// total memory:1805846464

CUDA_SAFE_CALL(cudaMalloc((void**)&vgg16_group_0_CUDA_GPU0_allocator_memory_pool,1252399872));
CUDA_SAFE_CALL(cudaMemset((void*)vgg16_group_0_CUDA_GPU0_allocator_memory_pool, 0, 1252399872));
vgg16_Reshape_48_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_49_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+19267584);
vgg16_Convolution_50_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+19274496);
vgg16_Broadcast_51_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+430316288);
vgg16_Relu_54_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+841358080);
vgg16_Reshape_55_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_56_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+147456);
vgg16_Broadcast_57_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+411189248);
vgg16_Relu_60_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+822231040);
vgg16_MaxPool_61_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_62_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+102760448);
vgg16_Convolution_63_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+103055360);
vgg16_Broadcast_64_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+308576256);
vgg16_Relu_67_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+514097152);
vgg16_Reshape_68_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_69_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+589824);
vgg16_Broadcast_70_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+206110720);
vgg16_Relu_73_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+411631616);
vgg16_MaxPool_74_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_75_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+51380224);
vgg16_Convolution_76_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+52559872);
vgg16_Broadcast_77_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+155320320);
vgg16_Relu_80_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+258080768);
vgg16_Reshape_81_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_82_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+2359296);
vgg16_Broadcast_83_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+105119744);
vgg16_Relu_86_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+207880192);
vgg16_Reshape_87_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_88_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+2359296);
vgg16_Broadcast_89_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+105119744);
vgg16_Relu_92_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+207880192);
vgg16_MaxPool_93_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_94_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+25690112);
vgg16_Convolution_95_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+30408704);
vgg16_Broadcast_96_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+81788928);
vgg16_Relu_99_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+133169152);
vgg16_Reshape_100_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_101_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+9437184);
vgg16_Broadcast_102_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+60817408);
vgg16_Relu_105_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+112197632);
vgg16_Reshape_106_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_107_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+9437184);
vgg16_Broadcast_108_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+60817408);
vgg16_Relu_111_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+112197632);
vgg16_MaxPool_112_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_113_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+12845056);
vgg16_Convolution_114_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+22282240);
vgg16_Broadcast_115_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Relu_118_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+35127296);
vgg16_Reshape_119_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_120_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+9437184);
vgg16_Broadcast_121_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+22282240);
vgg16_Relu_124_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+35127296);
vgg16_Reshape_125_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Convolution_126_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+9437184);
vgg16_Broadcast_127_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+22282240);
vgg16_Relu_130_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+35127296);
vgg16_MaxPool_131_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Reshape_132_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Dot_133_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+3211264);
vgg16_Relu_136_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Dot_137_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+524288);
vgg16_Relu_140_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Dot_141_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+524288);
vgg16_Broadcast_142_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Add_143_0 = (float*)(vgg16_group_0_CUDA_GPU0_allocator_memory_pool+524288);

CUDA_SAFE_CALL(cudaMalloc((void**)&vgg16_group_persist_CUDA_GPU0_allocator_memory_pool,553446592));
CUDA_SAFE_CALL(cudaMemset((void*)vgg16_group_persist_CUDA_GPU0_allocator_memory_pool, 0, 553446592));
vgg16_Constant_42_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+0);
vgg16_Constant_2_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411041792);
vgg16_Constant_3_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411048704);
vgg16_Constant_5_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411048960);
vgg16_Constant_6_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411196416);
vgg16_Constant_8_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411196672);
vgg16_Constant_9_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411491584);
vgg16_Constant_11_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+411492096);
vgg16_Constant_12_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+412081920);
vgg16_Constant_14_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+412082432);
vgg16_Constant_15_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+413262080);
vgg16_Constant_17_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+413263104);
vgg16_Constant_18_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+415622400);
vgg16_Constant_20_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+415623424);
vgg16_Constant_21_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+417982720);
vgg16_Constant_23_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+417983744);
vgg16_Constant_24_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+422702336);
vgg16_Constant_26_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+422704384);
vgg16_Constant_27_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+432141568);
vgg16_Constant_29_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+432143616);
vgg16_Constant_30_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+441580800);
vgg16_Constant_32_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+441582848);
vgg16_Constant_33_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+451020032);
vgg16_Constant_35_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+451022080);
vgg16_Constant_36_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+460459264);
vgg16_Constant_38_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+460461312);
vgg16_Constant_39_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+469898496);
vgg16_Constant_43_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+469900544);
vgg16_Constant_44_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+469916928);
vgg16_Constant_45_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+537025792);
vgg16_Constant_46_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+537042176);
vgg16_Constant_47_0 = (float*)(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool+553442560);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&vgg16_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&vgg16_cudnn_handle_0));
 // name=cg/affine0/weights
vgg16_Constant_float_cuda_Constant_42(0, vgg16_Constant_42_0);
 // name=cg/conv0/conv2d/kernel
vgg16_Constant_float_cuda_Constant_2(0, vgg16_Constant_2_0);
 // name=cg/conv0/biases
vgg16_Constant_float_cuda_Constant_3(0, vgg16_Constant_3_0);
 // name=cg/conv1/conv2d/kernel
vgg16_Constant_float_cuda_Constant_5(0, vgg16_Constant_5_0);
 // name=cg/conv1/biases
vgg16_Constant_float_cuda_Constant_6(0, vgg16_Constant_6_0);
 // name=cg/conv2/conv2d/kernel
vgg16_Constant_float_cuda_Constant_8(0, vgg16_Constant_8_0);
 // name=cg/conv2/biases
vgg16_Constant_float_cuda_Constant_9(0, vgg16_Constant_9_0);
 // name=cg/conv3/conv2d/kernel
vgg16_Constant_float_cuda_Constant_11(0, vgg16_Constant_11_0);
 // name=cg/conv3/biases
vgg16_Constant_float_cuda_Constant_12(0, vgg16_Constant_12_0);
 // name=cg/conv4/conv2d/kernel
vgg16_Constant_float_cuda_Constant_14(0, vgg16_Constant_14_0);
 // name=cg/conv4/biases
vgg16_Constant_float_cuda_Constant_15(0, vgg16_Constant_15_0);
 // name=cg/conv5/conv2d/kernel
vgg16_Constant_float_cuda_Constant_17(0, vgg16_Constant_17_0);
 // name=cg/conv5/biases
vgg16_Constant_float_cuda_Constant_18(0, vgg16_Constant_18_0);
 // name=cg/conv6/conv2d/kernel
vgg16_Constant_float_cuda_Constant_20(0, vgg16_Constant_20_0);
 // name=cg/conv6/biases
vgg16_Constant_float_cuda_Constant_21(0, vgg16_Constant_21_0);
 // name=cg/conv7/conv2d/kernel
vgg16_Constant_float_cuda_Constant_23(0, vgg16_Constant_23_0);
 // name=cg/conv7/biases
vgg16_Constant_float_cuda_Constant_24(0, vgg16_Constant_24_0);
 // name=cg/conv8/conv2d/kernel
vgg16_Constant_float_cuda_Constant_26(0, vgg16_Constant_26_0);
 // name=cg/conv8/biases
vgg16_Constant_float_cuda_Constant_27(0, vgg16_Constant_27_0);
 // name=cg/conv9/conv2d/kernel
vgg16_Constant_float_cuda_Constant_29(0, vgg16_Constant_29_0);
 // name=cg/conv9/biases
vgg16_Constant_float_cuda_Constant_30(0, vgg16_Constant_30_0);
 // name=cg/conv10/conv2d/kernel
vgg16_Constant_float_cuda_Constant_32(0, vgg16_Constant_32_0);
 // name=cg/conv10/biases
vgg16_Constant_float_cuda_Constant_33(0, vgg16_Constant_33_0);
 // name=cg/conv11/conv2d/kernel
vgg16_Constant_float_cuda_Constant_35(0, vgg16_Constant_35_0);
 // name=cg/conv11/biases
vgg16_Constant_float_cuda_Constant_36(0, vgg16_Constant_36_0);
 // name=cg/conv12/conv2d/kernel
vgg16_Constant_float_cuda_Constant_38(0, vgg16_Constant_38_0);
 // name=cg/conv12/biases
vgg16_Constant_float_cuda_Constant_39(0, vgg16_Constant_39_0);
 // name=cg/affine0/biases
vgg16_Constant_float_cuda_Constant_43(0, vgg16_Constant_43_0);
 // name=cg/affine1/weights
vgg16_Constant_float_cuda_Constant_44(0, vgg16_Constant_44_0);
 // name=cg/affine1/biases
vgg16_Constant_float_cuda_Constant_45(0, vgg16_Constant_45_0);
 // name=cg/affine2/weights
vgg16_Constant_float_cuda_Constant_46(0, vgg16_Constant_46_0);
 // name=cg/affine2/biases
vgg16_Constant_float_cuda_Constant_47(0, vgg16_Constant_47_0);
}

// Node name:	Dot_141
// Description:	Dot
// Input:
//	- name: vgg16_Relu_140_0	type: float	shape: Shape{32, 4096}
//	- name: vgg16_Constant_46_0	type: float	shape: Shape{4096, 1001}
// Output:
//	- name: vgg16_Dot_141_0	type: float	shape: Shape{32, 1001}
void Dot_float_float_float_cuda_lib_Dot_141(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 32, 4096, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Add_143
// Description:	Add
// Input:
//	- name: vgg16_Dot_141_0	type: float	shape: Shape{32, 1001}
//	- name: vgg16_Broadcast_142_0	type: float	shape: Shape{32, 1001}
// Output:
//	- name: vgg16_Add_143_0	type: float	shape: Shape{32, 1001}
extern "C" __launch_bounds__(416) __global__ void vgg16_Add_float_float_float_cuda_Add_143(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 416 + threadIdx.x] = add(input0[blockIdx.x * 416 + threadIdx.x], input1[blockIdx.x * 416 + threadIdx.x]);

}
extern void vgg16_Add_float_float_float_cuda_Add_143_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_Add_float_float_float_cuda_Add_143<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_63
// Description:	Convolution
// Input:
//	- name: vgg16_MaxPool_61_0	type: float	shape: Shape{32, 64, 112, 112}
//	- name: vgg16_Reshape_62_0	type: float	shape: Shape{128, 64, 3, 3}
// Output:
//	- name: vgg16_Convolution_63_0	type: float	shape: Shape{32, 128, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_63(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 64, 3, 3));
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
// Node name:	MaxPool_93
// Description:	MaxPool
// Input:
//	- name: vgg16_Relu_92_0	type: float	shape: Shape{32, 256, 56, 56}
// Output:
//	- name: vgg16_MaxPool_93_0	type: float	shape: Shape{32, 256, 28, 28}
void MaxPool_float_float_cuda_lib_MaxPool_93(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 28, 28));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Reshape_81
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_17_0	type: float	shape: Shape{3, 3, 256, 256}
// Output:
//	- name: vgg16_Reshape_81_0	type: float	shape: Shape{256, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_81(float* input0, float* output0)
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
extern void vgg16_Reshape_float_float_cuda_Reshape_81_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_81<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	MaxPool_61
// Description:	MaxPool
// Input:
//	- name: vgg16_Relu_60_0	type: float	shape: Shape{32, 64, 224, 224}
// Output:
//	- name: vgg16_MaxPool_61_0	type: float	shape: Shape{32, 64, 112, 112}
void MaxPool_float_float_cuda_lib_MaxPool_61(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 112, 112));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vgg16_Constant_43_0	type: float	shape: Shape{4096}
//	- name: vgg16_Dot_133_0	type: float	shape: Shape{32, 4096}
// Output:
//	- name: vgg16_Relu_136_0	type: float	shape: Shape{32, 4096}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_134<<<dim3(2048, 1, 1), dim3(64, 1, 1), 0, 0>>>(vgg16_Constant_43_0, Broadcast_134_0);
// Add_float_float_float_cuda_Add_135<<<dim3(256, 1, 1), dim3(512, 1, 1), 0, 0>>>(vgg16_Dot_133_0, Broadcast_134_0, Add_135_0);
// Relu_float_float_cuda_Relu_136<<<dim3(256, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_135_0, vgg16_Relu_136_0);
extern "C" __launch_bounds__(512) __global__ void vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 4096];
    float temp1 = add(input1[tid], temp0);
    float temp2 = relu(temp1);
    output0[tid] = temp2;

}
extern void vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_49
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_2_0	type: float	shape: Shape{3, 3, 3, 64}
// Output:
//	- name: vgg16_Reshape_49_0	type: float	shape: Shape{64, 3, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_49(float* input0, float* output0)
{
    uint32_t input_strides0 = 192;
    uint32_t input_strides1 = 64;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 27;
    size_t nx = 64;
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
extern void vgg16_Reshape_float_float_cuda_Reshape_49_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_49<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Result_144
// Description:	Result
// Input:
//	- name: vgg16_Add_143_0	type: float	shape: Shape{32, 1001}
// Output:
//	- name: Result_144_0	type: float	shape: Shape{32, 1001}
void Result_float_float_cuda_lib_Result_144(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Reshape_48
// Description:	Reshape
// Input:
//	- name: Parameter_0_0	type: float	shape: Shape{32, 224, 224, 3}
// Output:
//	- name: vgg16_Reshape_48_0	type: float	shape: Shape{32, 3, 224, 224}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_48(float* input0, float* output0)
{
    uint32_t input_strides0 = 150528;
    uint32_t input_strides1 = 3;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 150528;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 50176;
    size_t nx = 3;
    size_t ny = 50176;
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
extern void vgg16_Reshape_float_float_cuda_Reshape_48_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_48<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_64
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_9_0	type: float	shape: Shape{128}
// Output:
//	- name: vgg16_Broadcast_64_0	type: float	shape: Shape{32, 128, 112, 112}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_64(float* input0, float* output0)
{
    size_t nthreads = 51380224;uint32_t strides0 = 1605632;
    uint32_t strides1 = 12544;
    uint32_t strides2 = 112;
    uint32_t strides3 = 1;
    int stride_magic0 = 1402438301;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 19;
    int stride_shift1 = 12;
    int stride_shift2 = 6;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_64_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_64<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_55
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_5_0	type: float	shape: Shape{3, 3, 64, 64}
// Output:
//	- name: vgg16_Reshape_55_0	type: float	shape: Shape{64, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_55(float* input0, float* output0)
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
extern void vgg16_Reshape_float_float_cuda_Reshape_55_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_55<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_50
// Description:	Convolution
// Input:
//	- name: vgg16_Reshape_48_0	type: float	shape: Shape{32, 3, 224, 224}
//	- name: vgg16_Reshape_49_0	type: float	shape: Shape{64, 3, 3, 3}
// Output:
//	- name: vgg16_Convolution_50_0	type: float	shape: Shape{32, 64, 224, 224}
void Convolution_float_float_float_cuda_lib_Convolution_50(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 3, 3));
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
// Node name:	Convolution_114
// Description:	Convolution
// Input:
//	- name: vgg16_MaxPool_112_0	type: float	shape: Shape{32, 512, 14, 14}
//	- name: vgg16_Reshape_113_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: vgg16_Convolution_114_0	type: float	shape: Shape{32, 512, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_114(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
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
// Node name:	Broadcast_115
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_33_0	type: float	shape: Shape{512}
// Output:
//	- name: vgg16_Broadcast_115_0	type: float	shape: Shape{32, 512, 14, 14}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_115(float* input0, float* output0)
{
    size_t nthreads = 3211264;uint32_t strides0 = 100352;
    uint32_t strides1 = 196;
    uint32_t strides2 = 14;
    uint32_t strides3 = 1;
    int stride_magic0 = 1402438301;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 15;
    int stride_shift1 = 6;
    int stride_shift2 = 3;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_115_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_115<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_133
// Description:	Dot
// Input:
//	- name: vgg16_Reshape_132_0	type: float	shape: Shape{32, 25088}
//	- name: vgg16_Constant_42_0	type: float	shape: Shape{25088, 4096}
// Output:
//	- name: vgg16_Dot_133_0	type: float	shape: Shape{32, 4096}
void Dot_float_float_float_cuda_lib_Dot_133(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32, 25088, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 25088, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	Broadcast_77
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_15_0	type: float	shape: Shape{256}
// Output:
//	- name: vgg16_Broadcast_77_0	type: float	shape: Shape{32, 256, 56, 56}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_77(float* input0, float* output0)
{
    size_t nthreads = 25690112;uint32_t strides0 = 802816;
    uint32_t strides1 = 3136;
    uint32_t strides2 = 56;
    uint32_t strides3 = 1;
    int stride_magic0 = 1402438301;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 18;
    int stride_shift1 = 10;
    int stride_shift2 = 5;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_77_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_77<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_137
// Description:	Dot
// Input:
//	- name: vgg16_Relu_136_0	type: float	shape: Shape{32, 4096}
//	- name: vgg16_Constant_44_0	type: float	shape: Shape{4096, 4096}
// Output:
//	- name: vgg16_Dot_137_0	type: float	shape: Shape{32, 4096}
void Dot_float_float_float_cuda_lib_Dot_137(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32, 4096, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	Convolution_82
// Description:	Convolution
// Input:
//	- name: vgg16_Relu_80_0	type: float	shape: Shape{32, 256, 56, 56}
//	- name: vgg16_Reshape_81_0	type: float	shape: Shape{256, 256, 3, 3}
// Output:
//	- name: vgg16_Convolution_82_0	type: float	shape: Shape{32, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_82(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
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
// Node name:	MaxPool_112
// Description:	MaxPool
// Input:
//	- name: vgg16_Relu_111_0	type: float	shape: Shape{32, 512, 28, 28}
// Output:
//	- name: vgg16_MaxPool_112_0	type: float	shape: Shape{32, 512, 14, 14}
void MaxPool_float_float_cuda_lib_MaxPool_112(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	MaxPool_131
// Description:	MaxPool
// Input:
//	- name: vgg16_Relu_130_0	type: float	shape: Shape{32, 512, 14, 14}
// Output:
//	- name: vgg16_MaxPool_131_0	type: float	shape: Shape{32, 512, 7, 7}
void MaxPool_float_float_cuda_lib_MaxPool_131(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 7, 7));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	Broadcast_142
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_47_0	type: float	shape: Shape{1001}
// Output:
//	- name: vgg16_Broadcast_142_0	type: float	shape: Shape{32, 1001}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_142(float* input0, float* output0)
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_142_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_142<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_96
// Description:	Broadcast
// Input:
//	- name: vgg16_Constant_24_0	type: float	shape: Shape{512}
// Output:
//	- name: vgg16_Broadcast_96_0	type: float	shape: Shape{32, 512, 28, 28}
extern "C" __launch_bounds__(64) __global__ void vgg16_Broadcast_float_float_cuda_Broadcast_96(float* input0, float* output0)
{
    size_t nthreads = 12845056;uint32_t strides0 = 401408;
    uint32_t strides1 = 784;
    uint32_t strides2 = 28;
    uint32_t strides3 = 1;
    int stride_magic0 = 1402438301;
    int stride_magic1 = 1402438301;
    int stride_magic2 = -1840700269;
    int stride_magic3 = 1;
    int stride_shift0 = 17;
    int stride_shift1 = 8;
    int stride_shift2 = 4;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    uint32_t reduced_strides2 = 0;
    uint32_t reduced_strides3 = 0;
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
extern void vgg16_Broadcast_float_float_cuda_Broadcast_96_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_96<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: vgg16_Convolution_50_0	type: float	shape: Shape{32, 64, 224, 224}
//	- name: vgg16_Broadcast_51_0	type: float	shape: Shape{32, 64, 224, 224}
// Output:
//	- name: vgg16_Relu_54_0	type: float	shape: Shape{32, 64, 224, 224}
// Fused functions:
// Add_float_float_float_cuda_Add_52<<<dim3(200704, 1, 1), dim3(512, 1, 1), 0, 0>>>(vgg16_Convolution_50_0, vgg16_Broadcast_51_0, Add_52_0);
// Reshape_float_float_cuda_lib_Reshape_53(Add_52_0, Reshape_53_0);
// Relu_float_float_cuda_Relu_54<<<dim3(200704, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_53_0, vgg16_Relu_54_0);
extern "C" __launch_bounds__(512) __global__ void vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output0[tid] = temp1;

}
extern void vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Convolution_101
// Description:	Convolution
// Input:
//	- name: vgg16_Relu_99_0	type: float	shape: Shape{32, 512, 28, 28}
//	- name: vgg16_Reshape_100_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: vgg16_Convolution_101_0	type: float	shape: Shape{32, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_101(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
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
// Node name:	Convolution_95
// Description:	Convolution
// Input:
//	- name: vgg16_MaxPool_93_0	type: float	shape: Shape{32, 256, 28, 28}
//	- name: vgg16_Reshape_94_0	type: float	shape: Shape{512, 256, 3, 3}
// Output:
//	- name: vgg16_Convolution_95_0	type: float	shape: Shape{32, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_95(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 3, 3));
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
// Node name:	Reshape_75
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_14_0	type: float	shape: Shape{3, 3, 128, 256}
// Output:
//	- name: vgg16_Reshape_75_0	type: float	shape: Shape{256, 128, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_75(float* input0, float* output0)
{
    uint32_t input_strides0 = 32768;
    uint32_t input_strides1 = 256;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 1152;
    size_t nx = 256;
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
extern void vgg16_Reshape_float_float_cuda_Reshape_75_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_75<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_94
// Description:	Reshape
// Input:
//	- name: vgg16_Constant_23_0	type: float	shape: Shape{3, 3, 256, 512}
// Output:
//	- name: vgg16_Reshape_94_0	type: float	shape: Shape{512, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void vgg16_Reshape_float_float_cuda_Reshape_94(float* input0, float* output0)
{
    uint32_t input_strides0 = 131072;
    uint32_t input_strides1 = 512;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 9;
    uint32_t trans_strides2 = 2304;
    size_t nx = 512;
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
extern void vgg16_Reshape_float_float_cuda_Reshape_94_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_94<<<grids, blocks, mem, stream>>>(input0, output0);
}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {32, 224, 224, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {32, 1001}
#endif




extern "C" void vgg16_cuda_free()
{

CUDA_SAFE_CALL(cudaFree(vgg16_group_0_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(vgg16_group_persist_CUDA_GPU0_allocator_memory_pool));
CUBLAS_SAFE_CALL(cublasDestroy(vgg16_cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(vgg16_cudnn_handle_0));
}

#include "./include/dnn.h"

class vgg16_Reshape_float_float_cuda_Reshape_48_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_48_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_48_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_48_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_48<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_48_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_49_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_49_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_49_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_49_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_49<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_49_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_50Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_50Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_50";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 3, 224, 224, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_50(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 3, 3));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_50(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_51_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_51_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_51_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_51_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_51<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_51_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel : public Kernel {
public:
    vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_55_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_55_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_55_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_55_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_55<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_55_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_56Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_56Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_56";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 64, 224, 224, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_56(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_56(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_MaxPool_float_float_cuda_lib_MaxPool_61Kernel : public Kernel {
public:
    vgg16_MaxPool_float_float_cuda_lib_MaxPool_61Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_MaxPool_float_float_cuda_lib_MaxPool_61";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_61(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 224, 224));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 112, 112));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_61(cudnn_handle, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_62_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_62_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_62_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_62_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_62<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_62_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_63Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_63Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_63";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 64, 112, 112, 128, 64, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_63(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 64, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 64, 3, 3));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_63(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_64_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_64_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_64_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_64_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_64<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_64_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_68_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_68_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_68_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_68_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_68<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_68_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_69Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_69Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_69";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 128, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_69(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_69(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_MaxPool_float_float_cuda_lib_MaxPool_74Kernel : public Kernel {
public:
    vgg16_MaxPool_float_float_cuda_lib_MaxPool_74Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_MaxPool_float_float_cuda_lib_MaxPool_74";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_74(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 56, 56));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_74(cudnn_handle, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_75_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_75_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_75_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_75_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_75<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_75_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_76Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_76Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_76";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_76(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 128, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 128, 3, 3));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_76(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_77_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_77_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_77_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_77_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_77<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_77_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_81_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_81_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_81_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_81_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_81<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_81_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_82Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_82Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_82";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 256, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_82(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_82(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_MaxPool_float_float_cuda_lib_MaxPool_93Kernel : public Kernel {
public:
    vgg16_MaxPool_float_float_cuda_lib_MaxPool_93Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_MaxPool_float_float_cuda_lib_MaxPool_93";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_93(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 56, 56));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 28, 28));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_93(cudnn_handle, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_94_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_94_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_94_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_94_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_94<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_94_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_95Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_95Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_95";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_95(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 256, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 3, 3));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_95(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_96_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_96_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_96_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_96_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_96<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_96_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel : public Kernel {
public:
    vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Reshape_float_float_cuda_Reshape_100_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_100_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Reshape_float_float_cuda_Reshape_100<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Reshape_float_float_cuda_Reshape_100_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_101Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_101Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_101";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 512, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_101(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_101(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_MaxPool_float_float_cuda_lib_MaxPool_112Kernel : public Kernel {
public:
    vgg16_MaxPool_float_float_cuda_lib_MaxPool_112Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_MaxPool_float_float_cuda_lib_MaxPool_112";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_112(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 28, 28));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_112(cudnn_handle, input0, output0);
    }
};


class vgg16_Convolution_float_float_float_cuda_lib_Convolution_114Kernel : public Kernel {
public:
    vgg16_Convolution_float_float_float_cuda_lib_Convolution_114Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Convolution_float_float_float_cuda_lib_Convolution_114";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({32, 512, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_114(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
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
        this->Convolution_float_float_float_cuda_lib_Convolution_114(cudnn_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_115_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_115_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_115_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_115_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_115<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_115_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_MaxPool_float_float_cuda_lib_MaxPool_131Kernel : public Kernel {
public:
    vgg16_MaxPool_float_float_cuda_lib_MaxPool_131Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_MaxPool_float_float_cuda_lib_MaxPool_131";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_131(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 14, 14));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 32, 512, 7, 7));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}

    void executeImpl(cudaStream_t stream) {
        this->MaxPool_float_float_cuda_lib_MaxPool_131(cudnn_handle, input0, output0);
    }
};


class vgg16_Dot_float_float_float_cuda_lib_Dot_133Kernel : public Kernel {
public:
    vgg16_Dot_float_float_float_cuda_lib_Dot_133Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Dot_float_float_float_cuda_lib_Dot_133";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 4096;
    ret[1] = 32;
    ret[2] = 25088;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_133(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32, 25088, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 25088, &beta, static_cast<float*>(output0), 4096));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_133(cublas_handle, input0, input1, output0);
    }
};


class vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_CallKernel : public Kernel {
public:
    vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vgg16_Dot_float_float_float_cuda_lib_Dot_137Kernel : public Kernel {
public:
    vgg16_Dot_float_float_float_cuda_lib_Dot_137Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Dot_float_float_float_cuda_lib_Dot_137";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 4096;
    ret[1] = 32;
    ret[2] = 4096;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_137(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 32, 4096, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 4096));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_137(cublas_handle, input0, input1, output0);
    }
};


class vgg16_Dot_float_float_float_cuda_lib_Dot_141Kernel : public Kernel {
public:
    vgg16_Dot_float_float_float_cuda_lib_Dot_141Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Dot_float_float_float_cuda_lib_Dot_141";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1001;
    ret[1] = 32;
    ret[2] = 4096;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_141(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 32, 4096, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1001));

}

    void executeImpl(cudaStream_t stream) {
        this->Dot_float_float_float_cuda_lib_Dot_141(cublas_handle, input0, input1, output0);
    }
};


class vgg16_Broadcast_float_float_cuda_Broadcast_142_CallKernel : public Kernel {
public:
    vgg16_Broadcast_float_float_cuda_Broadcast_142_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Broadcast_float_float_cuda_Broadcast_142_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_142_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    vgg16_Broadcast_float_float_cuda_Broadcast_142<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Broadcast_float_float_cuda_Broadcast_142_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class vgg16_Add_float_float_float_cuda_Add_143_CallKernel : public Kernel {
public:
    vgg16_Add_float_float_float_cuda_Add_143_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Add_float_float_float_cuda_Add_143_Call";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_143_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    vgg16_Add_float_float_float_cuda_Add_143<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        this->Add_float_float_float_cuda_Add_143_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class vgg16_Result_float_float_cuda_lib_Result_144Kernel : public Kernel {
public:
    vgg16_Result_float_float_cuda_lib_Result_144Kernel(float*  input0, float**  output0, float*  Parameter_0_0, float**  Result_144_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_0_0 = Parameter_0_0, this->Result_144_0 = Result_144_0;
        this->kernelName = "vgg16_Result_float_float_cuda_lib_Result_144";
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
    float*  Parameter_0_0; float**  Result_144_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_144(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        this->Result_float_float_cuda_lib_Result_144(input0, output0);
    }
};
void VGG16::gen_vector(float*  Parameter_0_0, float**  Result_144_0) {
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_48_CallKernel(dim3(1, 3136, 32), dim3(16, 16, 1), 0, nullptr, std::move(Parameter_0_0), std::move(vgg16_Reshape_48_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_49_CallKernel(dim3(4, 3, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_2_0), std::move(vgg16_Reshape_49_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_50Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Reshape_48_0), std::move(vgg16_Reshape_49_0), std::move(vgg16_Convolution_50_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_51_CallKernel(dim3(1605632, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_3_0), std::move(vgg16_Broadcast_51_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(200704, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_50_0), std::move(vgg16_Broadcast_51_0), std::move(vgg16_Relu_54_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_55_CallKernel(dim3(4, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_5_0), std::move(vgg16_Reshape_55_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_56Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_54_0), std::move(vgg16_Reshape_55_0), std::move(vgg16_Convolution_56_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_51_CallKernel(dim3(1605632, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_6_0), std::move(vgg16_Broadcast_57_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(200704, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_56_0), std::move(vgg16_Broadcast_57_0), std::move(vgg16_Relu_60_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_MaxPool_float_float_cuda_lib_MaxPool_61Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_60_0), std::move(vgg16_MaxPool_61_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_62_CallKernel(dim3(8, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_8_0), std::move(vgg16_Reshape_62_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_63Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_MaxPool_61_0), std::move(vgg16_Reshape_62_0), std::move(vgg16_Convolution_63_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_64_CallKernel(dim3(802816, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_9_0), std::move(vgg16_Broadcast_64_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_63_0), std::move(vgg16_Broadcast_64_0), std::move(vgg16_Relu_67_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_68_CallKernel(dim3(8, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_11_0), std::move(vgg16_Reshape_68_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_69Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_67_0), std::move(vgg16_Reshape_68_0), std::move(vgg16_Convolution_69_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_64_CallKernel(dim3(802816, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_12_0), std::move(vgg16_Broadcast_70_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(100352, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_69_0), std::move(vgg16_Broadcast_70_0), std::move(vgg16_Relu_73_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_MaxPool_float_float_cuda_lib_MaxPool_74Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_73_0), std::move(vgg16_MaxPool_74_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_75_CallKernel(dim3(16, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_14_0), std::move(vgg16_Reshape_75_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_76Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_MaxPool_74_0), std::move(vgg16_Reshape_75_0), std::move(vgg16_Convolution_76_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_77_CallKernel(dim3(401408, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_15_0), std::move(vgg16_Broadcast_77_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_76_0), std::move(vgg16_Broadcast_77_0), std::move(vgg16_Relu_80_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_81_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_17_0), std::move(vgg16_Reshape_81_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_82Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_80_0), std::move(vgg16_Reshape_81_0), std::move(vgg16_Convolution_82_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_77_CallKernel(dim3(401408, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_18_0), std::move(vgg16_Broadcast_83_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_82_0), std::move(vgg16_Broadcast_83_0), std::move(vgg16_Relu_86_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_81_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_20_0), std::move(vgg16_Reshape_87_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_82Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_86_0), std::move(vgg16_Reshape_87_0), std::move(vgg16_Convolution_88_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_77_CallKernel(dim3(401408, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_21_0), std::move(vgg16_Broadcast_89_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(50176, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_88_0), std::move(vgg16_Broadcast_89_0), std::move(vgg16_Relu_92_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_MaxPool_float_float_cuda_lib_MaxPool_93Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_92_0), std::move(vgg16_MaxPool_93_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_94_CallKernel(dim3(32, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_23_0), std::move(vgg16_Reshape_94_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_95Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_MaxPool_93_0), std::move(vgg16_Reshape_94_0), std::move(vgg16_Convolution_95_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_96_CallKernel(dim3(200704, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_24_0), std::move(vgg16_Broadcast_96_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_95_0), std::move(vgg16_Broadcast_96_0), std::move(vgg16_Relu_99_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_26_0), std::move(vgg16_Reshape_100_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_101Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_99_0), std::move(vgg16_Reshape_100_0), std::move(vgg16_Convolution_101_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_96_CallKernel(dim3(200704, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_27_0), std::move(vgg16_Broadcast_102_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_101_0), std::move(vgg16_Broadcast_102_0), std::move(vgg16_Relu_105_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_29_0), std::move(vgg16_Reshape_106_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_101Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_105_0), std::move(vgg16_Reshape_106_0), std::move(vgg16_Convolution_107_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_96_CallKernel(dim3(200704, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_30_0), std::move(vgg16_Broadcast_108_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(25088, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_107_0), std::move(vgg16_Broadcast_108_0), std::move(vgg16_Relu_111_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_MaxPool_float_float_cuda_lib_MaxPool_112Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_111_0), std::move(vgg16_MaxPool_112_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_32_0), std::move(vgg16_Reshape_113_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_114Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_MaxPool_112_0), std::move(vgg16_Reshape_113_0), std::move(vgg16_Convolution_114_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_115_CallKernel(dim3(50176, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_33_0), std::move(vgg16_Broadcast_115_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_114_0), std::move(vgg16_Broadcast_115_0), std::move(vgg16_Relu_118_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_35_0), std::move(vgg16_Reshape_119_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_114Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_118_0), std::move(vgg16_Reshape_119_0), std::move(vgg16_Convolution_120_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_115_CallKernel(dim3(50176, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_36_0), std::move(vgg16_Broadcast_121_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_120_0), std::move(vgg16_Broadcast_121_0), std::move(vgg16_Relu_124_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Reshape_float_float_cuda_Reshape_100_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(vgg16_Constant_38_0), std::move(vgg16_Reshape_125_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Convolution_float_float_float_cuda_lib_Convolution_114Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_124_0), std::move(vgg16_Reshape_125_0), std::move(vgg16_Convolution_126_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_115_CallKernel(dim3(50176, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_39_0), std::move(vgg16_Broadcast_127_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Convolution_126_0), std::move(vgg16_Broadcast_127_0), std::move(vgg16_Relu_130_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_MaxPool_float_float_cuda_lib_MaxPool_131Kernel(std::move(vgg16_cudnn_handle_0), std::move(vgg16_Relu_130_0), std::move(vgg16_MaxPool_131_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Dot_float_float_float_cuda_lib_Dot_133Kernel(std::move(vgg16_cublas_handle_0), std::move(vgg16_Reshape_132_0), std::move(vgg16_Constant_42_0), std::move(vgg16_Dot_133_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Constant_43_0), std::move(vgg16_Dot_133_0), std::move(vgg16_Relu_136_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Dot_float_float_float_cuda_lib_Dot_137Kernel(std::move(vgg16_cublas_handle_0), std::move(vgg16_Relu_136_0), std::move(vgg16_Constant_44_0), std::move(vgg16_Dot_137_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_13_CallKernel(dim3(256, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(vgg16_Constant_45_0), std::move(vgg16_Dot_137_0), std::move(vgg16_Relu_140_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Dot_float_float_float_cuda_lib_Dot_141Kernel(std::move(vgg16_cublas_handle_0), std::move(vgg16_Relu_140_0), std::move(vgg16_Constant_46_0), std::move(vgg16_Dot_141_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Broadcast_float_float_cuda_Broadcast_142_CallKernel(dim3(501, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(vgg16_Constant_47_0), std::move(vgg16_Broadcast_142_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Add_float_float_float_cuda_Add_143_CallKernel(dim3(77, 1, 1), dim3(416, 1, 1), 0, nullptr, std::move(vgg16_Dot_141_0), std::move(vgg16_Broadcast_142_0), std::move(vgg16_Add_143_0), std::move(Parameter_0_0), std::move(Result_144_0)));
    kernels.emplace_back(new vgg16_Result_float_float_cuda_lib_Result_144Kernel(std::move(vgg16_Add_143_0), std::move(Result_144_0), std::move(Parameter_0_0), std::move(Result_144_0)));
}
