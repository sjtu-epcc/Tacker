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
float* Reshape_33_0;
float* Reshape_34_0;
float* Convolution_35_0;
float* Broadcast_36_0;
float* Relu_39_0;
float* MaxPool_40_0;
float* Reshape_41_0;
float* Convolution_42_0;
float* Broadcast_43_0;
float* Relu_46_0;
float* MaxPool_47_0;
float* Reshape_48_0;
float* Convolution_49_0;
float* Broadcast_50_0;
float* Relu_53_0;
float* Reshape_54_0;
float* Convolution_55_0;
float* Broadcast_56_0;
float* Relu_59_0;
float* MaxPool_60_0;
float* Reshape_61_0;
float* Convolution_62_0;
float* Broadcast_63_0;
float* Relu_66_0;
float* Reshape_67_0;
float* Convolution_68_0;
float* Broadcast_69_0;
float* Relu_72_0;
float* MaxPool_73_0;
float* Reshape_74_0;
float* Convolution_75_0;
float* Broadcast_76_0;
float* Relu_79_0;
float* Reshape_80_0;
float* Convolution_81_0;
float* Broadcast_82_0;
float* Relu_85_0;
float* MaxPool_86_0;
float* Reshape_87_0;
float* Dot_88_0;
float* Relu_91_0;
float* Dot_92_0;
float* Relu_95_0;
float* Dot_96_0;
float* Add_98_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
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
cublasHandle_t cublas_handle_0;
cudnnHandle_t cudnn_handle_0;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_10_0;
float* Constant_17_0;
float* Constant_16_0;
float* Constant_19_0;
float* Constant_18_0;
float* Constant_21_0;
float* Constant_20_0;
float* Constant_23_0;
float* Constant_22_0;
float* Constant_25_0;
float* Constant_24_0;
float* Constant_27_0;
float* Constant_26_0;
float* Constant_29_0;
float* Constant_28_0;
float* Constant_31_0;
float* Constant_30_0;
float* Constant_13_0;
float* Broadcast_89_0;
float* Constant_11_0;
float* Constant_14_0;
float* Broadcast_93_0;
float* Constant_12_0;
float* Constant_15_0;
float* Broadcast_97_0;
// Node name:	MaxPool_60
// Description:	MaxPool
// Input:
//	- name: Relu_59_0	type: float	shape: Shape{1, 256, 56, 56}
// Output:
//	- name: MaxPool_60_0	type: float	shape: Shape{1, 256, 28, 28}
void MaxPool_float_float_cuda_lib_MaxPool_60(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 28, 28));
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
// Node name:	Reshape_41
// Description:	Reshape
// Input:
//	- name: Constant_19_0	type: float	shape: Shape{3, 3, 64, 128}
// Output:
//	- name: Reshape_41_0	type: float	shape: Shape{128, 64, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_41(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_41_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_41<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_92
// Description:	Dot
// Input:
//	- name: Relu_91_0	type: float	shape: Shape{1, 4096}
//	- name: Constant_11_0	type: float	shape: Shape{4096, 4096}
// Output:
//	- name: Dot_92_0	type: float	shape: Shape{1, 4096}
void Dot_float_float_float_cuda_lib_Dot_92(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 1, 4096, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: Constant_21_0	type: float	shape: Shape{3, 3, 128, 256}
void Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_21_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1179648];
    bin_file.read(tmp_mem, 1179648);
    cudaMemcpyAsync(output0, tmp_mem, 1179648, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_16
// Description:	Constant
// Input:
// Output:
//	- name: Constant_16_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_16(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_16_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: Constant_24_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_24_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_17
// Description:	Constant
// Input:
// Output:
//	- name: Constant_17_0	type: float	shape: Shape{3, 3, 3, 64}
void Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6912];
    bin_file.read(tmp_mem, 6912);
    cudaMemcpyAsync(output0, tmp_mem, 6912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_13
// Description:	Constant
// Input:
// Output:
//	- name: Constant_13_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_13(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_13_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_13_0 failed.\n");
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
//	- name: Constant_12_0	type: float	shape: Shape{4096, 1001}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16400384];
    bin_file.read(tmp_mem, 16400384);
    cudaMemcpyAsync(output0, tmp_mem, 16400384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: Constant_23_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2359296];
    bin_file.read(tmp_mem, 2359296);
    cudaMemcpyAsync(output0, tmp_mem, 2359296, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: Constant_11_0	type: float	shape: Shape{4096, 4096}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[67108864];
    bin_file.read(tmp_mem, 67108864);
    cudaMemcpyAsync(output0, tmp_mem, 67108864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: Constant_15_0	type: float	shape: Shape{1001}
void Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_25
// Description:	Constant
// Input:
// Output:
//	- name: Constant_25_0	type: float	shape: Shape{3, 3, 256, 512}
void Constant_float_cuda_Constant_25(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_25_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_25_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4718592];
    bin_file.read(tmp_mem, 4718592);
    cudaMemcpyAsync(output0, tmp_mem, 4718592, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_10
// Description:	Constant
// Input:
// Output:
//	- name: Constant_10_0	type: float	shape: Shape{25088, 4096}
void Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[411041792];
    bin_file.read(tmp_mem, 411041792);
    cudaMemcpyAsync(output0, tmp_mem, 411041792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_26
// Description:	Constant
// Input:
// Output:
//	- name: Constant_26_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_26(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_26_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_26_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_20
// Description:	Constant
// Input:
// Output:
//	- name: Constant_20_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_20_0 failed.\n");
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
//	- name: Constant_22_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_22(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_22_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_22_0 failed.\n");
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
//	- name: Constant_29_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: Constant_31_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Broadcast_36
// Description:	Broadcast
// Input:
//	- name: Constant_16_0	type: float	shape: Shape{64}
// Output:
//	- name: Broadcast_36_0	type: float	shape: Shape{1, 64, 224, 224}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_36(float* input0, float* output0)
{
    size_t nthreads = 3211264;uint32_t strides0 = 3211264;
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
extern void Broadcast_float_float_cuda_Broadcast_36_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_36<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_50
// Description:	Broadcast
// Input:
//	- name: Constant_20_0	type: float	shape: Shape{256}
// Output:
//	- name: Broadcast_50_0	type: float	shape: Shape{1, 256, 56, 56}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_50(float* input0, float* output0)
{
    size_t nthreads = 802816;uint32_t strides0 = 802816;
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
extern void Broadcast_float_float_cuda_Broadcast_50_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_50<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_33
// Description:	Reshape
// Input:
//	- name: Parameter_32_0	type: float	shape: Shape{1, 224, 224, 3}
// Output:
//	- name: Reshape_33_0	type: float	shape: Shape{1, 3, 224, 224}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_33(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_33_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_33<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_48
// Description:	Reshape
// Input:
//	- name: Constant_21_0	type: float	shape: Shape{3, 3, 128, 256}
// Output:
//	- name: Reshape_48_0	type: float	shape: Shape{256, 128, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_48(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_48_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_48<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	MaxPool_73
// Description:	MaxPool
// Input:
//	- name: Relu_72_0	type: float	shape: Shape{1, 512, 28, 28}
// Output:
//	- name: MaxPool_73_0	type: float	shape: Shape{1, 512, 14, 14}
void MaxPool_float_float_cuda_lib_MaxPool_73(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
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
// Node name:	Convolution_68
// Description:	Convolution
// Input:
//	- name: Relu_66_0	type: float	shape: Shape{1, 512, 28, 28}
//	- name: Reshape_67_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: Convolution_68_0	type: float	shape: Shape{1, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_68(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
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
// Node name:	Convolution_75
// Description:	Convolution
// Input:
//	- name: MaxPool_73_0	type: float	shape: Shape{1, 512, 14, 14}
//	- name: Reshape_74_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: Convolution_75_0	type: float	shape: Shape{1, 512, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_75(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
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
// Node name:	MaxPool_86
// Description:	MaxPool
// Input:
//	- name: Relu_85_0	type: float	shape: Shape{1, 512, 14, 14}
// Output:
//	- name: MaxPool_86_0	type: float	shape: Shape{1, 512, 7, 7}
void MaxPool_float_float_cuda_lib_MaxPool_86(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
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
//	- name: Convolution_35_0	type: float	shape: Shape{1, 64, 224, 224}
//	- name: Broadcast_36_0	type: float	shape: Shape{1, 64, 224, 224}
// Output:
//	- name: Relu_39_0	type: float	shape: Shape{1, 64, 224, 224}
// Fused functions:
// Add_float_float_float_cuda_Add_37<<<dim3(6272, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_35_0, Broadcast_36_0, Add_37_0);
// Reshape_float_float_cuda_lib_Reshape_38(Add_37_0, Reshape_38_0);
// Relu_float_float_cuda_Relu_39<<<dim3(6272, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_38_0, Relu_39_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Reshape_61
// Description:	Reshape
// Input:
//	- name: Constant_25_0	type: float	shape: Shape{3, 3, 256, 512}
// Output:
//	- name: Reshape_61_0	type: float	shape: Shape{512, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_61(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_61_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_61<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Broadcast_63
// Description:	Broadcast
// Input:
//	- name: Constant_24_0	type: float	shape: Shape{512}
// Output:
//	- name: Broadcast_63_0	type: float	shape: Shape{1, 512, 28, 28}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_63(float* input0, float* output0)
{
    size_t nthreads = 401408;uint32_t strides0 = 401408;
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
extern void Broadcast_float_float_cuda_Broadcast_63_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_63<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Result_99
// Description:	Result
// Input:
//	- name: Add_98_0	type: float	shape: Shape{1, 1001}
// Output:
//	- name: Result_99_0	type: float	shape: Shape{1, 1001}
void Result_float_float_cuda_lib_Result_99(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: Constant_19_0	type: float	shape: Shape{3, 3, 64, 128}
void Constant_float_cuda_Constant_19(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("/home/jxdeng/workspace/tacker/runtime/dnn/vgg11/Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[294912];
    bin_file.read(tmp_mem, 294912);
    cudaMemcpyAsync(output0, tmp_mem, 294912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Dot_96
// Description:	Dot
// Input:
//	- name: Relu_95_0	type: float	shape: Shape{1, 4096}
//	- name: Constant_12_0	type: float	shape: Shape{4096, 1001}
// Output:
//	- name: Dot_96_0	type: float	shape: Shape{1, 1001}
void Dot_float_float_float_cuda_lib_Dot_96(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 1, 4096, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Dot_88
// Description:	Dot
// Input:
//	- name: Reshape_87_0	type: float	shape: Shape{1, 25088}
//	- name: Constant_10_0	type: float	shape: Shape{25088, 4096}
// Output:
//	- name: Dot_88_0	type: float	shape: Shape{1, 4096}
void Dot_float_float_float_cuda_lib_Dot_88(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 1, 25088, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 25088, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	Convolution_62
// Description:	Convolution
// Input:
//	- name: MaxPool_60_0	type: float	shape: Shape{1, 256, 28, 28}
//	- name: Reshape_61_0	type: float	shape: Shape{512, 256, 3, 3}
// Output:
//	- name: Convolution_62_0	type: float	shape: Shape{1, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_62(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 3, 3));
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
// Node name:	Reshape_34
// Description:	Reshape
// Input:
//	- name: Constant_17_0	type: float	shape: Shape{3, 3, 3, 64}
// Output:
//	- name: Reshape_34_0	type: float	shape: Shape{64, 3, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_34(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_34_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_34<<<grids, blocks, mem, stream>>>(input0, output0);
}

extern "C" void cuda_init()
{
// total memory:570613952

CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,39144192));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 39144192));
Reshape_33_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_34_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+602112);
Convolution_35_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+609024);
Broadcast_36_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+13454080);
Relu_39_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+26299136);
MaxPool_40_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_41_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Convolution_42_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3506176);
Broadcast_43_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9928704);
Relu_46_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16351232);
MaxPool_47_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_48_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Convolution_49_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2785280);
Broadcast_50_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+5996544);
Relu_53_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9207808);
Reshape_54_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_55_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2359296);
Broadcast_56_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+5570560);
Relu_59_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+8781824);
MaxPool_60_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_61_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_62_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+5521408);
Broadcast_63_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_66_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Reshape_67_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
Convolution_68_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_69_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1605632);
Relu_72_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3211264);
MaxPool_73_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_74_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Convolution_75_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9838592);
Broadcast_76_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_79_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Reshape_80_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
Convolution_81_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_82_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+401408);
Relu_85_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+802816);
MaxPool_86_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_87_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_88_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+100352);
Relu_91_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_92_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16384);
Relu_95_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_96_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16384);
Add_98_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+16384);

CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,531469760));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 531469760));
Constant_10_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_17_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411041792);
Constant_16_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411048704);
Constant_19_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411048960);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411343872);
Constant_21_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411344384);
Constant_20_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+412524032);
Constant_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+412525056);
Constant_22_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+414884352);
Constant_25_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+414885376);
Constant_24_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+419603968);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+419606016);
Constant_26_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+429043200);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+429045248);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+438482432);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+438484480);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447921664);
Constant_13_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447923712);
Broadcast_89_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447923712);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447940096);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+515048960);
Broadcast_93_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+515048960);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+515065344);
Constant_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+531465728);
Broadcast_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+531465728);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
 // name=cg/affine0/weights
Constant_float_cuda_Constant_10(0, Constant_10_0);
 // name=cg/conv0/conv2d/kernel/read/_4__cf__4
Constant_float_cuda_Constant_17(0, Constant_17_0);
 // name=cg/conv0/biases/read/_3__cf__3
Constant_float_cuda_Constant_16(0, Constant_16_0);
 // name=cg/conv1/conv2d/kernel/read/_6__cf__6
Constant_float_cuda_Constant_19(0, Constant_19_0);
 // name=cg/conv1/biases/read/_5__cf__5
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=cg/conv2/conv2d/kernel/read/_8__cf__8
Constant_float_cuda_Constant_21(0, Constant_21_0);
 // name=cg/conv2/biases/read/_7__cf__7
Constant_float_cuda_Constant_20(0, Constant_20_0);
 // name=cg/conv3/conv2d/kernel/read/_10__cf__10
Constant_float_cuda_Constant_23(0, Constant_23_0);
 // name=cg/conv3/biases/read/_9__cf__9
Constant_float_cuda_Constant_22(0, Constant_22_0);
 // name=cg/conv4/conv2d/kernel/read/_12__cf__12
Constant_float_cuda_Constant_25(0, Constant_25_0);
 // name=cg/conv4/biases/read/_11__cf__11
Constant_float_cuda_Constant_24(0, Constant_24_0);
 // name=cg/conv5/conv2d/kernel/read/_14__cf__14
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=cg/conv5/biases/read/_13__cf__13
Constant_float_cuda_Constant_26(0, Constant_26_0);
 // name=cg/conv6/conv2d/kernel/read/_16__cf__16
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=cg/conv6/biases/read/_15__cf__15
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=cg/conv7/conv2d/kernel/read/_18__cf__18
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=cg/conv7/biases/read/_17__cf__17
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=cg/affine0/biases/read/_0__cf__0
Constant_float_cuda_Constant_13(0, Constant_13_0);
 // name=cg/affine1/weights
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=cg/affine1/biases/read/_1__cf__1
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=cg/affine2/weights
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=cg/affine2/biases/read/_2__cf__2
Constant_float_cuda_Constant_15(0, Constant_15_0);
}

// Node name:	MaxPool_40
// Description:	MaxPool
// Input:
//	- name: Relu_39_0	type: float	shape: Shape{1, 64, 224, 224}
// Output:
//	- name: MaxPool_40_0	type: float	shape: Shape{1, 64, 112, 112}
void MaxPool_float_float_cuda_lib_MaxPool_40(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 224, 224));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
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
// Node name:	Convolution_42
// Description:	Convolution
// Input:
//	- name: MaxPool_40_0	type: float	shape: Shape{1, 64, 112, 112}
//	- name: Reshape_41_0	type: float	shape: Shape{128, 64, 3, 3}
// Output:
//	- name: Convolution_42_0	type: float	shape: Shape{1, 128, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_42(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 64, 3, 3));
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
// Node name:	Broadcast_43
// Description:	Broadcast
// Input:
//	- name: Constant_18_0	type: float	shape: Shape{128}
// Output:
//	- name: Broadcast_43_0	type: float	shape: Shape{1, 128, 112, 112}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_43(float* input0, float* output0)
{
    size_t nthreads = 1605632;uint32_t strides0 = 1605632;
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
extern void Broadcast_float_float_cuda_Broadcast_43_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_43<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_67
// Description:	Reshape
// Input:
//	- name: Constant_27_0	type: float	shape: Shape{3, 3, 512, 512}
// Output:
//	- name: Reshape_67_0	type: float	shape: Shape{512, 512, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_67(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_67_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_67<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_55
// Description:	Convolution
// Input:
//	- name: Relu_53_0	type: float	shape: Shape{1, 256, 56, 56}
//	- name: Reshape_54_0	type: float	shape: Shape{256, 256, 3, 3}
// Output:
//	- name: Convolution_55_0	type: float	shape: Shape{1, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_55(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
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
// Node name:	MaxPool_47
// Description:	MaxPool
// Input:
//	- name: Relu_46_0	type: float	shape: Shape{1, 128, 112, 112}
// Output:
//	- name: MaxPool_47_0	type: float	shape: Shape{1, 128, 56, 56}
void MaxPool_float_float_cuda_lib_MaxPool_47(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 56, 56));
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
// Node name:	Add_98
// Description:	Add
// Input:
//	- name: Dot_96_0	type: float	shape: Shape{1, 1001}
//	- name: Broadcast_97_0	type: float	shape: Shape{1, 1001}
// Output:
//	- name: Add_98_0	type: float	shape: Shape{1, 1001}
extern "C" __launch_bounds__(143) __global__ void Add_float_float_float_cuda_Add_98(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 143 + threadIdx.x] = add(input0[blockIdx.x * 143 + threadIdx.x], input1[blockIdx.x * 143 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_98_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_98<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Broadcast_76
// Description:	Broadcast
// Input:
//	- name: Constant_28_0	type: float	shape: Shape{512}
// Output:
//	- name: Broadcast_76_0	type: float	shape: Shape{1, 512, 14, 14}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_76(float* input0, float* output0)
{
    size_t nthreads = 100352;uint32_t strides0 = 100352;
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
extern void Broadcast_float_float_cuda_Broadcast_76_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_76<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_35
// Description:	Convolution
// Input:
//	- name: Reshape_33_0	type: float	shape: Shape{1, 3, 224, 224}
//	- name: Reshape_34_0	type: float	shape: Shape{64, 3, 3, 3}
// Output:
//	- name: Convolution_35_0	type: float	shape: Shape{1, 64, 224, 224}
void Convolution_float_float_float_cuda_lib_Convolution_35(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 224, 224));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 3, 3));
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
// Node name:	Reshape_54
// Description:	Reshape
// Input:
//	- name: Constant_23_0	type: float	shape: Shape{3, 3, 256, 256}
// Output:
//	- name: Reshape_54_0	type: float	shape: Shape{256, 256, 3, 3}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_54(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_54_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_54<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_49
// Description:	Convolution
// Input:
//	- name: MaxPool_47_0	type: float	shape: Shape{1, 128, 56, 56}
//	- name: Reshape_48_0	type: float	shape: Shape{256, 128, 3, 3}
// Output:
//	- name: Convolution_49_0	type: float	shape: Shape{1, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_49(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 128, 3, 3));
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

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {1, 224, 224, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {1, 1001}
#endif




extern "C" void vgg11_cuda_free()
{

CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));

CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));
}

#include "./include/dnn.h"

class Reshape_float_float_cuda_Reshape_33_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_33_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_33_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_33_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_33<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_33_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_34_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_34_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_34_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_34_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_34<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_34_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_35Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_35Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_35";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 3, 224, 224, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_35(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 224, 224));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 3, 3));
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
        Convolution_float_float_float_cuda_lib_Convolution_35(cudnn_handle, input0, input1, output0);
    }
};


class Broadcast_float_float_cuda_Broadcast_36_CallKernel : public Kernel {
public:
    Broadcast_float_float_cuda_Broadcast_36_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Broadcast_float_float_cuda_Broadcast_36_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_36_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_36<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Broadcast_float_float_cuda_Broadcast_36_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel : public Kernel {
public:
    FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_40Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_40Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_40";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_40(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 224, 224));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
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
        MaxPool_float_float_cuda_lib_MaxPool_40(cudnn_handle, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_41_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_41_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_41_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_41_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_41<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_41_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_42Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_42Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_42";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 64, 112, 112, 128, 64, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_42(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 112, 112));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 64, 3, 3));
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
        Convolution_float_float_float_cuda_lib_Convolution_42(cudnn_handle, input0, input1, output0);
    }
};


class Broadcast_float_float_cuda_Broadcast_43_CallKernel : public Kernel {
public:
    Broadcast_float_float_cuda_Broadcast_43_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Broadcast_float_float_cuda_Broadcast_43_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_43_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_43<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Broadcast_float_float_cuda_Broadcast_43_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_47Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_47Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_47";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_47(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 56, 56));
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
        MaxPool_float_float_cuda_lib_MaxPool_47(cudnn_handle, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_48_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_48_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_48_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_48_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_48<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_48_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_49Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_49Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_49";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_49(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 128, 3, 3));
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
        Convolution_float_float_float_cuda_lib_Convolution_49(cudnn_handle, input0, input1, output0);
    }
};


class Broadcast_float_float_cuda_Broadcast_50_CallKernel : public Kernel {
public:
    Broadcast_float_float_cuda_Broadcast_50_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Broadcast_float_float_cuda_Broadcast_50_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_50_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_50<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Broadcast_float_float_cuda_Broadcast_50_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_54_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_54_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_54_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_54_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_54<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_54_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_55Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_55Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_55";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_55(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
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
        Convolution_float_float_float_cuda_lib_Convolution_55(cudnn_handle, input0, input1, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_60Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_60Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_60";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_60(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 28, 28));
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
        MaxPool_float_float_cuda_lib_MaxPool_60(cudnn_handle, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_61_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_61_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_61_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_61_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_61<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_61_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_62Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_62Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_62";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_62(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 256, 3, 3));
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
        Convolution_float_float_float_cuda_lib_Convolution_62(cudnn_handle, input0, input1, output0);
    }
};


class Broadcast_float_float_cuda_Broadcast_63_CallKernel : public Kernel {
public:
    Broadcast_float_float_cuda_Broadcast_63_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Broadcast_float_float_cuda_Broadcast_63_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_63_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_63<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Broadcast_float_float_cuda_Broadcast_63_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Reshape_float_float_cuda_Reshape_67_CallKernel : public Kernel {
public:
    Reshape_float_float_cuda_Reshape_67_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Reshape_float_float_cuda_Reshape_67_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Reshape_float_float_cuda_Reshape_67_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_67<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Reshape_float_float_cuda_Reshape_67_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_68Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_68Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_68";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_68(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
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
        Convolution_float_float_float_cuda_lib_Convolution_68(cudnn_handle, input0, input1, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_73Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_73Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_73";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_73(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 28, 28));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
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
        MaxPool_float_float_cuda_lib_MaxPool_73(cudnn_handle, input0, output0);
    }
};


class Convolution_float_float_float_cuda_lib_Convolution_75Kernel : public Kernel {
public:
    Convolution_float_float_float_cuda_lib_Convolution_75Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Convolution_float_float_float_cuda_lib_Convolution_75";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>({1, 512, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1});
}

    void Convolution_float_float_float_cuda_lib_Convolution_75(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
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
        Convolution_float_float_float_cuda_lib_Convolution_75(cudnn_handle, input0, input1, output0);
    }
};


class Broadcast_float_float_cuda_Broadcast_76_CallKernel : public Kernel {
public:
    Broadcast_float_float_cuda_Broadcast_76_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Broadcast_float_float_cuda_Broadcast_76_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Broadcast_float_float_cuda_Broadcast_76_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_76<<<grids, blocks, mem, stream>>>(input0, output0);
}

    void executeImpl(cudaStream_t stream) {
        Broadcast_float_float_cuda_Broadcast_76_Call(grids, blocks, mem, stream, input0, output0);
    }
};


class MaxPool_float_float_cuda_lib_MaxPool_86Kernel : public Kernel {
public:
    MaxPool_float_float_cuda_lib_MaxPool_86Kernel(cudnnHandle_t  cudnn_handle, float*  input0, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cudnn_handle = cudnn_handle, this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "MaxPool_float_float_cuda_lib_MaxPool_86";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void MaxPool_float_float_cuda_lib_MaxPool_86(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 14, 14));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 7, 7));
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
        MaxPool_float_float_cuda_lib_MaxPool_86(cudnn_handle, input0, output0);
    }
};


class Dot_float_float_float_cuda_lib_Dot_88Kernel : public Kernel {
public:
    Dot_float_float_float_cuda_lib_Dot_88Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Dot_float_float_float_cuda_lib_Dot_88";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 4096;
    ret[1] = 1;
    ret[2] = 25088;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_88(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 1, 25088, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 25088, &beta, static_cast<float*>(output0), 4096));

}

    void executeImpl(cudaStream_t stream) {
        Dot_float_float_float_cuda_lib_Dot_88(cublas_handle, input0, input1, output0);
    }
};


class Dot_float_float_float_cuda_lib_Dot_92Kernel : public Kernel {
public:
    Dot_float_float_float_cuda_lib_Dot_92Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Dot_float_float_float_cuda_lib_Dot_92";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 4096;
    ret[1] = 1;
    ret[2] = 4096;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_92(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 1, 4096, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 4096));

}

    void executeImpl(cudaStream_t stream) {
        Dot_float_float_float_cuda_lib_Dot_92(cublas_handle, input0, input1, output0);
    }
};


class Dot_float_float_float_cuda_lib_Dot_96Kernel : public Kernel {
public:
    Dot_float_float_float_cuda_lib_Dot_96Kernel(cublasHandle_t  cublas_handle, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->cublas_handle = cublas_handle, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Dot_float_float_float_cuda_lib_Dot_96";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    std::vector<int> ret(3);
    ret[0] = 1001;
    ret[1] = 1;
    ret[2] = 4096;
    return ret;
}

    void Dot_float_float_float_cuda_lib_Dot_96(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(mycublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 1, 4096, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1001));

}

    void executeImpl(cudaStream_t stream) {
        Dot_float_float_float_cuda_lib_Dot_96(cublas_handle, input0, input1, output0);
    }
};


class Add_float_float_float_cuda_Add_98_CallKernel : public Kernel {
public:
    Add_float_float_float_cuda_Add_98_CallKernel(const dim3 & grids, const dim3 & blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Add_float_float_float_cuda_Add_98_Call";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

     void Add_float_float_float_cuda_Add_98_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_98<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

    void executeImpl(cudaStream_t stream) {
        Add_float_float_float_cuda_Add_98_Call(grids, blocks, mem, stream, input0, input1, output0);
    }
};


class Result_float_float_cuda_lib_Result_99Kernel : public Kernel {
public:
    Result_float_float_cuda_lib_Result_99Kernel(float*  input0, float**  output0, float*  Parameter_32_0, float**  Result_99_0) {
        this->input0 = input0, this->output0 = output0, this->Parameter_32_0 = Parameter_32_0, this->Result_99_0 = Result_99_0;
        this->kernelName = "Result_float_float_cuda_lib_Result_99";
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
    float*  Parameter_32_0; float**  Result_99_0;
private:

    
std::vector<int> getArgs() override {
    return std::vector<int>();
}

    void Result_float_float_cuda_lib_Result_99(float* input0, float** output0)
{
    *output0 = input0;
}

    void executeImpl(cudaStream_t stream) {
        Result_float_float_cuda_lib_Result_99(input0, output0);
    }
};
void VGG11::gen_vector(float*  Parameter_32_0, float**  Result_99_0) {
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_33_CallKernel(dim3(1, 3136, 1), dim3(16, 16, 1), 0, nullptr, std::move(Parameter_32_0), std::move(Reshape_33_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_34_CallKernel(dim3(4, 3, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_17_0), std::move(Reshape_34_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_35Kernel(std::move(cudnn_handle_0), std::move(Reshape_33_0), std::move(Reshape_34_0), std::move(Convolution_35_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_36_CallKernel(dim3(50176, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_16_0), std::move(Broadcast_36_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(6272, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_35_0), std::move(Broadcast_36_0), std::move(Relu_39_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_40Kernel(std::move(cudnn_handle_0), std::move(Relu_39_0), std::move(MaxPool_40_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_41_CallKernel(dim3(8, 64, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_19_0), std::move(Reshape_41_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_42Kernel(std::move(cudnn_handle_0), std::move(MaxPool_40_0), std::move(Reshape_41_0), std::move(Convolution_42_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_43_CallKernel(dim3(25088, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_18_0), std::move(Broadcast_43_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(3136, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_42_0), std::move(Broadcast_43_0), std::move(Relu_46_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_47Kernel(std::move(cudnn_handle_0), std::move(Relu_46_0), std::move(MaxPool_47_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_48_CallKernel(dim3(16, 128, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_21_0), std::move(Reshape_48_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_49Kernel(std::move(cudnn_handle_0), std::move(MaxPool_47_0), std::move(Reshape_48_0), std::move(Convolution_49_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_50_CallKernel(dim3(12544, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_20_0), std::move(Broadcast_50_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_49_0), std::move(Broadcast_50_0), std::move(Relu_53_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_54_CallKernel(dim3(16, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_23_0), std::move(Reshape_54_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_55Kernel(std::move(cudnn_handle_0), std::move(Relu_53_0), std::move(Reshape_54_0), std::move(Convolution_55_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_50_CallKernel(dim3(12544, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_22_0), std::move(Broadcast_56_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(1568, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_55_0), std::move(Broadcast_56_0), std::move(Relu_59_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_60Kernel(std::move(cudnn_handle_0), std::move(Relu_59_0), std::move(MaxPool_60_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_61_CallKernel(dim3(32, 256, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_25_0), std::move(Reshape_61_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_62Kernel(std::move(cudnn_handle_0), std::move(MaxPool_60_0), std::move(Reshape_61_0), std::move(Convolution_62_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_63_CallKernel(dim3(6272, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_24_0), std::move(Broadcast_63_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_62_0), std::move(Broadcast_63_0), std::move(Relu_66_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_67_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_27_0), std::move(Reshape_67_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_68Kernel(std::move(cudnn_handle_0), std::move(Relu_66_0), std::move(Reshape_67_0), std::move(Convolution_68_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_63_CallKernel(dim3(6272, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_26_0), std::move(Broadcast_69_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(784, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_68_0), std::move(Broadcast_69_0), std::move(Relu_72_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_73Kernel(std::move(cudnn_handle_0), std::move(Relu_72_0), std::move(MaxPool_73_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_67_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_29_0), std::move(Reshape_74_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_75Kernel(std::move(cudnn_handle_0), std::move(MaxPool_73_0), std::move(Reshape_74_0), std::move(Convolution_75_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_76_CallKernel(dim3(1568, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_28_0), std::move(Broadcast_76_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_75_0), std::move(Broadcast_76_0), std::move(Relu_79_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Reshape_float_float_cuda_Reshape_67_CallKernel(dim3(32, 512, 1), dim3(16, 1, 16), 0, nullptr, std::move(Constant_31_0), std::move(Reshape_80_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Convolution_float_float_float_cuda_lib_Convolution_75Kernel(std::move(cudnn_handle_0), std::move(Relu_79_0), std::move(Reshape_80_0), std::move(Convolution_81_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Broadcast_float_float_cuda_Broadcast_76_CallKernel(dim3(1568, 1, 1), dim3(64, 1, 1), 0, nullptr, std::move(Constant_30_0), std::move(Broadcast_82_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(196, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Convolution_81_0), std::move(Broadcast_82_0), std::move(Relu_85_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new MaxPool_float_float_cuda_lib_MaxPool_86Kernel(std::move(cudnn_handle_0), std::move(Relu_85_0), std::move(MaxPool_86_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Dot_float_float_float_cuda_lib_Dot_88Kernel(std::move(cublas_handle_0), std::move(Reshape_87_0), std::move(Constant_10_0), std::move(Dot_88_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(8, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Dot_88_0), std::move(Broadcast_89_0), std::move(Relu_91_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Dot_float_float_float_cuda_lib_Dot_92Kernel(std::move(cublas_handle_0), std::move(Relu_91_0), std::move(Constant_11_0), std::move(Dot_92_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_CallKernel(dim3(8, 1, 1), dim3(512, 1, 1), 0, nullptr, std::move(Dot_92_0), std::move(Broadcast_93_0), std::move(Relu_95_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Dot_float_float_float_cuda_lib_Dot_96Kernel(std::move(cublas_handle_0), std::move(Relu_95_0), std::move(Constant_12_0), std::move(Dot_96_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Add_float_float_float_cuda_Add_98_CallKernel(dim3(7, 1, 1), dim3(143, 1, 1), 0, nullptr, std::move(Dot_96_0), std::move(Broadcast_97_0), std::move(Add_98_0), std::move(Parameter_32_0), std::move(Result_99_0)));
    kernels.emplace_back(new Result_float_float_cuda_lib_Result_99Kernel(std::move(Add_98_0), std::move(Result_99_0), std::move(Parameter_32_0), std::move(Result_99_0)));
}
