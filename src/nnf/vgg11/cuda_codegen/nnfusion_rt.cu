// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nnfusion_rt.h"
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
float* Broadcast_97_0;
float* Add_98_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_27_0;
float* Constant_2_0;
float* Constant_3_0;
float* Constant_5_0;
float* Constant_6_0;
float* Constant_8_0;
float* Constant_9_0;
float* Constant_11_0;
float* Constant_12_0;
float* Constant_14_0;
float* Constant_15_0;
float* Constant_17_0;
float* Constant_18_0;
float* Constant_20_0;
float* Constant_21_0;
float* Constant_23_0;
float* Constant_24_0;
float* Constant_28_0;
float* Constant_29_0;
float* Constant_30_0;
float* Constant_31_0;
float* Constant_32_0;
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
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
cublasHandle_t cublas_handle_0;
cudnnHandle_t cudnn_handle_0;
// Node name:	Reshape_54
// Description:	Reshape
// Input:
//	- name: Constant_11_0	type: float	shape: Shape{3, 3, 256, 256}
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
// Node name:	Reshape_67
// Description:	Reshape
// Input:
//	- name: Constant_17_0	type: float	shape: Shape{3, 3, 512, 512}
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
// Node name:	Dot_88
// Description:	Dot
// Input:
//	- name: Reshape_87_0	type: float	shape: Shape{64, 25088}
//	- name: Constant_27_0	type: float	shape: Shape{25088, 4096}
// Output:
//	- name: Dot_88_0	type: float	shape: Shape{64, 4096}
void Dot_float_float_float_cuda_lib_Dot_88(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 64, 25088, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 25088, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{25088, 4096}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[411041792];
    bin_file.read(tmp_mem, 411041792);
    cudaMemcpyAsync(output0, tmp_mem, 411041792, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: Constant_5_0	type: float	shape: Shape{3, 3, 64, 128}
void Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[294912];
    bin_file.read(tmp_mem, 294912);
    cudaMemcpyAsync(output0, tmp_mem, 294912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: Constant_29_0	type: float	shape: Shape{4096, 4096}
void Constant_float_cuda_Constant_29(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[67108864];
    bin_file.read(tmp_mem, 67108864);
    cudaMemcpyAsync(output0, tmp_mem, 67108864, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_24
// Description:	Constant
// Input:
// Output:
//	- name: Constant_24_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_24(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_24_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_23
// Description:	Constant
// Input:
// Output:
//	- name: Constant_23_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_23(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_23_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_23_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
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
//	- name: Constant_17_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_17(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_17_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_17_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: Constant_8_0	type: float	shape: Shape{3, 3, 128, 256}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1179648];
    bin_file.read(tmp_mem, 1179648);
    cudaMemcpyAsync(output0, tmp_mem, 1179648, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: Constant_32_0	type: float	shape: Shape{1001}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4004];
    bin_file.read(tmp_mem, 4004);
    cudaMemcpyAsync(output0, tmp_mem, 4004, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_31
// Description:	Constant
// Input:
// Output:
//	- name: Constant_31_0	type: float	shape: Shape{4096, 1001}
void Constant_float_cuda_Constant_31(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_31_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_31_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16400384];
    bin_file.read(tmp_mem, 16400384);
    cudaMemcpyAsync(output0, tmp_mem, 16400384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2_0	type: float	shape: Shape{3, 3, 3, 64}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6912];
    bin_file.read(tmp_mem, 6912);
    cudaMemcpyAsync(output0, tmp_mem, 6912, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: float	shape: Shape{4096}
void Constant_float_cuda_Constant_30(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: Constant_11_0	type: float	shape: Shape{3, 3, 256, 256}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
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
//	- name: Constant_12_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
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
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{3, 3, 256, 512}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_14_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_14_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4718592];
    bin_file.read(tmp_mem, 4718592);
    cudaMemcpyAsync(output0, tmp_mem, 4718592, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_21
// Description:	Constant
// Input:
// Output:
//	- name: Constant_21_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_21(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_21_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_21_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: Constant_9_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_15
// Description:	Constant
// Input:
// Output:
//	- name: Constant_15_0	type: float	shape: Shape{512}
void Constant_float_cuda_Constant_15(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_15_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_15_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2048];
    bin_file.read(tmp_mem, 2048);
    cudaMemcpyAsync(output0, tmp_mem, 2048, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
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
//	- name: Constant_20_0	type: float	shape: Shape{3, 3, 512, 512}
void Constant_float_cuda_Constant_20(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_20_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_20_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[9437184];
    bin_file.read(tmp_mem, 9437184);
    cudaMemcpyAsync(output0, tmp_mem, 9437184, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Convolution_68
// Description:	Convolution
// Input:
//	- name: Relu_66_0	type: float	shape: Shape{64, 512, 28, 28}
//	- name: Reshape_67_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: Convolution_68_0	type: float	shape: Shape{64, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_68(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
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
//	- name: Relu_85_0	type: float	shape: Shape{64, 512, 14, 14}
// Output:
//	- name: MaxPool_86_0	type: float	shape: Shape{64, 512, 7, 7}
void MaxPool_float_float_cuda_lib_MaxPool_86(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 14, 14));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 7, 7));
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
// Node name:	Broadcast_97
// Description:	Broadcast
// Input:
//	- name: Constant_32_0	type: float	shape: Shape{1001}
// Output:
//	- name: Broadcast_97_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_97(float* input0, float* output0)
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
extern void Broadcast_float_float_cuda_Broadcast_97_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_97<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Dot_92
// Description:	Dot
// Input:
//	- name: Relu_91_0	type: float	shape: Shape{64, 4096}
//	- name: Constant_29_0	type: float	shape: Shape{4096, 4096}
// Output:
//	- name: Dot_92_0	type: float	shape: Shape{64, 4096}
void Dot_float_float_float_cuda_lib_Dot_92(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 64, 4096, &alpha, static_cast<const float*>(input1), 4096, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 4096));

}
// Node name:	MaxPool_73
// Description:	MaxPool
// Input:
//	- name: Relu_72_0	type: float	shape: Shape{64, 512, 28, 28}
// Output:
//	- name: MaxPool_73_0	type: float	shape: Shape{64, 512, 14, 14}
void MaxPool_float_float_cuda_lib_MaxPool_73(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 14, 14));
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

extern "C" void cuda_init()
{
CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:3036262592
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,2504792832));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 2504792832));
Reshape_33_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_34_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+38535168);
Convolution_35_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+38542080);
Broadcast_36_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+860625664);
Relu_39_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1682709248);
MaxPool_40_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_41_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+205520896);
Convolution_42_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+205815808);
Broadcast_43_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+616857600);
Relu_46_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1027899392);
MaxPool_47_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_48_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+102760448);
Convolution_49_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+103940096);
Broadcast_50_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+309460992);
Relu_53_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+514981888);
Reshape_54_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_55_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2359296);
Broadcast_56_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+207880192);
Relu_59_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+413401088);
MaxPool_60_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_61_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+51380224);
Convolution_62_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+56098816);
Broadcast_63_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+158859264);
Relu_66_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+261619712);
Reshape_67_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_68_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9437184);
Broadcast_69_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+112197632);
Relu_72_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+214958080);
MaxPool_73_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_74_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+25690112);
Convolution_75_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35127296);
Broadcast_76_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Relu_79_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60817408);
Reshape_80_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Convolution_81_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+9437184);
Broadcast_82_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+35127296);
Relu_85_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+60817408);
MaxPool_86_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_87_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_88_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+6422528);
Relu_91_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_92_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1048576);
Relu_95_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_96_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1048576);
Broadcast_97_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_98_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1048576);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,531469760));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 531469760));
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411041792);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411048704);
Constant_5_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411048960);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411343872);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+411344384);
Constant_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+412524032);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+412525056);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+414884352);
Constant_14_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+414885376);
Constant_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+419603968);
Constant_17_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+419606016);
Constant_18_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+429043200);
Constant_20_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+429045248);
Constant_21_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+438482432);
Constant_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+438484480);
Constant_24_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447921664);
Constant_28_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447923712);
Constant_29_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+447940096);
Constant_30_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+515048960);
Constant_31_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+515065344);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+531465728);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));
 // name=cg/affine0/weights
Constant_float_cuda_Constant_27(0, Constant_27_0);
 // name=cg/conv0/conv2d/kernel
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=cg/conv0/biases
Constant_float_cuda_Constant_3(0, Constant_3_0);
 // name=cg/conv1/conv2d/kernel
Constant_float_cuda_Constant_5(0, Constant_5_0);
 // name=cg/conv1/biases
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=cg/conv2/conv2d/kernel
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=cg/conv2/biases
Constant_float_cuda_Constant_9(0, Constant_9_0);
 // name=cg/conv3/conv2d/kernel
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=cg/conv3/biases
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=cg/conv4/conv2d/kernel
Constant_float_cuda_Constant_14(0, Constant_14_0);
 // name=cg/conv4/biases
Constant_float_cuda_Constant_15(0, Constant_15_0);
 // name=cg/conv5/conv2d/kernel
Constant_float_cuda_Constant_17(0, Constant_17_0);
 // name=cg/conv5/biases
Constant_float_cuda_Constant_18(0, Constant_18_0);
 // name=cg/conv6/conv2d/kernel
Constant_float_cuda_Constant_20(0, Constant_20_0);
 // name=cg/conv6/biases
Constant_float_cuda_Constant_21(0, Constant_21_0);
 // name=cg/conv7/conv2d/kernel
Constant_float_cuda_Constant_23(0, Constant_23_0);
 // name=cg/conv7/biases
Constant_float_cuda_Constant_24(0, Constant_24_0);
 // name=cg/affine0/biases
Constant_float_cuda_Constant_28(0, Constant_28_0);
 // name=cg/affine1/weights
Constant_float_cuda_Constant_29(0, Constant_29_0);
 // name=cg/affine1/biases
Constant_float_cuda_Constant_30(0, Constant_30_0);
 // name=cg/affine2/weights
Constant_float_cuda_Constant_31(0, Constant_31_0);
 // name=cg/affine2/biases
Constant_float_cuda_Constant_32(0, Constant_32_0);
}

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_35_0	type: float	shape: Shape{64, 64, 224, 224}
//	- name: Broadcast_36_0	type: float	shape: Shape{64, 64, 224, 224}
// Output:
//	- name: Relu_39_0	type: float	shape: Shape{64, 64, 224, 224}
// Fused functions:
// Add_float_float_float_cuda_Add_37<<<dim3(401408, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_35_0, Broadcast_36_0, Add_37_0);
// Reshape_float_float_cuda_lib_Reshape_38(Add_37_0, Reshape_38_0);
// Relu_float_float_cuda_Relu_39<<<dim3(401408, 1, 1), dim3(512, 1, 1), 0, 0>>>(Reshape_38_0, Relu_39_0);
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
// Node name:	Convolution_75
// Description:	Convolution
// Input:
//	- name: MaxPool_73_0	type: float	shape: Shape{64, 512, 14, 14}
//	- name: Reshape_74_0	type: float	shape: Shape{512, 512, 3, 3}
// Output:
//	- name: Convolution_75_0	type: float	shape: Shape{64, 512, 14, 14}
void Convolution_float_float_float_cuda_lib_Convolution_75(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 14, 14));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 14, 14));
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
// Node name:	Reshape_61
// Description:	Reshape
// Input:
//	- name: Constant_14_0	type: float	shape: Shape{3, 3, 256, 512}
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
//	- name: Constant_15_0	type: float	shape: Shape{512}
// Output:
//	- name: Broadcast_63_0	type: float	shape: Shape{64, 512, 28, 28}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_63(float* input0, float* output0)
{
    size_t nthreads = 25690112;uint32_t strides0 = 401408;
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
//	- name: Add_98_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Result_99_0	type: float	shape: Shape{64, 1001}
void Result_float_float_cuda_lib_Result_99(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Add_98
// Description:	Add
// Input:
//	- name: Dot_96_0	type: float	shape: Shape{64, 1001}
//	- name: Broadcast_97_0	type: float	shape: Shape{64, 1001}
// Output:
//	- name: Add_98_0	type: float	shape: Shape{64, 1001}
extern "C" __launch_bounds__(64) __global__ void Add_float_float_float_cuda_Add_98(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 64 + threadIdx.x] = add(input0[blockIdx.x * 64 + threadIdx.x], input1[blockIdx.x * 64 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_98_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_98<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Dot_96
// Description:	Dot
// Input:
//	- name: Relu_95_0	type: float	shape: Shape{64, 4096}
//	- name: Constant_31_0	type: float	shape: Shape{4096, 1001}
// Output:
//	- name: Dot_96_0	type: float	shape: Shape{64, 1001}
void Dot_float_float_float_cuda_lib_Dot_96(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1001, 64, 4096, &alpha, static_cast<const float*>(input1), 1001, static_cast<const float*>(input0), 4096, &beta, static_cast<float*>(output0), 1001));

}
// Node name:	Reshape_48
// Description:	Reshape
// Input:
//	- name: Constant_8_0	type: float	shape: Shape{3, 3, 128, 256}
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_28_0	type: float	shape: Shape{4096}
//	- name: Dot_88_0	type: float	shape: Shape{64, 4096}
// Output:
//	- name: Relu_91_0	type: float	shape: Shape{64, 4096}
// Fused functions:
// Broadcast_float_float_cuda_Broadcast_89<<<dim3(4096, 1, 1), dim3(64, 1, 1), 0, 0>>>(Constant_28_0, Broadcast_89_0);
// Add_float_float_float_cuda_Add_90<<<dim3(512, 1, 1), dim3(512, 1, 1), 0, 0>>>(Dot_88_0, Broadcast_89_0, Add_90_0);
// Relu_float_float_cuda_Relu_91<<<dim3(512, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_90_0, Relu_91_0);
extern "C" __launch_bounds__(512) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_8(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = input0[tid % 4096];
    float temp1 = add(input1[tid], temp0);
    float temp2 = relu(temp1);
    output0[tid] = temp2;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_8_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_8<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Broadcast_36
// Description:	Broadcast
// Input:
//	- name: Constant_3_0	type: float	shape: Shape{64}
// Output:
//	- name: Broadcast_36_0	type: float	shape: Shape{64, 64, 224, 224}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_36(float* input0, float* output0)
{
    size_t nthreads = 205520896;uint32_t strides0 = 3211264;
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
//	- name: Constant_9_0	type: float	shape: Shape{256}
// Output:
//	- name: Broadcast_50_0	type: float	shape: Shape{64, 256, 56, 56}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_50(float* input0, float* output0)
{
    size_t nthreads = 51380224;uint32_t strides0 = 802816;
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
//	- name: Parameter_0_0	type: float	shape: Shape{64, 224, 224, 3}
// Output:
//	- name: Reshape_33_0	type: float	shape: Shape{64, 3, 224, 224}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_33(float* input0, float* output0)
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
extern void Reshape_float_float_cuda_Reshape_33_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_33<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Convolution_62
// Description:	Convolution
// Input:
//	- name: MaxPool_60_0	type: float	shape: Shape{64, 256, 28, 28}
//	- name: Reshape_61_0	type: float	shape: Shape{512, 256, 3, 3}
// Output:
//	- name: Convolution_62_0	type: float	shape: Shape{64, 512, 28, 28}
void Convolution_float_float_float_cuda_lib_Convolution_62(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 28, 28));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 512, 28, 28));
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
//	- name: Constant_2_0	type: float	shape: Shape{3, 3, 3, 64}
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
// Node name:	MaxPool_40
// Description:	MaxPool
// Input:
//	- name: Relu_39_0	type: float	shape: Shape{64, 64, 224, 224}
// Output:
//	- name: MaxPool_40_0	type: float	shape: Shape{64, 64, 112, 112}
void MaxPool_float_float_cuda_lib_MaxPool_40(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 224, 224));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
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
//	- name: Constant_5_0	type: float	shape: Shape{3, 3, 64, 128}
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
// Node name:	Convolution_42
// Description:	Convolution
// Input:
//	- name: MaxPool_40_0	type: float	shape: Shape{64, 64, 112, 112}
//	- name: Reshape_41_0	type: float	shape: Shape{128, 64, 3, 3}
// Output:
//	- name: Convolution_42_0	type: float	shape: Shape{64, 128, 112, 112}
void Convolution_float_float_float_cuda_lib_Convolution_42(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 112, 112));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 112, 112));
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
//	- name: Constant_6_0	type: float	shape: Shape{128}
// Output:
//	- name: Broadcast_43_0	type: float	shape: Shape{64, 128, 112, 112}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_43(float* input0, float* output0)
{
    size_t nthreads = 102760448;uint32_t strides0 = 1605632;
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
// Node name:	Convolution_55
// Description:	Convolution
// Input:
//	- name: Relu_53_0	type: float	shape: Shape{64, 256, 56, 56}
//	- name: Reshape_54_0	type: float	shape: Shape{256, 256, 3, 3}
// Output:
//	- name: Convolution_55_0	type: float	shape: Shape{64, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_55(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
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
//	- name: Relu_46_0	type: float	shape: Shape{64, 128, 112, 112}
// Output:
//	- name: MaxPool_47_0	type: float	shape: Shape{64, 128, 56, 56}
void MaxPool_float_float_cuda_lib_MaxPool_47(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 112, 112));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 56, 56));
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
// Node name:	Convolution_49
// Description:	Convolution
// Input:
//	- name: MaxPool_47_0	type: float	shape: Shape{64, 128, 56, 56}
//	- name: Reshape_48_0	type: float	shape: Shape{256, 128, 3, 3}
// Output:
//	- name: Convolution_49_0	type: float	shape: Shape{64, 256, 56, 56}
void Convolution_float_float_float_cuda_lib_Convolution_49(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 128, 56, 56));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
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
// Node name:	Broadcast_76
// Description:	Broadcast
// Input:
//	- name: Constant_21_0	type: float	shape: Shape{512}
// Output:
//	- name: Broadcast_76_0	type: float	shape: Shape{64, 512, 14, 14}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_76(float* input0, float* output0)
{
    size_t nthreads = 6422528;uint32_t strides0 = 100352;
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
// Node name:	MaxPool_60
// Description:	MaxPool
// Input:
//	- name: Relu_59_0	type: float	shape: Shape{64, 256, 56, 56}
// Output:
//	- name: MaxPool_60_0	type: float	shape: Shape{64, 256, 28, 28}
void MaxPool_float_float_cuda_lib_MaxPool_60(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 56, 56));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 256, 28, 28));
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
// Node name:	Convolution_35
// Description:	Convolution
// Input:
//	- name: Reshape_33_0	type: float	shape: Shape{64, 3, 224, 224}
//	- name: Reshape_34_0	type: float	shape: Shape{64, 3, 3, 3}
// Output:
//	- name: Convolution_35_0	type: float	shape: Shape{64, 64, 224, 224}
void Convolution_float_float_float_cuda_lib_Convolution_35(cudnnHandle_t cudnn_handle, float* input0, float* input1, float* output0)
{
    cudnnTensorDescriptor_t tensor_desc_0;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 3, 224, 224));
    cudnnTensorDescriptor_t tensor_desc_1;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 64, 64, 224, 224));
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

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {64, 224, 224, 3}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {64, 1001}
#endif


extern "C" int kernel_entry(float* Parameter_0_0, float** Result_99_0)
{
// kernel_entry_init
 // name=transpose
Reshape_float_float_cuda_Reshape_33_Call(dim3(1, 3136, 64), dim3(16, 16, 1), 0, 0, Parameter_0_0, Reshape_33_0);
 // name=Reshape_34
Reshape_float_float_cuda_Reshape_34_Call(dim3(4, 3, 1), dim3(16, 1, 16), 0, 0, Constant_2_0, Reshape_34_0);
 // name=cg/conv0/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_35(cudnn_handle_0, Reshape_33_0, Reshape_34_0, Convolution_35_0);
 // name=Broadcast_36
Broadcast_float_float_cuda_Broadcast_36_Call(dim3(3211264, 1, 1), dim3(64, 1, 1), 0, 0, Constant_3_0, Broadcast_36_0);
 // name=fused_kernel_100
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(401408, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_35_0, Broadcast_36_0, Relu_39_0);
 // name=cg/mpool0/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_40(cudnn_handle_0, Relu_39_0, MaxPool_40_0);
 // name=Reshape_41
Reshape_float_float_cuda_Reshape_41_Call(dim3(8, 64, 1), dim3(16, 1, 16), 0, 0, Constant_5_0, Reshape_41_0);
 // name=cg/conv1/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_42(cudnn_handle_0, MaxPool_40_0, Reshape_41_0, Convolution_42_0);
 // name=Broadcast_43
Broadcast_float_float_cuda_Broadcast_43_Call(dim3(1605632, 1, 1), dim3(64, 1, 1), 0, 0, Constant_6_0, Broadcast_43_0);
 // name=fused_kernel_101
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(200704, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_42_0, Broadcast_43_0, Relu_46_0);
 // name=cg/mpool1/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_47(cudnn_handle_0, Relu_46_0, MaxPool_47_0);
 // name=Reshape_48
Reshape_float_float_cuda_Reshape_48_Call(dim3(16, 128, 1), dim3(16, 1, 16), 0, 0, Constant_8_0, Reshape_48_0);
 // name=cg/conv2/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_49(cudnn_handle_0, MaxPool_47_0, Reshape_48_0, Convolution_49_0);
 // name=Broadcast_50
Broadcast_float_float_cuda_Broadcast_50_Call(dim3(802816, 1, 1), dim3(64, 1, 1), 0, 0, Constant_9_0, Broadcast_50_0);
 // name=fused_kernel_102
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(100352, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_49_0, Broadcast_50_0, Relu_53_0);
 // name=Reshape_54
Reshape_float_float_cuda_Reshape_54_Call(dim3(16, 256, 1), dim3(16, 1, 16), 0, 0, Constant_11_0, Reshape_54_0);
 // name=cg/conv3/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_55(cudnn_handle_0, Relu_53_0, Reshape_54_0, Convolution_55_0);
 // name=Broadcast_56
Broadcast_float_float_cuda_Broadcast_50_Call(dim3(802816, 1, 1), dim3(64, 1, 1), 0, 0, Constant_12_0, Broadcast_56_0);
 // name=fused_kernel_103
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(100352, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_55_0, Broadcast_56_0, Relu_59_0);
 // name=cg/mpool2/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_60(cudnn_handle_0, Relu_59_0, MaxPool_60_0);
 // name=Reshape_61
Reshape_float_float_cuda_Reshape_61_Call(dim3(32, 256, 1), dim3(16, 1, 16), 0, 0, Constant_14_0, Reshape_61_0);
 // name=cg/conv4/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_62(cudnn_handle_0, MaxPool_60_0, Reshape_61_0, Convolution_62_0);
 // name=Broadcast_63
Broadcast_float_float_cuda_Broadcast_63_Call(dim3(401408, 1, 1), dim3(64, 1, 1), 0, 0, Constant_15_0, Broadcast_63_0);
 // name=fused_kernel_104
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(50176, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_62_0, Broadcast_63_0, Relu_66_0);
 // name=Reshape_67
Reshape_float_float_cuda_Reshape_67_Call(dim3(32, 512, 1), dim3(16, 1, 16), 0, 0, Constant_17_0, Reshape_67_0);
 // name=cg/conv5/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_68(cudnn_handle_0, Relu_66_0, Reshape_67_0, Convolution_68_0);
 // name=Broadcast_69
Broadcast_float_float_cuda_Broadcast_63_Call(dim3(401408, 1, 1), dim3(64, 1, 1), 0, 0, Constant_18_0, Broadcast_69_0);
 // name=fused_kernel_105
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(50176, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_68_0, Broadcast_69_0, Relu_72_0);
 // name=cg/mpool3/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_73(cudnn_handle_0, Relu_72_0, MaxPool_73_0);
 // name=Reshape_74
Reshape_float_float_cuda_Reshape_67_Call(dim3(32, 512, 1), dim3(16, 1, 16), 0, 0, Constant_20_0, Reshape_74_0);
 // name=cg/conv6/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_75(cudnn_handle_0, MaxPool_73_0, Reshape_74_0, Convolution_75_0);
 // name=Broadcast_76
Broadcast_float_float_cuda_Broadcast_76_Call(dim3(100352, 1, 1), dim3(64, 1, 1), 0, 0, Constant_21_0, Broadcast_76_0);
 // name=fused_kernel_106
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(12544, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_75_0, Broadcast_76_0, Relu_79_0);
 // name=Reshape_80
Reshape_float_float_cuda_Reshape_67_Call(dim3(32, 512, 1), dim3(16, 1, 16), 0, 0, Constant_23_0, Reshape_80_0);
 // name=cg/conv7/conv2d/Conv2D
Convolution_float_float_float_cuda_lib_Convolution_75(cudnn_handle_0, Relu_79_0, Reshape_80_0, Convolution_81_0);
 // name=Broadcast_82
Broadcast_float_float_cuda_Broadcast_76_Call(dim3(100352, 1, 1), dim3(64, 1, 1), 0, 0, Constant_24_0, Broadcast_82_0);
 // name=fused_kernel_107
FusedKernel_float_float_float_cuda_Add_Reshape_Relu_0_Call(dim3(12544, 1, 1), dim3(512, 1, 1), 0, 0, Convolution_81_0, Broadcast_82_0, Relu_85_0);
 // name=cg/mpool4/MaxPool
MaxPool_float_float_cuda_lib_MaxPool_86(cudnn_handle_0, Relu_85_0, MaxPool_86_0);
 // name=cg/Reshape
// eliminated
 // name=cg/affine0/xw_plus_b/MatMul
Dot_float_float_float_cuda_lib_Dot_88(cublas_handle_0, Reshape_87_0, Constant_27_0, Dot_88_0);
 // name=fused_kernel_108
FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_8_Call(dim3(512, 1, 1), dim3(512, 1, 1), 0, 0, Constant_28_0, Dot_88_0, Relu_91_0);
 // name=cg/affine1/xw_plus_b/MatMul
Dot_float_float_float_cuda_lib_Dot_92(cublas_handle_0, Relu_91_0, Constant_29_0, Dot_92_0);
 // name=fused_kernel_109
FusedKernel_float_float_float_cuda_Broadcast_Add_Relu_8_Call(dim3(512, 1, 1), dim3(512, 1, 1), 0, 0, Constant_30_0, Dot_92_0, Relu_95_0);
 // name=cg/affine2/xw_plus_b/MatMul
Dot_float_float_float_cuda_lib_Dot_96(cublas_handle_0, Relu_95_0, Constant_31_0, Dot_96_0);
 // name=Broadcast_97
Broadcast_float_float_cuda_Broadcast_97_Call(dim3(1001, 1, 1), dim3(64, 1, 1), 0, 0, Constant_32_0, Broadcast_97_0);
 // name=cg/affine2/xw_plus_b
Add_float_float_float_cuda_Add_98_Call(dim3(1001, 1, 1), dim3(64, 1, 1), 0, 0, Dot_96_0, Broadcast_97_0, Add_98_0);
 // name=Result_99
Result_float_float_cuda_lib_Result_99(Add_98_0, Result_99_0);
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

