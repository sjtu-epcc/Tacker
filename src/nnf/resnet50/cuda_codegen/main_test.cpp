// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion_rt.h"
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <cuda_profiler_api.h>
#include <stdexcept>
#include <limits>
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

int main(void){

    cuda_init();

    //input argument
    float* Parameter_0_0_host, *Parameter_0_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_0_0_host, sizeof(float)* 9633792));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_0_0, sizeof(float) * 9633792));

    //output arguments
    float* Result_505_0_host, *Result_505_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_505_0_host, sizeof(float) * 64064));

    // fill input values
    for (int i = 0; i < 9633792; ++i) Parameter_0_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 9633792, cudaMemcpyHostToDevice));


    //warm up for 5 iters:
    for(int i_=0; i_< 5; i_++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 9633792, cudaMemcpyHostToDevice));
        kernel_entry(Parameter_0_0, &Result_505_0);
        CUDA_SAFE_CALL(cudaMemcpy(Result_505_0_host, Result_505_0,  sizeof(float) * 64064, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize()); 
        printf("%s \n", "Result_505_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_505_0_host[i]); 
        printf(" .. (size = 64064, ends with %e);\n", (float)Result_505_0_host[64063]);
    }

    //GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;
    cudaEvent_t start, stop, start_i, stop_i;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    //time measurement
    cudaEventRecord(start);

    //kernel call
    int steps = 100;
    cudaProfilerStart();
    for (int i_=0; i_<steps; i_++)
    {
        cudaEventRecord(start_i, 0);
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 9633792, cudaMemcpyHostToDevice));
        kernel_entry(Parameter_0_0, &Result_505_0);
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
    }
    cudaProfilerStop();
    //time measurement

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total/steps);

    //free context
    CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
    cuda_free();

    cudaFreeHost(Parameter_0_0_host);
    cudaFreeHost(Result_505_0_host);
    return 0;
}
