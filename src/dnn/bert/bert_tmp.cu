
#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/bert/bert.h"
#include "./dnn/bert/bert_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void Bert::initParams() {
    bert_cuda_init();
    
    //input argument
    int32_t* Parameter_0_0_host, *Parameter_0_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_0_0_host, sizeof(int32_t)* 16384));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_0_0, sizeof(int32_t) * 16384));
    for (int i = 0; i < 16384; ++i) Parameter_0_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(int32_t) * 16384, cudaMemcpyHostToDevice));
    this->Input[0] = Parameter_0_0;
    this->InputHost[0] = Parameter_0_0_host;
    this->InputSize[0] = 16384;

    //input argument
    int32_t* Parameter_1_0_host, *Parameter_1_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_1_0_host, sizeof(int32_t)* 16384));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_1_0, sizeof(int32_t) * 16384));
    for (int i = 0; i < 16384; ++i) Parameter_1_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_1_0, Parameter_1_0_host, sizeof(int32_t) * 16384, cudaMemcpyHostToDevice));
    this->Input[1] = Parameter_1_0;
    this->InputHost[1] = Parameter_1_0_host;
    this->InputSize[1] = 16384;

    //input argument
    int32_t* Parameter_2_0_host, *Parameter_2_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_2_0_host, sizeof(int32_t)* 16384));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_2_0, sizeof(int32_t) * 16384));
    for (int i = 0; i < 16384; ++i) Parameter_2_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_2_0, Parameter_2_0_host, sizeof(int32_t) * 16384, cudaMemcpyHostToDevice));
    this->Input[2] = Parameter_2_0;
    this->InputHost[2] = Parameter_2_0_host;
    this->InputSize[2] = 16384;

    //output arguments
    float* Result_394_0_host, *Result_394_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_394_0_host, sizeof(float) * 128128));

    this->Result = (void**)&Result_394_0;

    //fill input values
    this->gen_vector(Parameter_0_0, Parameter_1_0, Parameter_2_0, (float**)Result);
}