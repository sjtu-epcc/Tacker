
#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/vgg16/vgg16.h"
#include "./dnn/vgg16/vgg16_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void VGG16::initParams() {
    vgg16_cuda_init();
    
    //input argument
    float* Parameter_0_0_host, *Parameter_0_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_0_0_host, sizeof(float)* 4816896));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_0_0, sizeof(float) * 4816896));
    for (int i = 0; i < 4816896; ++i) Parameter_0_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 4816896, cudaMemcpyHostToDevice));
    this->Input[0] = Parameter_0_0;
    this->InputHost[0] = Parameter_0_0_host;
    this->InputSize[0] = 4816896;

    //output arguments
    float* Result_144_0_host, *Result_144_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_144_0_host, sizeof(float) * 32032));

    this->Result = (void**)&Result_144_0;

    //fill input values
    this->gen_vector(Parameter_0_0, (float**)Result);
}