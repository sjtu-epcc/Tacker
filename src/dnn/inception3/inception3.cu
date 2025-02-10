
#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/inception3/inception3.h"
#include "./dnn/inception3/inception3_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void Inception3::initParams() {
    inception3_cuda_init();
    
    //input argument
    float* Parameter_0_0_host, *Parameter_0_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_0_0_host, sizeof(float)* 17164992));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_0_0, sizeof(float) * 17164992));
    for (int i = 0; i < 17164992; ++i) Parameter_0_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 17164992, cudaMemcpyHostToDevice));
    this->Input[0] = Parameter_0_0;
    this->InputHost[0] = Parameter_0_0_host;
    this->InputSize[0] = 17164992;

    //output arguments
    float* Result_892_0_host, *Result_892_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_892_0_host, sizeof(float) * 64064));

    this->Result = (void**)&Result_892_0;

    //fill input values
    this->gen_vector(Parameter_0_0, (float**)Result);
}