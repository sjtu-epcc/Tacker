
#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/vit/vit.h"
#include "./dnn/vit/vit_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void ViT::initParams() {
    vit_cuda_init();
    
    //input argument
    float* Parameter_207_0_host, *Parameter_207_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_207_0_host, sizeof(float)* 7225344));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_207_0, sizeof(float) * 7225344));
    for (int i = 0; i < 7225344; ++i) Parameter_207_0_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_207_0, Parameter_207_0_host, sizeof(float) * 7225344, cudaMemcpyHostToDevice));
    this->Input[0] = Parameter_207_0;
    this->InputHost[0] = Parameter_207_0_host;
    this->InputSize[0] = 7225344;

    //output arguments
    float* Result_1527_0_host, *Result_1527_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_1527_0_host, sizeof(float) * 7262208));

    this->Result = (void**)&Result_1527_0;

    //fill input values
    this->gen_vector(Parameter_207_0, (float**)Result);
}