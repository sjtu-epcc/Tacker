#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/lstm/lstm.h"
#include "./dnn/lstm/lstm_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void LSTM::initParams() {
    cuda_init();
    //input argument
    float* Parameter_162_0_host, *Parameter_162_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_162_0_host, sizeof(float)* 25600));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_162_0, sizeof(float) * 25600));

    //output arguments
    float* Result_32346_0_host, *Result_32346_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_32346_0_host, sizeof(float) * 256));

    // fill input values
    for (int i = 0; i < 25600; ++i) Parameter_162_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_162_0, Parameter_162_0_host, sizeof(float) * 25600, cudaMemcpyHostToDevice));
    this->Parameter_162_0 = Parameter_162_0;
    this->Parameter_162_0_host = Parameter_162_0_host;
    this->Result_32346_0 = &Result_32346_0;

    // put all kernel
    this->gen_vector(Parameter_162_0, &Result_32346_0);
}