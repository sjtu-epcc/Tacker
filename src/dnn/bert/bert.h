#pragma once
#include "Task.h"
#include "util.h"

class Bert: public Task {
public:
    // override constructor
    Bert(int taskId) : Task(taskId) {
        this->taskName = "Bert";
        this->taskId = taskId;
        initParams();
    }
    Bert(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Input[0], InputHost[0], sizeof(int32_t) * InputSize[0], cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Input[1], InputHost[1], sizeof(int32_t) * InputSize[1], cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Input[2], InputHost[2], sizeof(int32_t) * InputSize[2], cudaMemcpyHostToDevice));
    }
    void gen_vector(int32_t*  Parameter_964_0, int32_t*  Parameter_965_0, int32_t*  Parameter_966_0, float**  Result_3689_0);
    void initParams();  

};