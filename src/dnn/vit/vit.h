#pragma once
#include "Task.h"
#include "util.h"

class ViT: public Task {
public:
    // override constructor
    ViT(int taskId) : Task(taskId) {
        this->taskName = "ViT";
        this->taskId = taskId;
        initParams();
    }
    ViT(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Input[0], InputHost[0], sizeof(float) * InputSize[0], cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_32_0, float**  Result_99_0);
    void initParams();  
    int input_size = -1;

};