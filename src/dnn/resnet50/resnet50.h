#pragma once
#include "Task.h"
#include "util.h"

class Resnet50: public Task {
public:
    // override constructor
    Resnet50(int taskId) : Task(taskId) {
        this->taskName = "Resnet50";
        this->taskId = taskId;
        initParams();
    }
    Resnet50(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        // printf("Resnet50 Input[0] = %p, InputHost[0] = %p, InputSize[0] = %d\n", Input[0], InputHost[0], InputSize[0]);
        CUDA_SAFE_CALL(cudaMemcpy((float*)Input[0], (float*)InputHost[0], sizeof(float) * InputSize[0], cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_270_0, float**  Result_505_0);
    void initParams();  
    int input_size = -1;

};