#pragma once
#include "Task.h"
#include "util.h"

class Inception3: public Task {
public:
    // override constructor
    Inception3(int taskId) : Task(taskId) {
        this->taskName = "Inception3";
        this->taskId = taskId;
        initParams();
    }
    Inception3(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Input[0], InputHost[0], sizeof(float) * InputSize[0], cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_270_0, float**  Result_505_0);
    void initParams();  

};