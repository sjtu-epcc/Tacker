#pragma once
#include "Task.h"
#include "util.h"

class LSTM: public Task {
public:
    // override constructor
    LSTM(int taskId) : Task(taskId) {
        this->taskName = "LSTM";
        this->taskId = taskId;
        initParams();
    }
    LSTM(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Input, InputHost, sizeof(float) * input_size, cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_162_0, float**  Result_32346_0);
    void initParams();  
    int input_size = -1;

};