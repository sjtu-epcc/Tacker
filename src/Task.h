// Task.h
#pragma once
#include "Kernel.h"
#include <vector>
#include <queue>

enum ExecutionMode{
    Aker = 0,
    Tacker
};

enum TaskType{
    LC = 0,
    BE
};
class Task {
public:
    Task(int taskId, std::string taskName);
    Task(int taskId);
    ~Task();
    void addKernel(Kernel* kernel);
    void executeTask(ExecutionMode mode, cudaStream_t stream);
    void executeTask(ExecutionMode mode, int idx, cudaStream_t stream);

    virtual void initExecution() = 0;

    int taskId;
    std::string taskName;
    std::vector<Kernel*> kernels; // 容器一般不能存放引用

    TaskType taskType;
    
    void* Input[3];
    void** Result;
    void* InputHost[3];
    int InputSize[3];
};