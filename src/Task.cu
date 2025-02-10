// Task.cpp
#include "Task.h"
#include "Logger.h"
#include "Recorder.h"

extern Logger logger;
extern std::string SYSTEM;
extern Recorder recorder;

Task::Task(int taskId, std::string taskName) : taskId(taskId), taskName(taskName){}
Task::Task(int taskId) : taskId(taskId) {}

void Task::addKernel(Kernel* kernel) {
    kernels.emplace_back(kernel);
}

void Task::executeTask(ExecutionMode mode, cudaStream_t stream) {
    float kernel_time = 0.0f;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    for (auto &kernel : kernels) {
        logger.DEBUG("kernel name: " + kernel->kernelName + ", id: " + std::to_string(kernel->Id) + " is executing ...");
        // execute kernel
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
        kernel->execute(stream);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        
        if (mode == ExecutionMode::Aker) {
            recorder.recordKernel(taskId, kernel->Id, kernel->kernelName, kernel_time);
        }
        // logger.DEBUG("kernel name: " + kernel->getKernelName() + ", kernel time: " + std::to_string(kernel_time) + " ms");
        // ~kernel
    }
    if (mode == ExecutionMode::Aker) {
        recorder.recordTask(taskId, taskName);
    }
}

void Task::executeTask(ExecutionMode mode, int idx, cudaStream_t stream) {
    float kernel_time = 0.0f;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));
    auto& kernel = this->kernels[idx];
    logger.DEBUG("kernel name: " + kernel->kernelName + ", id: " + std::to_string(kernel->Id) + " is executing ...");
    // execute kernel
    CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, 0));
    kernel->execute(stream);
    CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    
    if (mode == ExecutionMode::Aker) {
        recorder.recordKernel(taskId, kernel->Id, kernel->kernelName, kernel_time);
    }
    // logger.DEBUG("kernel name: " + kernel->getKernelName() + ", kernel time: " + std::to_string(kernel_time) + " ms");
}

Task::~Task() {
    // std::vector<std::unique_ptr<Kernel>>().swap(kernels);
    logger.DEBUG("task name: " + taskName + ", id: " + std::to_string(taskId) + " is destroyed!");
}
