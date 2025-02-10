// Kernel.h
#pragma once

#include "util.h"
#include <vector>

enum lmode {
    DRIVER_LAUNCH = 0,
    RUNTIME_LAUNCH
};

class Kernel {
public:
    template <typename... Args>
    void execute(Args&&... args) {
        executeImpl(std::forward<Args>(args)...);
    }
    virtual void initParams() = 0;
    virtual void loadKernel() = 0;

    virtual ~Kernel() = default; // 声明虚析构函数，以确保子类的析构函数被调用

    int getKernelId() { return Id; }
    std::string& getKernelName() { return kernelName; }

    virtual std::vector<int> getArgs() = 0;

    int Id;
    std::string kernelName;
    std::string moduleName;
    std::string funcKey;
    CUfunction function;
    unsigned int smem;
    std::vector<void*> kernelParams;
    std::vector<void*> cudaFreeList;
    dim3 launchGridDim;
    dim3 launchBlockDim;
        
    lmode launchMode;

    void* kernelFunc;

    uint mixable = 0;
    bool initialized = false;
protected:
    virtual void executeImpl(cudaStream_t stream) = 0; // implement in subclass
};

struct GPTBParams {
    int grid_dimension_x;
    int grid_dimension_y;
    int grid_dimension_z;
    int block_dimension_x;
    int block_dimension_y;
    int block_dimension_z;
    int ptb_start_block_pos;
    int ptb_iter_block_step;
    int ptb_end_block_pos;
};