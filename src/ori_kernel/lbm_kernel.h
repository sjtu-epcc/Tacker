//lbm_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriLBMParamsStruct {
    float* src;
    float* dst;
};

class OriLBMKernel : public Kernel {
public:
    // 构造函数
    OriLBMKernel(int id);

    // 析构函数
    ~OriLBMKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel(){};  
    OriLBMParamsStruct* LBMKernelParams;
    
};