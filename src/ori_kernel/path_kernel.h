//path_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriPATHParamsStruct {
    int iteration; 
    int *gpuWall;
    int *gpuSrc;
    int *gpuResults;
    int cols; 
    int rows;
    int startStep;
    int border;
};

class OriPATHKernel : public Kernel {
public:
    // 构造函数
    OriPATHKernel(int id);

    // 析构函数
    ~OriPATHKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel(){};  
    OriPATHParamsStruct* PATHKernelParams;
    
};