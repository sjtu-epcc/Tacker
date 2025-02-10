//lava_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriHOT3DParamsStruct {
    float *p; 
    float* tIn; 
    float *tOut; 
    float sdc;
    int nx; 
    int ny; 
    int nz;
    float ce; 
    float cw; 
    float cn; 
    float cs;
    float ct; 
    float cb; 
    float cc;
};

class OriHOT3DKernel : public Kernel {
public:
    // 构造函数
    OriHOT3DKernel(int id);

    // 析构函数
    ~OriHOT3DKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel(){};  
    OriHOT3DParamsStruct* HOT3DKernelParams;
    
};