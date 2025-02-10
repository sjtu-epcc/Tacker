// cutcp_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"

struct OriCUTCPParamsStruct {
    int binDim_x;
    int binDim_y;
    float4 *binZeroAddr;    /* address of atom bins starting at origin */
    float h;                /* lattice spacing */
    float cutoff2;          /* square of cutoff distance */
    float inv_cutoff2;
    float *regionZeroAddr; /* address of lattice regions starting at origin */
    int zRegionIndex_t; // NOTICE
};

class OriCUTCPKernel : public Kernel {
public:
    // 构造函数
    OriCUTCPKernel(int id, const std::string& moduleName, const std::string& kernelName);
    OriCUTCPKernel(int id);

    // 析构函数
    ~OriCUTCPKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel();  
    OriCUTCPParamsStruct* CUTCPKernelParams;
    
};