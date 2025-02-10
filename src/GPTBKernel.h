// GPTBKernel.h
#pragma once
#include "Kernel.h"
#include "Logger.h"
#include "ModuleCenter.h"
#include <unordered_map>

extern Logger logger;
extern ModuleCenter moduleCenter;
extern std::unordered_map<std::string, void*> fmap;

class GPTBKernel : public Kernel {
public:
    GPTBKernel(int id, const std::string kernelName, const std::string funcKey, Kernel* kernel, dim3 gridDim, dim3 blockDim, int ptb_start_block_pos, int ptb_end_block_pos, int* kernelParams)  
    : kernel_(kernel), funcKey(funcKey) {
        this->kernelName = kernelName;
        this->launchGridDim = gridDim;
        this->launchBlockDim = blockDim;
        this->Id = id;
        // kernel
        gptbParams.grid_dimension_x = kernel_->launchGridDim.x;
        gptbParams.grid_dimension_y = kernel_->launchGridDim.y;
        gptbParams.grid_dimension_z = kernel_->launchGridDim.z;
        gptbParams.block_dimension_x = kernel_->launchBlockDim.x;
        gptbParams.block_dimension_y = kernel_->launchBlockDim.y;
        gptbParams.block_dimension_z = kernel_->launchBlockDim.z;

        // ptb
        gptbParams.ptb_iter_block_step = gridDim.x * gridDim.y * gridDim.z;
        gptbParams.ptb_start_block_pos = ptb_start_block_pos;
        gptbParams.ptb_end_block_pos = ptb_end_block_pos;

        this->kernelParams = kernel_->kernelParams;
        for (int i = 0; i < 20; i++) {
            this->kernelParams.push_back((void*)&kernelParams[i]);
        }

        this->smem = kernel_->smem;
        // logger.INFO("kernelParams size: " + std::to_string(kernelParams.size()));
        loadKernel();
        initParams();
        // printf("gptb kernel class - %s created\n", this->kernelName.c_str());
    }

    GPTBKernel(int id, const std::string kernelName, const std::string funcKey, Kernel* kernel, dim3 gridDim, dim3 blockDim, int ptb_start_block_pos, int ptb_end_block_pos) 
    : kernel_(kernel), funcKey(funcKey) {
        this->kernelName = kernelName;
        this->launchGridDim = gridDim;
        this->launchBlockDim = blockDim;
        this->Id = id;
        // kernel
        gptbParams.grid_dimension_x = kernel_->launchGridDim.x;
        gptbParams.grid_dimension_y = kernel_->launchGridDim.y;
        gptbParams.grid_dimension_z = kernel_->launchGridDim.z;
        gptbParams.block_dimension_x = kernel_->launchBlockDim.x;
        gptbParams.block_dimension_y = kernel_->launchBlockDim.y;
        gptbParams.block_dimension_z = kernel_->launchBlockDim.z;

        // ptb
        gptbParams.ptb_iter_block_step = gridDim.x * gridDim.y * gridDim.z;
        gptbParams.ptb_start_block_pos = ptb_start_block_pos;
        gptbParams.ptb_end_block_pos = ptb_end_block_pos;

        this->kernelParams = kernel_->kernelParams;

        this->smem = kernel_->smem;
        // logger.INFO("kernelParams size: " + std::to_string(kernelParams.size()));
        loadKernel();
        initParams();
        // printf("gptb kernel class - %s created\n", this->kernelName.c_str());
    }

    ~GPTBKernel() {
        logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is destroyed!");
    }

    GPTBParams gptbParams;
    const std::string funcKey;
    Kernel* kernel_;

    void initParams() override{
        kernelParams.push_back(&gptbParams.grid_dimension_x);
        kernelParams.push_back(&gptbParams.grid_dimension_y);
        kernelParams.push_back(&gptbParams.grid_dimension_z);
        kernelParams.push_back(&gptbParams.block_dimension_x);
        kernelParams.push_back(&gptbParams.block_dimension_y);
        kernelParams.push_back(&gptbParams.block_dimension_z);
        // 方便跟mix kernel统一，不提前push后续ptb args
    }

    void loadKernel() override{
        if (fmap.find(funcKey) != fmap.end()) {
            this->kernelFunc = (void*)fmap[funcKey];
        } else {
            logger.ERROR("load kernel {" + funcKey + "} failed!");
        }
    }

    void executeImpl(cudaStream_t stream) override{
        // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
        gptbParams.ptb_iter_block_step = launchGridDim.x * launchGridDim.y * launchGridDim.z;
        kernelParams.push_back(&gptbParams.ptb_start_block_pos);
        kernelParams.push_back(&gptbParams.ptb_iter_block_step);
        kernelParams.push_back(&gptbParams.ptb_end_block_pos);
        int thread_base = 0;
        kernelParams.push_back(&thread_base); // for gptb only, no offset for thread
        // logger.INFO("smem: " + std::to_string(smem));
        // launch kernel
        CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
            launchGridDim, launchBlockDim,
            (void **)kernelParams.data(), 0, stream));
        
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        kernelParams.pop_back();
        kernelParams.pop_back();
        kernelParams.pop_back();
        kernelParams.pop_back();
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    std::vector<int> getArgs() {
        return std::vector<int>();
    }
};