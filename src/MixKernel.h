// MixKernel.h
#pragma once
#include "GPTBKernel.h"
#include "json.h"

extern Logger logger;
extern std::unordered_map<std::string, void*> fmap;

class MixKernel : public Kernel {
public:
    MixKernel(int id, const std::string& funcKey,
        GPTBKernel* kernel1, GPTBKernel* kernel2, dim3 gridDim, dim3 blockDim, 
        int ptb_start_block_pos1, int ptb_end_block_pos1, int ptb_start_block_pos2, int ptb_end_block_pos2)
        : kernel1(kernel1), kernel2(kernel2), funcKey(funcKey){
            this->Id = id;
            this->kernelName = funcKey;
            this->launchGridDim = gridDim;
            this->launchBlockDim = blockDim;
            // if (get_kernel_info(funcKey, "shared_memory") != JSON_NOT_FOUND) {
            //     this->smem = get_kernel_info(funcKey, "shared_memory");
            // }
            // else {
            //     this->smem = this->kernel1->smem + this->kernel2->smem;
            // }
            this->smem = 0;

            // override gptb params
            this->kernel1_iter_block_step = gridDim.x * gridDim.y * gridDim.z;
            this->kernel2_iter_block_step = kernel1_iter_block_step;

            this->kernel1_start_block_pos = ptb_start_block_pos1;
            this->kernel1_end_block_pos = ptb_end_block_pos1;
            this->kernel2_start_block_pos = ptb_start_block_pos2;
            this->kernel2_end_block_pos = ptb_end_block_pos2;

            initParams();
            loadKernel();
    }

    ~MixKernel() {
        // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
    }

    const std::string funcKey;

    int kernel1_start_block_pos;
    int kernel1_iter_block_step;
    int kernel1_end_block_pos;
    int kernel2_start_block_pos;
    int kernel2_iter_block_step;
    int kernel2_end_block_pos;

    // TODO extra kernel if divided need

    void initParams() override{
        // kernel1 args push
        for (auto &param : kernel1->kernelParams) {
            kernelParams.push_back(param);
        }
        // kernel overrided args push
        kernelParams.push_back(&kernel1_start_block_pos);
        kernelParams.push_back(&kernel1_iter_block_step);
        kernelParams.push_back(&kernel1_end_block_pos);

        // kernel2 args push
        for (auto &param : kernel2->kernelParams) {
            kernelParams.push_back(param);
        }
        // kernel overrided args push
        kernelParams.push_back(&kernel2_start_block_pos);
        kernelParams.push_back(&kernel2_iter_block_step);
        kernelParams.push_back(&kernel2_end_block_pos);
    }

    void loadKernel() override {
        auto iter = fmap.find(funcKey);
        if (iter != fmap.end()) {
            this->kernelFunc = (void*)iter->second;
        } else {
            logger.ERROR("load kernel {" + funcKey + "} failed!");
        }

        return ;
    }

    void executeImpl(cudaStream_t stream) override {
        // printf("mix kernel: %s, gridDim.x: %d, blockDim.x: %d\n", this->kernelName.c_str(), this->launchGridDim.x, this->launchBlockDim.x);
        // logger.INFO("name: " + this->kernelName);
        // // // logger.INFO("smem: " + std::to_string(this->smem) + "=" + std::to_string(this->kernel1->smem) + "+" + std::to_string(this->kernel2->smem));
        // logger.INFO("gridDim: " + std::to_string(this->launchGridDim.x) + " " + std::to_string(this->launchGridDim.y) + " " + std::to_string(this->launchGridDim.z));
        // logger.INFO("blockDim: " + std::to_string(this->launchBlockDim.x) + " " + std::to_string(this->launchBlockDim.y) + " " + std::to_string(this->launchBlockDim.z));
        // logger.INFO("kernelParams size: " + std::to_string(kernelParams.size()));
        // // for (auto &param : kernelParams) {
        // //     logger.INFO("param: " + std::to_string((long)param));
        // // }
        // logger.INFO("mix kernel1 blks range: " + std::to_string(kernel1_start_block_pos) + " - " + std::to_string(kernel1_end_block_pos));
        // logger.INFO("mix kernel1 blks step: " + std::to_string(kernel1_iter_block_step));
        // logger.INFO("mix kernel2 blks range: " + std::to_string(kernel2_start_block_pos) + " - " + std::to_string(kernel2_end_block_pos));
        // logger.INFO("mix kernel2 blks step: " + std::to_string(kernel2_iter_block_step));
        CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
            launchGridDim, launchBlockDim,
            (void **)kernelParams.data(), 0, stream));
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    GPTBKernel* kernel1;
    GPTBKernel* kernel2;

    GPTBKernel* extraKernel;
};
