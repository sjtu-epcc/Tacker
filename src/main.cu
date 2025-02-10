// main.cc
#include "header/pets_common.h"
#include "TackerConfig.h"
#include "util.h"
#include "TaskManager.h"
#include "Task.h"
#include "Kernel.h"
#include "Logger.h"
#include "ModuleCenter.h"
#include "Recorder.h"
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
#include "./include/clipp.h"


#include "ori_kernel/cp_kernel.cu"
#include "ori_kernel/cutcp_kernel.cu"
#include "ori_kernel/fft_kernel.cu"
#include "ori_kernel/lbm_kernel.cu"
#include "ori_kernel/mrif_kernel.cu"
#include "ori_kernel/mriq_kernel.cu"
#include "ori_kernel/sgemm_kernel.cu"
#include "ori_kernel/stencil_kernel.cu"

#include "ori_kernel/lava_kernel.cu"
#include "ori_kernel/hot3d_kernel.cu"
#include "ori_kernel/nn_kernel.cu"
#include "ori_kernel/path_kernel.cu"

#include "GPTBKernel.h"
#include "MixKernel.h"

#include <unordered_map>
#include "gptb_kernel/cp_kernel.cu"
#include "gptb_kernel/cutcp_kernel.cu"
#include "gptb_kernel/fft_kernel.cu"
#include "gptb_kernel/lbm_kernel.cu"
#include "gptb_kernel/mrif_kernel.cu"
#include "gptb_kernel/mriq_kernel.cu"
#include "gptb_kernel/sgemm_kernel.cu"
#include "gptb_kernel/stencil_kernel.cu"

#include "gptb_kernel/lava_kernel.cu"
#include "gptb_kernel/hot3d_kernel.cu"
#include "gptb_kernel/nn_kernel.cu"
#include "gptb_kernel/path_kernel.cu"

#include "mix_kernel/cp_fft_3_1.cu"
#include "mix_kernel/cp_sgemm_1_1.cu"
#include "mix_kernel/cutcp_fft_1_1.cu"
#include "mix_kernel/cutcp_sgemm_1_1.cu"
#include "mix_kernel/fft_lbm_6_1.cu"
#include "mix_kernel/fft_mriq_3_2.cu"
#include "mix_kernel/fft_sgemm_1_4.cu"
#include "mix_kernel/fft_stencil_5_3.cu"
#include "mix_kernel/lbm_mrif_1_3.cu"
#include "mix_kernel/lbm_mriq_1_2.cu"
#include "mix_kernel/lbm_sgemm_1_7.cu"
#include "mix_kernel/mrif_sgemm_1_4.cu"
#include "mix_kernel/mrif_stencil_3_2.cu"
#include "mix_kernel/mriq_sgemm_1_2.cu"
#include "mix_kernel/hot3d_lava.cu"
#include "mix_kernel/hot3d_nn.cu"
#include "mix_kernel/hot3d_path.cu"
#include "mix_kernel/lava_nn.cu"
#include "mix_kernel/lava_path.cu"
#include "mix_kernel/nn_path.cu"

// dnn
#include "dnn/resnet50/resnet50.h"
#include "dnn/bert/bert.h"
#include "dnn/inception3/inception3.h"
#include "dnn/vgg11/vgg11.h"
#include "dnn/vgg16/vgg16.h"
#include "dnn/vit/vit.h"

#include "gptb_kernel/tzgemm_kernel.cu"
#include "tzgemm_kernel.h"
#include <cublas_v2.h>

// tzgemm mix
#include "mix_kernel/tzgemm_cp.cu"
#include "mix_kernel/tzgemm_cutcp.cu"
#include "mix_kernel/tzgemm_fft.cu"
#include "mix_kernel/tzgemm_lbm.cu"
#include "mix_kernel/tzgemm_mrif.cu"
#include "mix_kernel/tzgemm_mriq.cu"
#include "mix_kernel/tzgemm_sgemm.cu"
#include "mix_kernel/tzgemm_stencil.cu"
#include "mix_kernel/tzgemm_lava.cu"
#include "mix_kernel/tzgemm_hot3d.cu"
#include "mix_kernel/tzgemm_nn.cu"
#include "mix_kernel/tzgemm_path.cu"

#include "json.h"
#include "Creator.h"


std::unordered_set<int> gemm_ks;

Logger logger(LOG_FILE_PATH, true, true);

ModuleCenter moduleCenter;

Recorder recorder;

std::unordered_map<std::string, void*> fmap = {
    {"gptb_cp", (void*)g_general_ptb_cp},
    {"gptb_cutcp", (void*)general_ptb_cutcp},
    {"gptb_fft", (void*)g_general_ptb_fft},
    {"gptb_lbm", (void*)general_ptb_lbm},
    {"gptb_mrif", (void*)g_general_ptb_mrif},
    {"gptb_mriq", (void*)g_general_ptb_mriq},
    {"gptb_sgemm", (void*)general_ptb_sgemm},
    {"gptb_stencil", (void*)general_ptb_stencil},
    {"gptb_tzgemm", (void*)general_ptb_tzgemm},
    {"gptb_cp_int", (void*)g_general_ptb_cp_int},
    {"gptb_fft_int", (void*)g_general_ptb_fft_int},
    {"gptb_mrif_int", (void*)g_general_ptb_mrif_int},
    {"gptb_mriq_int", (void*)g_general_ptb_mriq_int},
    {"gptb_lava", (void*)general_ptb_lava},
    {"gptb_hot3d", (void*)general_ptb_hot3d},
    {"gptb_nn", (void*)general_ptb_nn},
    {"gptb_path", (void*)general_ptb_path},
    {"cp_fft", (void*)mixed_cp_fft_kernel_3_1},
    {"cp_sgemm", (void*)mixed_cp_sgemm_kernel_1_1},
    {"fft_lbm", (void*)mixed_fft_lbm_kernel_6_1},
    {"fft_mriq", (void*)mixed_fft_mriq_kernel_3_2},
    {"fft_sgemm", (void*)mixed_fft_sgemm_kernel_1_4},
    {"fft_stencil", (void*)mixed_fft_stencil_kernel_5_3},
    {"lbm_mrif", (void*)mixed_lbm_mrif_kernel_1_3},
    {"lbm_mriq", (void*)mixed_lbm_mriq_kernel_1_2},
    {"lbm_sgemm", (void*)mixed_lbm_sgemm_kernel_1_7},
    {"mrif_sgemm", (void*)mixed_mrif_sgemm_kernel_1_4},
    {"mrif_stencil", (void*)mixed_mrif_stencil_kernel_3_2},
    {"mriq_sgemm", (void*)mixed_mriq_sgemm_kernel_1_2},
    {"cutcp_fft", (void*)mixed_cutcp_fft_kernel_1_1},
    {"cutcp_sgemm", (void*)mixed_cutcp_sgemm_kernel_1_1},
    {"hot3d_lava", (void*)mixed_hot3d_lava_kernel},
    {"hot3d_nn", (void*)mixed_hot3d_nn_kernel},
    {"hot3d_path", (void*)mixed_hot3d_path_kernel},
    {"lava_nn", (void*)mixed_lava_nn_kernel},
    {"lava_path", (void*)mixed_lava_path_kernel},
    {"nn_path", (void*)mixed_nn_path_kernel},
    {"tzgemm_cp", (void*)cp_tzgemm_mix},
    {"tzgemm_cutcp", (void*)cutcp_tzgemm_mix}, 
    {"tzgemm_fft", (void*)fft_tzgemm_mix},
    {"tzgemm_lbm", (void*)lbm_tzgemm_mix},
    {"tzgemm_mrif", (void*)mrif_tzgemm_mix},
    {"tzgemm_mriq", (void*)mriq_tzgemm_mix},
    {"tzgemm_sgemm", (void*)sgemm_tzgemm_mix},
    {"tzgemm_stencil", (void*)stencil_tzgemm_mix},
    {"tzgemm_lava", (void*)lava_tzgemm_mix},
    {"tzgemm_hot3d", (void*)hot3d_tzgemm_mix},
    {"tzgemm_nn", (void*)nn_tzgemm_mix},
    {"tzgemm_path", (void*)path_tzgemm_mix},
    {"tzgemm_cp_int", (void*)cp_tzgemm_mix_int},
};

void compileInfo() {
    std::cout << "Acker Version: " + std::to_string(Tacker_VERSION_MAJOR) + "." + std::to_string(Tacker_VERSION_MINOR) + "." + std::to_string(Tacker_VERSION_PATCH) << std::endl;
    std::cout << "Compile Timestamp: " + std::string(COMPILE_TIMESTAMP) << std::endl;
}

void initCUDA(int device=0) {
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        logger.ERROR("No CUDA-compatible devices found.");
        exit(EXIT_FAILURE);
    }
    logger.INFO("CUDA device count: " + std::to_string(deviceCount));

    CUDA_SAFE_CALL(cudaSetDevice(device));

    CUDA_SAFE_CALL(cudaDeviceReset()); // Reset device state
    CUDA_SAFE_CALL(cudaSetDevice(device)); // Set the current device

    CUDA_SAFE_CALL(cudaFree(0)); // Create a CUDA context

    logger.INFO("CUDA init complete!");
}

void printDeviceProp() {
    // uses cuda runtime API
    int SMnum, blocknum, threads, warp, kernel, overlap, sharedmemory;
    cudaDeviceGetAttribute(&SMnum, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&blocknum, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, 0);
    cudaDeviceGetAttribute(&sharedmemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&kernel, cudaDevAttrConcurrentKernels, 0);
    cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, 0);

    std::cout << "SM num\t\t\t" + std::to_string(SMnum) << std::endl;
    std::cout << "max block num per sm\t" + std::to_string(blocknum) << std::endl;
    std::cout << "max threads per blk\t" + std::to_string(threads) << std::endl;
    std::cout << "warp size\t\t" + std::to_string(warp) << std::endl;
    std::cout << "shared memory\t\t" + std::to_string(sharedmemory) << std::endl;
    std::cout << "concurrent kernels\t" + std::to_string(kernel) << std::endl;
    std::cout << "overlap\t\t\t" + std::to_string(overlap) << std::endl;

}

void my_exit() {
    recorder.text();
    logger.INFO("System exit");
}

std::string SYSTEM = "aker";
std::string ROOT_PATH = "/workspace/tacker/src";
std::string CONFIG_PATH = ROOT_PATH + "/config";
std::string MODEL_NAME = "none";

extern float* ori_wmma_results1;
extern float* ori_wmma_results2;
#ifdef AKER_INT8
extern int16_t* ori_wmma_C;
#else
extern float* ori_wmma_C;
#endif
extern float* cublas_wmma_C;


Task* createTask(std::string taskName) {
    if (taskName == "resnet50") {
        return new Resnet50(1000);
    } else if (taskName == "bert") {
        return new Bert(1000);
    } else if (taskName == "inception3") {
        return new Inception3(1001);
    } else if (taskName == "vgg11") {
        return new VGG11(1000);
    } else if (taskName == "vgg16") {
        return new VGG16(1000);
    } else if (taskName == "vit") {
        return new ViT(1000);
    } else {
        logger.ERROR("Task name not found");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    using namespace clipp;

    int device_no = 0;

    auto cli = (
        required("-s", "--system") & clipp::value("system_name", SYSTEM).doc("system name, aker/tacker"),
        required("-m", "--model") & clipp::value("model_name", MODEL_NAME).doc("model name"),
        option("-d", "--device") & clipp::value("device_no", device_no).doc("device number")
    );

    if(!parse(argc, argv, cli)) {
        std::cout << make_man_page(cli, argv[0]);
        return 0;
    } else {
        std::cout << "system: " << SYSTEM << ", model: " << MODEL_NAME << std::endl;
    }

    atexit (my_exit);

    read_json(CONFIG_PATH + "/kinfo-" + MODEL_NAME + ".json");
    read_common_json(CONFIG_PATH + "/kinfo-common.json");
    // read_json(ROOT_PATH + "/kinfo.json");

    initCUDA(device_no);
    // Print compile info
    compileInfo();

    // Print device properties
    printDeviceProp();
    // system("nvidia-smi > nvidia-smi.log");

    // profile area
    cudaEvent_t startKERNEL, stopKERNEL;
	CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
	CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));
    float milliseconds = 0;

    cudaStream_t streams[2];
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[0]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[1]));

    // [Aker] throughput test fig15
    auto lc_task = createTask(MODEL_NAME);
    for (int i = 0; i < 5; ++i) {
        lc_task->initExecution();
        for (auto& kernel: lc_task->kernels) {
            // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
            kernel->execute(nullptr);
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    std::string a = sget_kernel_info("throughput_test", "a");
    std::string b = sget_kernel_info("throughput_test", "b");
    printf("[Result] cd1: %s, cd2: %s, dnn: %s\n", a.c_str(), b.c_str(), MODEL_NAME.c_str());
    TaskManager taskManager(lc_task, a, b);
    
    taskManager.executeAllTasks(ExecutionMode::Aker, streams[0]);
    taskManager.executeAllTasks(ExecutionMode::Tacker, streams[1]);

    // [Aker] throughput test(1:1 version)
    // auto lc_task = createTask(MODEL_NAME);
    // for (int i = 0; i < 5; ++i) {
    //     lc_task->initExecution();
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     for (auto& kernel: lc_task->kernels) {
    //         if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //         kernel->execute(nullptr);
    //         CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     }
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // }
    // cudaDeviceSynchronize();
    // std::string a = sget_kernel_info("throughput_test", "a");
    // std::string b = sget_kernel_info("throughput_test", "b");
    // printf("[Result] cd: %s, dnn: %s\n", a.c_str(), MODEL_NAME.c_str());
    // TaskManager taskManager(lc_task, a, b);

    // printf("----float----\n");
    // // taskManager.execute_with_one_cd_kernel(ExecutionMode::Aker, streams[0]);
    // taskManager.execute_with_one_cd_kernel(ExecutionMode::Tacker, streams[0]);
    // taskManager.be_task1_name = a + "_int";
    // printf("taskManager.be_task1_name: %s\n", taskManager.be_task1_name.c_str());
    // printf("----int----\n");
    // taskManager.execute_with_one_cd_kernel(ExecutionMode::Tacker, streams[1]);



    return 0;
}
