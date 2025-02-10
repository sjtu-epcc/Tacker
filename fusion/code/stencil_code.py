stencil_header_code = """
#include <mma.h>
using namespace nvcuda; 
#include "header/stencil_header.h"
#include "kernel/stencil_kernel.cu"
"""

stencil_variables_code = """
    int stencil_blks = 3;
    int stencil_iter = 1;
    // stencil variables
    // ---------------------------------------------------------------------------------------
        float *host_stencil_ori_a0;
        float *stencil_ori_a0;
        float *stencil_ori_anext;

        float *host_stencil_ptb_a0;
        float *stencil_ptb_a0;
        float *stencil_ptb_anext;

        float *host_stencil_gptb_a0;
        float *stencil_gptb_a0;
        float *stencil_gptb_anext;

        float c0=1.0f/6.0f;
        float c1=1.0f/6.0f/6.0f;

        // nx = 128 ny = 128 nz = 32 iter = 100
        // nx = 512 ny = 512 nz = 64 iter = 100
        int nx = 128 * 4;
        int ny = 128 * 4;
        int nz = 32 * 2;
        // int nz = 16 * 1;
        
        // printf("nx: %d, ny: %d, nz: %d, iteration: %d \\n", nx, ny, nz, iteration);
        host_stencil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_anext, nx * ny * nz * sizeof(float)));

        host_stencil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_anext, nx * ny * nz * sizeof(float)));

        host_stencil_gptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_gptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_gptb_anext, nx * ny * nz * sizeof(float)));

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, stencil_ori_a0, nx * ny * nz));
        cudaErrCheck(cudaMemcpy(stencil_ori_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_gptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_gptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------
"""

stencil_solo_running_code = """
    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 stencil_grid, ori_stencil_grid;
        dim3 stencil_block, ori_stencil_block;
        stencil_block.x = tile_x;
        stencil_block.y = tile_y;
        stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
        stencil_grid.y = (ny + tile_y - 1) / tile_y;
        ori_stencil_block = stencil_block;
        ori_stencil_grid = stencil_grid;

        printf("[ORI] Running with stencil...\\n");
        printf("[ORI] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \\n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ori_a0, stencil_ori_anext, nx, ny, nz, stencil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] stencil took %f ms\\n\\n", kernel_time);
    // ---------------------------------------------------------------------------------------
"""

stencil_gptb_params_list = """c0, c1, stencil_gptb_a0, stencil_gptb_anext, nx, ny, nz,
    ori_stencil_grid.x, ori_stencil_grid.y, ori_stencil_grid.z, ori_stencil_block.x, ori_stencil_block.y, ori_stencil_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_stencil_grid.x * ori_stencil_grid.y * ori_stencil_grid.z
"""

stencil_gptb_params_list_new = """c0, c1, stencil_gptb_a0, stencil_gptb_anext, nx, ny, nz,
    ori_stencil_grid.x, ori_stencil_grid.y, ori_stencil_grid.z, ori_stencil_block.x, ori_stencil_block.y, ori_stencil_block.z,
    start_blk_no, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, end_blk_no, 0"""

def get_stencil_header_code():
    return stencil_header_code

def get_stencil_code_before_mix_kernel():
    return stencil_variables_code + stencil_solo_running_code

def get_stencil_code_after_mix_kernel():
    return ""
