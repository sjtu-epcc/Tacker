#include "stencil_kernel.h"
#include "Logger.h"
#include "header/stencil_header.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_stencil(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
    //thread coarsening along x direction
    const int i = blockIdx.x*blockDim.x*2+threadIdx.x;
    const int i2= blockIdx.x*blockDim.x*2+threadIdx.x+blockDim.x;
    const int j = blockIdx.y*blockDim.y+threadIdx.y;
    const int sh_id=threadIdx.x + threadIdx.y*blockDim.x*2;
    const int sh_id2=threadIdx.x +blockDim.x+ threadIdx.y*blockDim.x*2;

    //shared memeory
    // extern __shared__ float sh_A0[];
    __shared__ float sh_A0[tile_x * tile_y * 2];
    sh_A0[sh_id]=0.0f;
    sh_A0[sh_id2]=0.0f;
    __syncthreads();

    //get available region for load and store
    const bool w_region =  i>0 && j>0 &&(i<(nx-1)) &&(j<(ny-1)) ;
    const bool w_region2 =  j>0 &&(i2<nx-1) &&(j<ny-1) ;
    const bool x_l_bound = (threadIdx.x==0);
    const bool x_h_bound = ((threadIdx.x+blockDim.x)==(blockDim.x*2-1));
    const bool y_l_bound = (threadIdx.y==0);
    const bool y_h_bound = (threadIdx.y==(blockDim.y-1));

    //register for bottom and top planes
    //because of thread coarsening, we need to doulbe registers
    float bottom=0.0f,bottom2=0.0f,top=0.0f,top2=0.0f;

    //load data for bottom and current 
    if((i<nx) &&(j<ny))
    {
        bottom=A0[Index3D (nx, ny, i, j, 0)];
        sh_A0[sh_id]=A0[Index3D (nx, ny, i, j, 1)];
    }
    if((i2<nx) &&(j<ny))
    {
        bottom2=A0[Index3D (nx, ny, i2, j, 0)];
        sh_A0[sh_id2]=A0[Index3D (nx, ny, i2, j, 1)];
    }

    __syncthreads();
    
    for(int k=1;k<nz-1;k++)
    {

        float a_left_right,a_up,a_down;		
        
        //load required data on xy planes
        //if it on shared memory, load from shared memory
        //if not, load from global memory
        if((i<nx) &&(j<ny))
            top=A0[Index3D (nx, ny, i, j, k+1)];
            
        if(w_region)
        {
            a_up        =y_h_bound?A0[Index3D (nx, ny, i, j+1, k )]:sh_A0[sh_id+2*blockDim.x];
            a_down      =y_l_bound?A0[Index3D (nx, ny, i, j-1, k )]:sh_A0[sh_id-2*blockDim.x];
            a_left_right=x_l_bound?A0[Index3D (nx, ny, i-1, j, k )]:sh_A0[sh_id-1];
    
            Anext[Index3D (nx, ny, i, j, k)] = (top + bottom + a_up + a_down + sh_A0[sh_id+1] +a_left_right)*c1
                                        -  sh_A0[sh_id]*c0;		
        }
        
        //load another block 
        if((i2<nx) &&(j<ny))
            top2=A0[Index3D (nx, ny, i2, j, k+1)];
            
        if(w_region2)
        {
            a_up        =y_h_bound?A0[Index3D (nx, ny, i2, j+1, k )]:sh_A0[sh_id2+2*blockDim.x];
            a_down      =y_l_bound?A0[Index3D (nx, ny, i2, j-1, k )]:sh_A0[sh_id2-2*blockDim.x];
            a_left_right=x_h_bound?A0[Index3D (nx, ny, i2+1, j, k )]:sh_A0[sh_id2+1];

            Anext[Index3D (nx, ny, i2, j, k)] = (top2 + bottom2 + a_up + a_down + a_left_right +sh_A0[sh_id2-1])*c1
                                        -  sh_A0[sh_id2]*c0;
        }

        //swap data
        __syncthreads();
        bottom=sh_A0[sh_id];
        sh_A0[sh_id]=top;
        bottom2=sh_A0[sh_id2];
        sh_A0[sh_id2]=top2;
        __syncthreads();
    }
}

OriSTENCILKernel::OriSTENCILKernel(int id) {
    this->Id = id;
    this->kernelName = "stencil";
    initParams();
}

OriSTENCILKernel::~OriSTENCILKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->STENCILKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriSTENCILKernel::initParams() {
    int stencil_blks = 3;
    int stencil_iter = 1;

    // stencil variables
    // ---------------------------------------------------------------------------------------
        float *host_stencil_ori_a0;
        float *stencil_ori_a0;
        float *stencil_ori_anext;

        // float *host_stencil_ptb_a0;
        // float *stencil_ptb_a0;
        // float *stencil_ptb_anext;

        // float *host_stencil_gptb_a0;
        // float *stencil_gptb_a0;
        // float *stencil_gptb_anext;

        float c0=1.0f/6.0f;
        float c1=1.0f/6.0f/6.0f;

        // nx = 128 ny = 128 nz = 32 iter = 100
        // nx = 512 ny = 512 nz = 64 iter = 100
        int nx = 128 * 4;
        int ny = 128 * 4;
        int nz = 32 * 2 * 3; // scale up mem by 3x
        // int nz = 16 * 1;
        
        // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
        host_stencil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_anext, nx * ny * nz * sizeof(float)));

        // host_stencil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        // cudaErrCheck(cudaMalloc((void**)&stencil_ptb_a0, nx * ny * nz * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&stencil_ptb_anext, nx * ny * nz * sizeof(float)));

        // host_stencil_gptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        // cudaErrCheck(cudaMalloc((void**)&stencil_gptb_a0, nx * ny * nz * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&stencil_gptb_anext, nx * ny * nz * sizeof(float)));

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, stencil_ori_a0, nx * ny * nz));
        cudaErrCheck(cudaMemcpy(stencil_ori_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(stencil_ptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(stencil_ptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(stencil_gptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaErrCheck(cudaMemcpy(stencil_gptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------
    nz /= 3;
    
    dim3 stencil_grid, ori_stencil_grid;
    dim3 stencil_block, ori_stencil_block;
    stencil_block.x = tile_x;
    stencil_block.y = tile_y;
    stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
    stencil_grid.y = (ny + tile_y - 1) / tile_y;
    ori_stencil_block = stencil_block;
    ori_stencil_grid = stencil_grid;

    this->launchGridDim = stencil_grid;
    this->launchBlockDim = stencil_block;

    // c0, c1, stencil_ori_a0, stencil_ori_anext, nx, ny, nz
    this->STENCILKernelParams = new OriSTENCILParamsStruct();
    this->STENCILKernelParams->c0 = c0;
    this->STENCILKernelParams->c1 = c1;
    this->STENCILKernelParams->A0 = stencil_ori_a0;
    this->STENCILKernelParams->Anext = stencil_ori_anext;
    this->STENCILKernelParams->nx = nx;
    this->STENCILKernelParams->ny = ny;
    this->STENCILKernelParams->nz = nz;

    this->kernelParams.push_back(&(this->STENCILKernelParams->c0));
    this->kernelParams.push_back(&(this->STENCILKernelParams->c1));
    this->kernelParams.push_back(&(this->STENCILKernelParams->A0));
    this->kernelParams.push_back(&(this->STENCILKernelParams->Anext));
    this->kernelParams.push_back(&(this->STENCILKernelParams->nx));
    this->kernelParams.push_back(&(this->STENCILKernelParams->ny));
    this->kernelParams.push_back(&(this->STENCILKernelParams->nz));

    this->smem = 1024;
    this->kernelFunc = (void*) ori_stencil;

}

void OriSTENCILKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, 0));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}