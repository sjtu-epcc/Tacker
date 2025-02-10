#pragma once
#include <mma.h>
using namespace nvcuda; 
#include "header/atom.h"
#include "cutcp_kernel.h"
#include "Logger.h"
#include "header/cutcp_header.h"
#include "util.h"
#include "TackerConfig.h"
#include "ModuleCenter.h"

__constant__ int NbrListLen;
__constant__ int3 NbrList[NBRLIST_MAXLEN];
// __constant__ int NbrListLen_gptb_cutcp;
// __constant__ int3 NbrList_gptb_cutcp[NBRLIST_MAXLEN];
// __constant__ int NbrListLen_mix_cutcp;
// __constant__ int3 NbrList_mix_cutcp[NBRLIST_MAXLEN];

extern "C" __global__ void ori_cutcp(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex_t
    )
{
	__shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0)
    //     printf("%d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

	const int xRegionIndex = blockIdx.x;
	const int yRegionIndex = blockIdx.y;
    const int zRegionIndex = blockIdx.z;
	/* thread id */
	const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

        /* neighbor index */
        int nbrid;

        /* this is the start of the sub-region indexed by tid */
        myRegionAddr = regionZeroAddr + ((zRegionIndex*gridDim.y + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;

        /* spatial coordinate of this lattice point */
        float x = (8 * xRegionIndex + threadIdx.x) * h;
        float y = (8 * yRegionIndex + threadIdx.y) * h;
        float z = (8 * zRegionIndex + threadIdx.z) * h;

        int totalbins = 0;
        int numbins;

        /* bin number determined by center of region */
        myBinIndex.x = (int) floorf((8 * xRegionIndex + 4) * h * BIN_INVLEN);
        myBinIndex.y = (int) floorf((8 * yRegionIndex + 4) * h * BIN_INVLEN);
        myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

        /* first neighbor in list for me to cache */
        nbrid = (tid >> 4);

        numbins = BIN_CACHE_MAXLEN;

        float energy0 = 0.f;
        float energy1 = 0.f;
        float energy2 = 0.f;
        float energy3 = 0.f;


        // printf("NbrListLen: %d\n", NbrListLen);
        for (totalbins = 0;  totalbins < NbrListLen;  totalbins += numbins) {
            int bincnt;

            /* start of where to write in shared memory */
            int startoff = BIN_SIZE * (tid >> 4);

            /* each half-warp to cache up to 4 atom bins */
            for (bincnt = 0;  bincnt < 4 && nbrid < NbrListLen;  bincnt++, nbrid += 8) {
                int i = myBinIndex.x + NbrList[nbrid].x;
                int j = myBinIndex.y + NbrList[nbrid].y;
                int k = myBinIndex.z + NbrList[nbrid].z;

                /* determine global memory location of atom bin */
                float *p_global = ((float *) binZeroAddr) + (((__mul24(k, binDim_y) + j)*binDim_x + i) << BIN_SHIFT);

                /* coalesced read from global memory -
                * retain same ordering in shared memory for now */
                int binIndex = startoff + (bincnt << (3 + BIN_SHIFT));
                int tidmask = tid & 15;

                AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
                AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
            }
            __syncthreads();

            /* no warp divergence */
            if (totalbins + BIN_CACHE_MAXLEN > NbrListLen) {
                numbins = NbrListLen - totalbins;
            }

            int stopbin = (numbins << BIN_SHIFT);
            for (bincnt = 0; bincnt < stopbin; bincnt+=BIN_SIZE) {
                for (int i = 0;  i < BIN_DEPTH;  i++) {
                    int off = bincnt + (i<<2);

                    float aq = AtomBinCache[off + 3];
                    if (0.f == aq) 
                        break;  /* no more atoms in bin */

                    float dx = AtomBinCache[off    ] - x;
                    float dz = AtomBinCache[off + 2] - z;
                    float dxdz2 = dx*dx + dz*dz;
                    float dy = AtomBinCache[off + 1] - y;
                    float r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy0 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy1 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy2 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy3 += aq * rsqrtf(r2) * s * s;
                    }
                } /* end loop over atoms in bin */
            } /* end loop over cached atom bins */
            __syncthreads();
            // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //     printf("numbins: %d\n", numbins);
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
}


extern Logger logger;
extern ModuleCenter moduleCenter;

// 构造函数
OriCUTCPKernel::OriCUTCPKernel(int id, const std::string& moduleName, const std::string& kernelName) {
    Id = id;
    this->kernelName = kernelName;
    this->moduleName = moduleName;
    // loadKernel();
    initParams();
}

OriCUTCPKernel::OriCUTCPKernel(int id){
    Id = id;
    this->kernelName = "cutcp";
    // loadKernel();
    initParams();
}

void OriCUTCPKernel::initParams() {
    int cutcp_blks = 6;
    int cutcp_iter = 1;
    Atoms *atom;
    LatticeDim lattice_dim;
    Lattice *gpu_lattice;
    Vec3 min_ext, max_ext;	    /* Bounding box of atoms */
    Vec3 lo, hi;			    /* Bounding box with padding  */
    float h = 0.5f;		        /* Lattice spacing */
    float cutoff = 12.f;		/* Cutoff radius */
    float padding = 0.5f;		/* Bounding box padding distance */
    
    const char *pqrfilename = "/home/jxdeng/workspace/tacker/0_mybench/file_t/cutcp_input.pqr";
    if (!(atom = read_atom_file(pqrfilename))) {
        logger.ERROR("read_atom_file() failed");
        exit(EXIT_FAILURE);
    }
    get_atom_extent(&min_ext, &max_ext, atom);
    lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
    hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
    lattice_dim = lattice_from_bounding_box(lo, hi, h);
    gpu_lattice = create_lattice(lattice_dim);

    float4 *binBaseAddr;
    int3 *nbrlist;
    nbrlist = (int3 *)malloc(NBRLIST_MAXLEN * sizeof(int3));
    int nbins = 32768;
    binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
    prepare_input(gpu_lattice, cutoff, atom, binBaseAddr, nbrlist);

    int nbrlistlen = 256;
    float *cutcp_ori_regionZeroCuda, *host_cutcp_ori_regionZeroCuda;
    float4 *cutcp_ori_binBaseCuda, *cutcp_ori_binZeroCuda;
    // logger.INFO("cutcp_ori_regionZeroCuda: " + std::to_string((long long)(cutcp_ori_regionZeroCuda)));
    // logger.INFO("cutcp_ori_binZeroCuda: " + std::to_string((long long)(cutcp_ori_binZeroCuda)));


    int lnx = 208;
    int lny = 208;
    int lnz = 208;
    int lnall = lnx * lny * lnz;

    int xRegionDim = 26;
    int yRegionDim = 26;
    int zRegionDim = 26;
    int binDim_x = 32;
    int binDim_y = 32;
    float cutoff2 = 144.0;
    float inv_cutoff2 = 0.006944;

    CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ori_regionZeroCuda, lnall * sizeof(float)));
    cudaFreeList.push_back(cutcp_ori_regionZeroCuda);
    CUDA_SAFE_CALL(cudaMemset(cutcp_ori_regionZeroCuda, 0, lnall * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    cudaFreeList.push_back(cutcp_ori_binBaseCuda);

    CUDA_SAFE_CALL(cudaMemcpy(cutcp_ori_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
        cudaMemcpyHostToDevice));


    cutcp_ori_binZeroCuda = cutcp_ori_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrListLen_gptb_cutcp, &nbrlistlen, sizeof(int), 0));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrList_gptb_cutcp, nbrlist, nbrlistlen * sizeof(int3), 0));

    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrListLen_mix_cutcp, &nbrlistlen, sizeof(int), 0));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrList_mix_cutcp, nbrlist, nbrlistlen * sizeof(int3), 0));

    dim3 cutcp_grid, cutcp_block;
    cutcp_grid.x = xRegionDim;
    cutcp_grid.y = yRegionDim;
    cutcp_grid.z = cutcp_iter * 2;
    cutcp_block.x = 8;
    cutcp_block.y = 2;
    cutcp_block.z = 8;

    int zRegionIndex_t = 25;

    this->smem = 4096;

    this->CUTCPKernelParams = new OriCUTCPParamsStruct();
    this->CUTCPKernelParams->binDim_x = binDim_x;
    // logger.INFO("binDim_x: " + std::to_string(binDim_x));
    this->CUTCPKernelParams->binDim_y = binDim_y;
    // logger.INFO("binDim_y: " + std::to_string(binDim_y));
    this->CUTCPKernelParams->binZeroAddr = cutcp_ori_binZeroCuda;
    // logger.INFO("cutcp_ori_binZeroCuda: " + std::to_string((long long)(cutcp_ori_binZeroCuda)));
    this->CUTCPKernelParams->h = h;
    // logger.INFO("h: " + std::to_string(h));
    this->CUTCPKernelParams->cutoff2 = cutoff2;
    // logger.INFO("cutoff2: " + std::to_string(cutoff2));
    this->CUTCPKernelParams->inv_cutoff2 = inv_cutoff2;
    // logger.INFO("inv_cutoff2: " + std::to_string(inv_cutoff2));
    this->CUTCPKernelParams->regionZeroAddr = cutcp_ori_regionZeroCuda;
    // logger.INFO("cutcp_ori_regionZeroCuda: " + std::to_string((long long)(cutcp_ori_regionZeroCuda)));
    this->CUTCPKernelParams->zRegionIndex_t = zRegionIndex_t;
    // logger.INFO("zRegionIndex_t: " + std::to_string(zRegionIndex_t));

    this->kernelParams.push_back(&(this->CUTCPKernelParams->binDim_x));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->binDim_y));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->binZeroAddr));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->h));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->cutoff2));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->inv_cutoff2));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->regionZeroAddr));
    this->kernelParams.push_back(&(this->CUTCPKernelParams->zRegionIndex_t));

    this->launchGridDim = cutcp_grid;
    this->launchBlockDim = cutcp_block;

    this->kernelFunc = (void*) ori_cutcp;

    // float kernel_time;
    // cudaEvent_t startKERNEL;
    // cudaEvent_t stopKERNEL;
    // CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    // CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    // binDim_x, binDim_y, cutcp_ori_binZeroCuda, h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, 1
    // void *launchargs[] = {
    //     (void *)&binDim_x,
    //     (void *)&binDim_y,
    //     (void *)&cutcp_ori_binZeroCuda,
    //     (void *)&h,
    //     (void *)&cutoff2,
    //     (void *)&inv_cutoff2,
    //     (void *)&cutcp_ori_regionZeroCuda,
    //     (void *)&zRegionIndex_t
    // };
    // CUmodule module_;
	// CUfunction function_;

    // char *module_file = (char *)"/home/jxdeng/home/jxdeng/workspace/tacker/runtime/cubins/ori_cutcp.cubin";
	// char *cdkernel_name = (char *)"ori_cutcp";
    // CU_SAFE_CALL(cuModuleLoad(&module_, module_file));
	// CU_SAFE_CALL(cuModuleGetFunction(&function_, module_, cdkernel_name));



    // printf("before launch\n");
    // std::cin.get();
    // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    // CU_SAFE_CALL(cuLaunchKernel(function_, 
    // cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, 
    // cutcp_block.x, cutcp_block.y, cutcp_block.z, 
    // 4096, NULL, launchargs, NULL));
    // //checkKernelErrors((ori_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, 1)));
    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("kernel time: %f ms\n", kernel_time);
    // std::cin.get();

    // free nbrlist
    // free(nbrlist);
}

OriCUTCPKernel::~OriCUTCPKernel() {
    // free gpu memory
    // for (auto &ptr : cudaFreeList) {
    //     CUDA_SAFE_CALL(cudaFree(ptr));
    // }

    // // free cpu heap memory
    // free(this->CUTCPKernelParams);
    
    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is destroyed!");
}

void OriCUTCPKernel::loadKernel() {
    
}

void OriCUTCPKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    //logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    //logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));

    // print kernelParams
    // logger.INFO("-- kernelParams: ");
    // logger.INFO("---- binDim_x: " + std::to_string(*(int*)this->kernelParams[0]));
    // logger.INFO("---- binDim_y: " + std::to_string(*(int*)this->kernelParams[1]));
    // logger.INFO("---- binZeroAddr: " + std::to_string((long long)(*(float4**)this->kernelParams[2])));
    // logger.INFO("---- h: " + std::to_string(*(float*)this->kernelParams[3]));
    // logger.INFO("---- cutoff2: " + std::to_string(*(float*)this->kernelParams[4]));
    // logger.INFO("---- inv_cutoff2: " + std::to_string(*(float*)this->kernelParams[5]));
    // logger.INFO("---- regionZeroAddr: " + std::to_string((long long)(*(float**)this->kernelParams[6])));
    // logger.INFO("---- zRegionIndex_t: " + std::to_string(*(int*)this->kernelParams[7]));

    
    // CU_SAFE_CALL(cuLaunchKernel(this->function, 
    // launchGridDim.x, launchGridDim.y, launchGridDim.z, 
    // launchBlockDim.x, launchBlockDim.y, launchBlockDim.z, 
    // this->smem, NULL, (void**)this->kernelParams.data(), NULL));

    // checkKernelErrors((ori_cutcp<<<this->launchGridDim, this->launchBlockDim>>>(this->CUTCPKernelParams->binDim_x, this->CUTCPKernelParams->binDim_y, this->CUTCPKernelParams->binZeroAddr, this->CUTCPKernelParams->h, this->CUTCPKernelParams->cutoff2, this->CUTCPKernelParams->inv_cutoff2, this->CUTCPKernelParams->regionZeroAddr, this->CUTCPKernelParams->zRegionIndex_t)));
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), 0, stream));
    
}