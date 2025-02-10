#include "mrif_kernel.h"
#include "Logger.h"
#include "header/mrif_header.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

__constant__ __device__ mrif_kValues c[KERNEL_FH_K_ELEMS_PER_GRID];
__constant__ __device__ mrif_kValues_int c_int[KERNEL_FH_K_ELEMS_PER_GRID];
// __constant__ __device__ mrif_kValues c_gptb_mrif[KERNEL_FH_K_ELEMS_PER_GRID];
// __constant__ __device__ mrif_kValues c_mix_mrif[KERNEL_FH_K_ELEMS_PER_GRID];

// #include "gptb_kernel/mrif_kernel.cu"

void createDataStructs(int numK, int numX, 
                       float*& realRhoPhi, float*& imagRhoPhi, 
                       float*& outR, float*& outI) {
  realRhoPhi = (float* ) calloc(numK, sizeof(float));
  imagRhoPhi = (float* ) calloc(numK, sizeof(float));
  outR = (float*) calloc (numX, sizeof (float));
  outI = (float*) calloc (numX, sizeof (float));
}


void inputData(int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI,
               float** dR, float** dI) {
    int numK, numX;
    FILE* fid = fopen("/home/jxdeng/workspace/tacker/0_mybench/file_t/mrif_input.bin", "r");
    fread (&numK, sizeof (int), 1, fid);
    *_numK = numK;
    fread (&numX, sizeof (int), 1, fid);
    numX *= 2; // scale up by 2x
    *_numX = numX;
    *kx = (float *) memalign(16, numK * sizeof (float));
    fread (*kx, sizeof (float), numK, fid);
    *ky = (float *) memalign(16, numK * sizeof (float));
    fread (*ky, sizeof (float), numK, fid);
    *kz = (float *) memalign(16, numK * sizeof (float));
    fread (*kz, sizeof (float), numK, fid);
    *x = (float *) memalign(16, numX * sizeof (float));
    fread (*x, sizeof (float), numX, fid);
    *y = (float *) memalign(16, numX * sizeof (float));
    fread (*y, sizeof (float), numX, fid);
    *z = (float *) memalign(16, numX * sizeof (float));
    fread (*z, sizeof (float), numX, fid);
    *phiR = (float *) memalign(16, numK * sizeof (float));
    fread (*phiR, sizeof (float), numK, fid);
    *phiI = (float *) memalign(16, numK * sizeof (float));
    fread (*phiI, sizeof (float), numK, fid);
    *dR = (float *) memalign(16, numK * sizeof (float));
    fread (*dR, sizeof (float), numK, fid);
    *dI = (float *) memalign(16, numK * sizeof (float));
    fread (*dI, sizeof (float), numK, fid);
    fclose (fid); 
}


extern "C" __global__ void ComputeRhoPhiGPU(int numK,
        float* phiR, float* phiI, 
        float* dR, float* dI, 
        float* realRhoPhi, float* imagRhoPhi) {
    int indexK = blockIdx.x*KERNEL_RHO_PHI_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float rPhiR = phiR[indexK];
        float rPhiI = phiI[indexK];
        float rDR = dR[indexK];
        float rDI = dI[indexK];
        realRhoPhi[indexK] = rPhiR * rDR + rPhiI * rDI;
        imagRhoPhi[indexK] = rPhiR * rDI - rPhiI * rDR;
    }
}

extern "C" __global__ void ori_mrif(int numK, int kGlobalIndex,
        float* x, float* y, float* z, 
        float* outR, float* outI) {
    for (int FHGrid = 0; FHGrid < 1; FHGrid++) {
        kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;

        float sX;
        float sY;
        float sZ;
        float sOutR;
        float sOutI;
        // Determine the element of the X arrays computed by this thread
        int xIndex = blockIdx.x * KERNEL_FH_THREADS_PER_BLOCK + threadIdx.x;

        sX = x[xIndex];
        sY = y[xIndex];
        sZ = z[xIndex];
        sOutR = outR[xIndex];
        sOutI = outI[xIndex];

        // Loop over all elements of K in constant mem to compute a partial value
        // for X.
        int kIndex = 0;
        int kCnt = numK - kGlobalIndex;
        if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
            for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                float expArg = PIx2 *
                (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
            }
        }

        for (;(kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 4, kGlobalIndex += 4) {
            float expArg = PIx2 *
                (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
            float cosArg = cos(expArg);
            float sinArg = sin(expArg);
            sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
            sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

            int kIndex1 = kIndex + 1;
            float expArg1 = PIx2 *
                (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
            float cosArg1 = cos(expArg1);
            float sinArg1 = sin(expArg1);
            sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
            sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

            int kIndex2 = kIndex + 2;
            float expArg2 = PIx2 *
                (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
            float cosArg2 = cos(expArg2);
            float sinArg2 = sin(expArg2);
            sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
            sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

            int kIndex3 = kIndex + 3;
            float expArg3 = PIx2 *
                (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
            float cosArg3 = cos(expArg3);
            float sinArg3 = sin(expArg3);
            sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
            sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
        }

        outR[xIndex] = sOutR;
        outI[xIndex] = sOutI;
    }
}

OriMRIFKernel::OriMRIFKernel(int id){
    Id = id;
    this->kernelName = "mrif";
    initParams();
}

OriMRIFKernel::~OriMRIFKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->MRIFKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriMRIFKernel::initParams() {
        // mrif variables
    // ---------------------------------------------------------------------------------------
        int mrif_blks = 3;
        int mrif_iter = 1;
        int mrif_numX, mrif_numK;		                /* Number of X and K values */
        int original_numK;		            /* Number of K values in input file */
        float *mrif_base_kx, *mrif_base_ky, *mrif_base_kz;		        /* K trajectory (3D vectors) */
        float *mrif_base_x, *mrif_base_y, *mrif_base_z;		            /* X coordinates (3D vectors) */
        float *mrif_base_phiR, *mrif_base_phiI;		            /* Phi values (complex) */
        float *mrif_base_dR, *mrif_base_dI;		                /* D values (complex) */
        float *mrif_base_realRhoPhi, *mrif_base_imagRhoPhi;     /* RhoPhi values (complex) */
        mrif_kValues* mrif_kVals;		                /* Copy of X and RhoPhi.  Its
                                            * data layout has better cache
                                            * performance. */
        inputData(
            &original_numK, &mrif_numX,
            &mrif_base_kx, &mrif_base_ky, &mrif_base_kz,
            &mrif_base_x, &mrif_base_y, &mrif_base_z,
            &mrif_base_phiR, &mrif_base_phiI,
            &mrif_base_dR, &mrif_base_dI);
        mrif_numK = original_numK;

        // createDataStructs(mrif_numK, mrif_numX, mrif_base_realRhoPhi, mrif_base_imagRhoPhi, base_outR, base_outI);
        mrif_kVals = (mrif_kValues *)calloc(mrif_numK, sizeof (mrif_kValues));
        mrif_base_realRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));
        mrif_base_imagRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));

        // kernel 1
        float *ori_phiR, *ori_phiI;
        float *ori_dR, *ori_dI;
        float *ori_realRhoPhi, *ori_imagRhoPhi;
        // kernel 2
        float *ori_x, *ori_y, *ori_z;
        float *ori_outI, *ori_outR;
        float *host_ori_outI;		            /* Output signal (complex) */

        // // kernel 2
        // float *ptb_x, *ptb_y, *ptb_z;
        // float *ptb_outI, *ptb_outR;
        // float *host_ptb_outI;		            /* Output signal (complex) */

        // // gptb kernel
        // float *gptb_x, *gptb_y, *gptb_z;
        // float *gptb_outI, *gptb_outR;
        // float *host_gptb_outI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&ori_phiR, mrif_numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, mrif_numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, mrif_numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, mrif_base_phiR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, mrif_base_phiI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, mrif_base_dR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, mrif_base_dI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&ori_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ori_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, mrif_numX * sizeof(float)));

        // cudaErrCheck(cudaMalloc((void **)&ptb_x, mrif_numX * sizeof(float)));   
        // cudaErrCheck(cudaMalloc((void **)&ptb_y, mrif_numX * sizeof(float)));   
        // cudaErrCheck(cudaMalloc((void **)&ptb_z, mrif_numX * sizeof(float)));   
        // cudaErrCheck(cudaMemcpy(ptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(ptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(ptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMalloc((void **)&ptb_outR, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&ptb_outI, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMemset(ptb_outR, 0, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMemset(ptb_outI, 0, mrif_numX * sizeof(float)));

        // cudaErrCheck(cudaMalloc((void **)&gptb_x, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&gptb_y, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&gptb_z, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMemcpy(gptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(gptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(gptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMalloc((void **)&gptb_outR, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&gptb_outI, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMemset(gptb_outR, 0, mrif_numX * sizeof(float)));
        // cudaErrCheck(cudaMemset(gptb_outI, 0, mrif_numX * sizeof(float)));

        host_ori_outI = (float*) calloc (mrif_numX, sizeof (float));
        // host_ptb_outI = (float*) calloc (mrif_numX, sizeof (float));
        // host_gptb_outI = (float*) calloc (mrif_numX, sizeof (float));
    // ---------------------------------------------------------------------------------------

    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
        // computeRhoPhi_GPU(mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
        dim3 mrif_grid1;
        dim3 mrif_block1;
        mrif_grid1.x = mrif_numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_grid1.y = 1;
        mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_block1.y = 1;
        // printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \\n", 
        //             mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrif_grid1, mrif_block1 >>> (mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
        cudaErrCheck(cudaMemcpy(mrif_base_realRhoPhi, ori_realRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(mrif_base_imagRhoPhi, ori_imagRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < mrif_numK; k++) {
            mrif_kVals[k].Kx = mrif_base_kx[k];
            mrif_kVals[k].Ky = mrif_base_ky[k];
            mrif_kVals[k].Kz = mrif_base_kz[k];
            mrif_kVals[k].RhoPhiR = mrif_base_realRhoPhi[k];
            mrif_kVals[k].RhoPhiI = mrif_base_imagRhoPhi[k];
        }
    // ---------------------------------------------------------------------------------------

        dim3 mrif_grid2, ori_mrif_grid2;
        dim3 mrif_block2, ori_mrif_block2;
        mrif_grid2.x = mrif_numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrif_grid2.y = 1;
        mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrif_block2.y = 1;
        ori_mrif_grid2 = mrif_grid2;
        ori_mrif_block2 = mrif_block2;
        // printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \\n", 
        //             mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        mrif_kValues* mrif_kValsTile = mrif_kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, mrif_numK - FHGridBase);
        cudaMemcpyToSymbol(c, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);
        // cudaMemcpyToSymbol(c_gptb_mrif, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);
        // cudaMemcpyToSymbol(c_mix_mrif, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);

        this->launchGridDim = mrif_grid2;
        this->launchBlockDim = mrif_block2;

        this->MRIFKernelParams = new OriMRIFParamsStruct();
        this->MRIFKernelParams->numK = mrif_numK;
        this->MRIFKernelParams->kGlobalIndex = FHGridBase;
        this->MRIFKernelParams->x = ori_x;
        this->MRIFKernelParams->y = ori_y;
        this->MRIFKernelParams->z = ori_z;
        this->MRIFKernelParams->outR = ori_outR;
        this->MRIFKernelParams->outI = ori_outI;

        this->kernelParams.push_back(&(this->MRIFKernelParams->numK));
        this->kernelParams.push_back(&(this->MRIFKernelParams->kGlobalIndex));
        this->kernelParams.push_back(&(this->MRIFKernelParams->x));
        this->kernelParams.push_back(&(this->MRIFKernelParams->y));
        this->kernelParams.push_back(&(this->MRIFKernelParams->z));
        this->kernelParams.push_back(&(this->MRIFKernelParams->outR));
        this->kernelParams.push_back(&(this->MRIFKernelParams->outI));

        this->smem = 0;
        this->kernelFunc = (void*)ori_mrif;

        // dim3 mix_kernel_grid(SM_NUM*4, 1, 1);


        // g_general_ptb_mrif<<<mix_kernel_grid, 256>>>(mrif_numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, 
        //     ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
        //     0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mrif_grid2.x * ori_mrif_grid2.y * ori_mrif_grid2.z, 0);    

        // cudaDeviceSynchronize();
}


void OriMRIFKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, stream));
}