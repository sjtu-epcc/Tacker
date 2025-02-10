#include "mriq_kernel.h"
#include "Logger.h"
#include "header/mriq_header.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

__constant__ __device__ mriq_kValues ck[KERNEL_Q_K_ELEMS_PER_GRID];
__constant__ __device__ mriq_kValues_int ck_int[KERNEL_Q_K_ELEMS_PER_GRID];

void inputData(int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI) {
    int numK, numX;
    FILE* fid = fopen("/home/jxdeng/workspace/tacker/0_mybench/file_t/mriq_input.bin", "r");
    if (fid == NULL) {
        fprintf(stderr, "Cannot open input file\n");
        exit(-1);
    }

    fread (&numK, sizeof (int), 1, fid);
    numK *= 10;
    *_numK = numK;
    fread (&numX, sizeof (int), 1, fid);
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
    fclose (fid); 
}

extern "C" __global__ void ori_ComputePhiMag(float* phiR, float* phiI, float* phiMag, int numK) {
    int indexK = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}

extern "C" __global__ void ori_mriq(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi) {
    for (int QGrid = 0; QGrid < 1; QGrid++) {
        kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

        float sX;
        float sY;
        float sZ;
        float sQr;
        float sQi;

        // Determine the element of the X arrays computed by this thread
        int xIndex = blockIdx.x * KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

        // Read block's X values from global mem to shared mem
        sX = x[xIndex];
        sY = y[xIndex];
        sZ = z[xIndex];
        sQr = Qr[xIndex];
        sQi = Qi[xIndex];

        // Loop over all elements of K in constant mem to compute a partial value
        // for X.
        int kIndex = 0;
        // if (numK % 2) {
        //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
        //     sQr += ck[0].PhiMag * cos(expArg);
        //     sQi += ck[0].PhiMag * sin(expArg);
        //     kIndex++;
        //     kGlobalIndex++;
        // }

        for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
            kIndex += 2, kGlobalIndex += 2) {
            float expArg = PIx2_MRIQ * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                        ck[kIndex].Kz * sZ);
            sQr += ck[kIndex].PhiMag * cos(expArg);
            sQi += ck[kIndex].PhiMag * sin(expArg);

            int kIndex1 = kIndex + 1;
            float expArg1 = PIx2_MRIQ * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                        ck[kIndex1].Kz * sZ);
            sQr += ck[kIndex1].PhiMag * cos(expArg1);
            sQi += ck[kIndex1].PhiMag * sin(expArg1);
        }

        Qr[xIndex] = sQr;
        Qi[xIndex] = sQi;
    }
}

OriMRIQKernel::OriMRIQKernel(int id){
    Id = id;
    this->kernelName = "mriq";
    initParams();
}

OriMRIQKernel::~OriMRIQKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->MRIQKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriMRIQKernel::initParams() {
    // mriq variables
    // ---------------------------------------------------------------------------------------
        int mriq_blks = 4;
        int mriq_iter = 1;
        int numK = 2097152;
        int numX = 2097152;
        float *base_kx, *base_ky, *base_kz;		/* K trajectory (3D vectors) */
        float *base_x, *base_y, *base_z;		/* X coordinates (3D vectors) */
        float *base_phiR, *base_phiI;		    /* Phi values (complex) */
        // float *base_phiMag;		                /* Magnitude of Phi */
        // float *base_Qr, *base_Qi;		        /* Q signal (complex) */
        struct mriq_kValues* mriq_kVals;

        // kernel 1
        float *mriq_ori_phiR, *mriq_ori_phiI;
        float *mriq_ori_phiMag, *host_mriq_ori_phiMag;
        // kernel 2
        float *mriq_ori_x, *mriq_ori_y, *mriq_ori_z;
        float *mriq_ori_Qr, *mriq_ori_Qi, *host_mriq_ori_Qi;

        // // kernel 1
        // float *ptb_phiR, *ptb_phiI;
        // float *ptb_phiMag, *host_ptb_phiMag;
        // kernel 2
        // float *mriq_ptb_x, *mriq_ptb_y, *mriq_ptb_z;
        // float *mriq_ptb_Qr, *mriq_ptb_Qi, *host_mriq_ptb_Qi;

        // gptb kernel 2
        // float *mriq_gptb_x, *mriq_gptb_y, *mriq_gptb_z;
        // float *mriq_gptb_Qr, *mriq_gptb_Qi, *host_mriq_gptb_Qi;

        inputData(&numK, &numX,
            &base_kx, &base_ky, &base_kz,
            &base_x, &base_y, &base_z,
            &base_phiR, &base_phiI);
        numK = 2097152;

        // Memory allocation
        // base_phiMag = (float* ) memalign(16, numK * sizeof(float));
        // base_Qr = (float*) memalign(16, numX * sizeof (float));
        // base_Qi = (float*) memalign(16, numX * sizeof (float));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiR, numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiMag, numK * sizeof(float)));
        host_mriq_ori_phiMag = (float* ) memalign(16, numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(mriq_ori_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&mriq_ori_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(mriq_ori_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qr, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qi, numX * sizeof(float)));
        cudaMemset((void *)mriq_ori_Qr, 0, numX * sizeof(float));
        cudaMemset((void *)mriq_ori_Qi, 0, numX * sizeof(float));
        host_mriq_ori_Qi = (float*) memalign(16, numX * sizeof (float));

        // cudaErrCheck(cudaMalloc((void **)&ptb_phiR, numK * sizeof(float)));   
        // cudaErrCheck(cudaMalloc((void **)&ptb_phiI, numK * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&ptb_phiMag, numK * sizeof(float)));
        // host_ptb_phiMag = (float* ) memalign(16, numK * sizeof(float));
        // cudaErrCheck(cudaMemcpy(ptb_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(ptb_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

        // cudaErrCheck(cudaMalloc((void **)&mriq_ptb_x, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_ptb_y, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_ptb_z, numX * sizeof(float)));
        // cudaErrCheck(cudaMemcpy(mriq_ptb_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(mriq_ptb_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(mriq_ptb_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMalloc((void **)&mriq_ptb_Qr, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_ptb_Qi, numX * sizeof(float)));
        // cudaMemset((void *)mriq_ptb_Qr, 0, numX * sizeof(float));
        // cudaMemset((void *)mriq_ptb_Qi, 0, numX * sizeof(float));
        // host_mriq_ptb_Qi = (float*) memalign(16, numX * sizeof (float));

        // // gptb
        // cudaErrCheck(cudaMalloc((void **)&mriq_gptb_x, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_gptb_y, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_gptb_z, numX * sizeof(float)));
        // cudaErrCheck(cudaMemcpy(mriq_gptb_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(mriq_gptb_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(mriq_gptb_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMalloc((void **)&mriq_gptb_Qr, numX * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&mriq_gptb_Qi, numX * sizeof(float)));
        // cudaMemset((void *)mriq_gptb_Qr, 0, numX * sizeof(float));
        // cudaMemset((void *)mriq_gptb_Qi, 0, numX * sizeof(float));
        // host_mriq_gptb_Qi = (float*) memalign(16, numX * sizeof (float));
    // ---------------------------------------------------------------------------------------

        // PRE running
    // ---------------------------------------------------------------------------------------
        dim3 mriq_grid1;
        dim3 mriq_block1;
        mriq_grid1.x = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
        mriq_grid1.y = 1;
        mriq_block1.x = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
        mriq_block1.y = 1;
        // printf("[ORI] Running with mriq...\\n");
        // printf("[ORI] mriq_grid1 -- %d * %d * %d mriq_block1 -- %d * %d * %d \\n", 
            // mriq_grid1.x, mriq_grid1.y, mriq_grid1.z, mriq_block1.x, mriq_block1.y, mriq_block1.z);

        checkKernelErrors((ori_ComputePhiMag <<< mriq_grid1, mriq_block1 >>> (mriq_ori_phiR, mriq_ori_phiI, mriq_ori_phiMag, numK)));
        cudaMemcpy(host_mriq_ori_phiMag, mriq_ori_phiMag, numK * sizeof(float), cudaMemcpyDeviceToHost);

        mriq_kVals = (struct mriq_kValues*)calloc(numK, sizeof (struct mriq_kValues));
        for (int k = 0; k < numK; k++) {
            mriq_kVals[k].Kx = base_kx[k];
            mriq_kVals[k].Ky = base_ky[k];
            mriq_kVals[k].Kz = base_kz[k];
            mriq_kVals[k].PhiMag = host_mriq_ori_phiMag[k];
        }
    // ---------------------------------------------------------------------------------------
        numX = (numX / 10) * mriq_iter;

        dim3 mriq_grid2, ori_mriq_grid2;
        dim3 mriq_block2, ori_mriq_block2;
        mriq_grid2.x = numX / KERNEL_Q_THREADS_PER_BLOCK;
        mriq_grid2.y = 1;
        mriq_block2.x = KERNEL_Q_THREADS_PER_BLOCK;
        mriq_block2.y = 1;
        ori_mriq_grid2 = mriq_grid2;
        ori_mriq_block2 = mriq_block2;
        // printf("[ORI] mriq_grid2 -- %d * %d * %d mriq_block2 -- %d * %d * %d \\n", 
        //     mriq_grid2.x, mriq_grid2.y, mriq_grid2.z, mriq_block2.x, mriq_block2.y, mriq_block2.z);

        int QGridBase = 0 * KERNEL_Q_K_ELEMS_PER_GRID;
        mriq_kValues* kValsTile = mriq_kVals + QGridBase;
        cudaMemcpyToSymbol(ck, kValsTile, KERNEL_Q_K_ELEMS_PER_GRID * sizeof(mriq_kValues), 0);

        this->launchGridDim = mriq_grid2;
        this->launchBlockDim = mriq_block2;

        this->MRIQKernelParams = new OriMRIQParamsStruct();
        this->MRIQKernelParams->numK = numK;
        this->MRIQKernelParams->kGlobalIndex = QGridBase;
        this->MRIQKernelParams->x = mriq_ori_x;
        this->MRIQKernelParams->y = mriq_ori_y;
        this->MRIQKernelParams->z = mriq_ori_z;
        this->MRIQKernelParams->Qr = mriq_ori_Qr;
        this->MRIQKernelParams->Qi = mriq_ori_Qi;

        this->kernelParams.push_back(&(this->MRIQKernelParams->numK));
        this->kernelParams.push_back(&(this->MRIQKernelParams->kGlobalIndex));
        this->kernelParams.push_back(&(this->MRIQKernelParams->x));
        this->kernelParams.push_back(&(this->MRIQKernelParams->y));
        this->kernelParams.push_back(&(this->MRIQKernelParams->z));
        this->kernelParams.push_back(&(this->MRIQKernelParams->Qr));
        this->kernelParams.push_back(&(this->MRIQKernelParams->Qi));
        

        this->smem = 0;
        this->kernelFunc = (void*)ori_mriq;

}

void OriMRIQKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, stream));
}