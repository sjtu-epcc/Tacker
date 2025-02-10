#include "lbm_kernel.h"
#include "Logger.h"
#include "header/lbm_header.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_lbm( float* srcGrid, float* dstGrid)  {
    //Using some predefined macros here.  Consider this the declaration 
    //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

    SWEEP_VAR
    SWEEP_X = threadIdx.x;
    SWEEP_Y = blockIdx.x;
    SWEEP_Z = blockIdx.y;

    float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
    float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
    float tempSB, tempET, tempEB, tempWT, tempWB ;

    //Load all of the input fields
    //This is a gather operation of the SCATTER preprocessor variable
        // is undefined in layout_config.h, or a "local" read otherwise
    tempC = SRC_C(srcGrid);
    tempN = SRC_N(srcGrid);
    tempS = SRC_S(srcGrid);
    tempE = SRC_E(srcGrid);
    tempW = SRC_W(srcGrid);
    tempT = SRC_T(srcGrid);
    tempB = SRC_B(srcGrid);
    tempNE= SRC_NE(srcGrid);
    tempNW= SRC_NW(srcGrid);
    tempSE = SRC_SE(srcGrid);
    tempSW = SRC_SW(srcGrid);
    tempNT = SRC_NT(srcGrid);
    tempNB = SRC_NB(srcGrid);
    tempST = SRC_ST(srcGrid);
    tempSB = SRC_SB(srcGrid);
    tempET = SRC_ET(srcGrid);
    tempEB = SRC_EB(srcGrid);
    tempWT = SRC_WT(srcGrid);
    tempWB = SRC_WB(srcGrid);

    //Test whether the cell is fluid or obstacle
    if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
        //Swizzle the inputs: reflect any fluid coming into this cell 
        // back to where it came from
        temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
        temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
        temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
        temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
        temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
        temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
        temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
        temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
        temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
    }
    else {
        //The math meat of LBM: ignore for optimization
        float ux, uy, uz, rho, u2;
        float temp1, temp2, temp_base;
        rho = tempC + tempN
            + tempS + tempE
            + tempW + tempT
            + tempB + tempNE
            + tempNW + tempSE
            + tempSW + tempNT
            + tempNB + tempST
            + tempSB + tempET
            + tempEB + tempWT
            + tempWB;

        ux = + tempE - tempW
            + tempNE - tempNW
            + tempSE - tempSW
            + tempET + tempEB
            - tempWT - tempWB;
        uy = + tempN - tempS
            + tempNE + tempNW
            - tempSE - tempSW
            + tempNT + tempNB
            - tempST - tempSB;
        uz = + tempT - tempB
            + tempNT - tempNB
            + tempST - tempSB
            + tempET - tempEB
            + tempWT - tempWB;

        ux /= rho;
        uy /= rho;
        uz /= rho;
        if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
            ux = 0.005f;
            uy = 0.002f;
            uz = 0.000f;
        }
        u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
        temp_base = OMEGA*rho;
        temp1 = DFL1*temp_base;


        //Put the output values for this cell in the shared memory
        temp_base = OMEGA*rho;
        temp1 = DFL1*temp_base;
        temp2 = 1.0f-OMEGA;
        tempC = temp2*tempC + temp1*(                                 - u2);
            temp1 = DFL2*temp_base;	
        tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
        tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
        tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
        tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
        tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
        tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
        temp1 = DFL3*temp_base;
        tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
        tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
        tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
        tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
        tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
        tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
        tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
        tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
        tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
        tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
        tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
        tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
    }

    //Write the results computed above
    //This is a scatter operation of the SCATTER preprocessor variable
        // is defined in layout_config.h, or a "local" write otherwise
    DST_C ( dstGrid ) = tempC;

    DST_N ( dstGrid ) = tempN; 
    DST_S ( dstGrid ) = tempS;
    DST_E ( dstGrid ) = tempE;
    DST_W ( dstGrid ) = tempW;
    DST_T ( dstGrid ) = tempT;
    DST_B ( dstGrid ) = tempB;

    DST_NE( dstGrid ) = tempNE;
    DST_NW( dstGrid ) = tempNW;
    DST_SE( dstGrid ) = tempSE;
    DST_SW( dstGrid ) = tempSW;
    DST_NT( dstGrid ) = tempNT;
    DST_NB( dstGrid ) = tempNB;
    DST_ST( dstGrid ) = tempST;
    DST_SB( dstGrid ) = tempSB;
    DST_ET( dstGrid ) = tempET;
    DST_EB( dstGrid ) = tempEB;
    DST_WT( dstGrid ) = tempWT;
    DST_WB( dstGrid ) = tempWB;
}

OriLBMKernel::OriLBMKernel(int id){
    Id = id;
    this->kernelName = "lbm";
    // loadKernel();
    initParams();
}

OriLBMKernel::~OriLBMKernel(){
    // free gpu memory
    // for (auto &ptr : cudaFreeList) {
    //     CUDA_SAFE_CALL(cudaFree(ptr));
    // }

    // // free cpu heap memory
    // free(this->LBMKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriLBMKernel::initParams(){
    // lbm variables
    // ---------------------------------------------------------------------------------------
    int lbm_blks = 1;
    int lbm_iter = 1;
    float *lbm_ori_src;
    float *lbm_ori_dst;
    // float *lbm_ptb_src;
    // float *lbm_ptb_dst;
    // float *lbm_gptb_src;
    // float *lbm_gptb_dst;
    float *host_lbm_ori_dst;
    // float *host_lbm_ptb_dst;
    // float *host_lbm_gptb_dst;

    size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

    size *= 2;

    host_lbm_ori_dst = (float *)malloc(size);
    // host_lbm_ptb_dst = (float *)malloc(size);
    // host_lbm_gptb_dst = (float *)malloc(size);
    CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_ori_src, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_ori_dst, size));
    cudaFreeList.push_back(lbm_ori_src);
    cudaFreeList.push_back(lbm_ori_dst);
    // CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_ptb_src, size));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_ptb_dst, size));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_gptb_src, size));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&lbm_gptb_dst, size));

    curandGenerator_t lbm_gen;
    curandErrCheck(curandCreateGenerator(&lbm_gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(lbm_gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
    curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
    // CUDA_SAFE_CALL(cudaMemcpy(lbm_ptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(lbm_ptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(lbm_gptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(lbm_gptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
    lbm_ori_src += REAL_MARGIN;
    lbm_ori_dst += REAL_MARGIN;
    // lbm_ptb_src += REAL_MARGIN;
    // lbm_ptb_dst += REAL_MARGIN;
    // lbm_gptb_src += REAL_MARGIN;
    // lbm_gptb_dst += REAL_MARGIN;

    dim3 lbm_block, lbm_grid;
    lbm_block.x = SIZE_X;
    lbm_grid.x = SIZE_Y;
    lbm_grid.y = SIZE_Z;
    lbm_block.y = lbm_block.z = lbm_grid.z = 1;

    this->launchGridDim = lbm_grid;
    this->launchBlockDim = lbm_block;

    this->LBMKernelParams = new OriLBMParamsStruct();
    this->LBMKernelParams->src = lbm_ori_src;
    this->LBMKernelParams->dst = lbm_ori_dst;

    this->kernelParams.push_back(&(this->LBMKernelParams->src));
    this->kernelParams.push_back(&(this->LBMKernelParams->dst));

    this->smem = 0;
    this->kernelFunc = (void*)ori_lbm;

}

void OriLBMKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, stream));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}