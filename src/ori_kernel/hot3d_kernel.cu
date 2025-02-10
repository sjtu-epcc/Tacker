#include "Logger.h"
#include "header/hot3d_header.h"
#include "hot3d_kernel.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_hot3d(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc) 
{
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;  
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}

OriHOT3DKernel::OriHOT3DKernel(int id) {
    this->Id = id;
    this->kernelName = "hot3d";
    initParams();
}

OriHOT3DKernel::~OriHOT3DKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->HOT3DKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriHOT3DKernel::initParams() {
    int numCols = 1024 * 8 * 2;
    int numRows = 1024 * 8;
    int layers = 8;

    /* calculating parameters*/
    float dx = chip_height/numRows;
    float dy = chip_width/numCols;
    float dz = t_chip/layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;

    float *powerIn, *tempOut, *tempIn, *tempCopy;
    int size = numCols * numRows * layers;

    powerIn = (float*)calloc(size, sizeof(float));
    tempCopy = (float*)malloc(size * sizeof(float));
    tempIn = (float*)calloc(size,sizeof(float));
    tempOut = (float*)calloc(size, sizeof(float));
    float* answer = (float*)calloc(size, sizeof(float));

    memcpy(tempCopy,tempIn, size * sizeof(float));

	float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * numCols * numRows * layers;  
    float  *tIn_d, *tOut_d, *p_d;
    cudaMalloc((void**)&p_d,s);
    cudaMalloc((void**)&tIn_d,s);
    cudaMalloc((void**)&tOut_d,s);
    cudaMemcpy(tIn_d, tempIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, powerIn, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(ori_hot3d, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(numCols / 64, numRows / 4, 1);

	this->launchGridDim = grid_dim;
	this->launchBlockDim = block_dim;

	this->HOT3DKernelParams = new OriHOT3DParamsStruct();
	this->HOT3DKernelParams->p = p_d;
	this->HOT3DKernelParams->tIn = tIn_d;
	this->HOT3DKernelParams->tOut = tOut_d;
	this->HOT3DKernelParams->sdc = stepDivCap;
	this->HOT3DKernelParams->nx = numCols;
	this->HOT3DKernelParams->ny = numRows;
	this->HOT3DKernelParams->nz = layers;
	this->HOT3DKernelParams->ce = ce;
	this->HOT3DKernelParams->cw = cw;
	this->HOT3DKernelParams->cn = cn;
	this->HOT3DKernelParams->cs = cs;
	this->HOT3DKernelParams->ct = ct;
	this->HOT3DKernelParams->cb = cb;
	this->HOT3DKernelParams->cc = cc;

	this->kernelParams.push_back(&(HOT3DKernelParams->p));
	this->kernelParams.push_back(&(HOT3DKernelParams->tIn));
	this->kernelParams.push_back(&(HOT3DKernelParams->tOut));
	this->kernelParams.push_back(&(HOT3DKernelParams->sdc));
	this->kernelParams.push_back(&(HOT3DKernelParams->nx));
	this->kernelParams.push_back(&(HOT3DKernelParams->ny));
	this->kernelParams.push_back(&(HOT3DKernelParams->nz));
	this->kernelParams.push_back(&(HOT3DKernelParams->ce));
	this->kernelParams.push_back(&(HOT3DKernelParams->cw));
	this->kernelParams.push_back(&(HOT3DKernelParams->cn));
	this->kernelParams.push_back(&(HOT3DKernelParams->cs));
	this->kernelParams.push_back(&(HOT3DKernelParams->ct));
	this->kernelParams.push_back(&(HOT3DKernelParams->cb));
	this->kernelParams.push_back(&(HOT3DKernelParams->cc));
	
    this->smem = 0;
    this->kernelFunc = (void*)ori_hot3d;

}

void OriHOT3DKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, 0));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}