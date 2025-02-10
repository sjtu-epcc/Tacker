/*** 
 * @Author: diagonal
 * @Date: 2023-12-08 21:52:35
 * @LastEditors: diagonal
 * @LastEditTime: 2023-12-09 12:40:59
 * @FilePath: /tacker/runtime/cp_kernel.cu
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */
// cp_kernel.cu
#pragma once
#include "cp_kernel.h"
#include "Logger.h"
#include "header/cp_header.h"
#include "util.h"
#include "TackerConfig.h"
#include "ModuleCenter.h"

extern Logger logger;
extern ModuleCenter moduleCenter;

__constant__ float4 atominfo[MAXATOMS];
__constant__ int4 atominfo_int[MAXATOMS];

extern "C" __global__ void ori_cp(int numatoms, float gridspacing, float * energygrid) {
		unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
								+ threadIdx.x;
		unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
		unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
								+ xindex;

		float coory = gridspacing * yindex;
		float coorx = gridspacing * xindex;

		float energyvalx1=0.0f;
		float energyvalx2=0.0f;
		float energyvalx3=0.0f;
		float energyvalx4=0.0f;
		float energyvalx5=0.0f;
		float energyvalx6=0.0f;
		float energyvalx7=0.0f;
		float energyvalx8=0.0f;

		float gridspacing_u = gridspacing * BLOCKSIZEX;

		int atomid;
		for (atomid=0; atomid<numatoms; atomid++) {
			float dy = coory - atominfo[atomid].y;
			float dyz2 = (dy * dy) + atominfo[atomid].z;

			float dx1 = coorx - atominfo[atomid].x;
			float dx2 = dx1 + gridspacing_u;
			float dx3 = dx2 + gridspacing_u;
			float dx4 = dx3 + gridspacing_u;
			float dx5 = dx4 + gridspacing_u;
			float dx6 = dx5 + gridspacing_u;
			float dx7 = dx6 + gridspacing_u;
			float dx8 = dx7 + gridspacing_u;

			energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dyz2));
			energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dyz2));
			energyvalx3 += atominfo[atomid].w * (1.0f / sqrtf(dx3*dx3 + dyz2));
			energyvalx4 += atominfo[atomid].w * (1.0f / sqrtf(dx4*dx4 + dyz2));
			energyvalx5 += atominfo[atomid].w * (1.0f / sqrtf(dx5*dx5 + dyz2));
			energyvalx6 += atominfo[atomid].w * (1.0f / sqrtf(dx6*dx6 + dyz2));
			energyvalx7 += atominfo[atomid].w * (1.0f / sqrtf(dx7*dx7 + dyz2));
			energyvalx8 += atominfo[atomid].w * (1.0f / sqrtf(dx8*dx8 + dyz2));
		}

		energygrid[outaddr]   += energyvalx1;
		energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
		energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
		energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
		energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
		energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
		energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
		energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
}

extern "C" __global__ void ori_cp_d(int numatoms, float gridspacing, int * energygrid) {
		unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
								+ threadIdx.x;
		unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
		unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
								+ xindex;

		int coory = gridspacing * yindex;
		int coorx = gridspacing * xindex;

		int energyvalx1=0.0f;
		int energyvalx2=0.0f;
		int energyvalx3=0.0f;
		int energyvalx4=0.0f;
		int energyvalx5=0.0f;
		int energyvalx6=0.0f;
		int energyvalx7=0.0f;
		int energyvalx8=0.0f;

		int gridspacing_u = gridspacing * BLOCKSIZEX;

		int atomid;
		for (atomid=0; atomid<numatoms; atomid++) {
			int dy = coory - atominfo_int[atomid].y;
			int dyz2 = (dy * dy) + atominfo_int[atomid].z;

			int dx1 = coorx - atominfo_int[atomid].x;
			int dx2 = dx1 + gridspacing_u;
			int dx3 = dx2 + gridspacing_u;
			int dx4 = dx3 + gridspacing_u;
			int dx5 = dx4 + gridspacing_u;
			int dx6 = dx5 + gridspacing_u;
			int dx7 = dx6 + gridspacing_u;
			int dx8 = dx7 + gridspacing_u;

			energyvalx1 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx1*dx1 + dyz2));
			energyvalx2 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx2*dx2 + dyz2));
			energyvalx3 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx3*dx3 + dyz2));
			energyvalx4 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx4*dx4 + dyz2));
			energyvalx5 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx5*dx5 + dyz2));
			energyvalx6 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx6*dx6 + dyz2));
			energyvalx7 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx7*dx7 + dyz2));
			energyvalx8 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx8*dx8 + dyz2));
		}

		energygrid[outaddr]   += energyvalx1;
		energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
		energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
		energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
		energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
		energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
		energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
		energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
}

// 构造函数
OriCPKernel::OriCPKernel(int id, const std::string& moduleName, const std::string& kernelName) {
    Id = id;
    this->kernelName = kernelName;
    this->moduleName = moduleName;
    initParams();
    // loadKernel();
}

OriCPKernel::OriCPKernel(int id){
    Id = id;
    kernelName = "cp";
    moduleName = "ori_cp";
	if (id < 0) {
		initParams_int();
	} else {
		initParams();
	}
    // loadKernel();
}

// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.
int copyatomstoconstbuf(float *atoms, int count, float zplane) {
	if (count > MAXATOMS) {
		printf("Atom count exceeds constant buffer storage capacity\n");
		return -1;
	}

	float atompre[4*MAXATOMS];
	int i;
	for (i=0; i<count*4; i+=4) {
		atompre[i    ] = atoms[i    ];
		atompre[i + 1] = atoms[i + 1];
		float dz = zplane - atoms[i + 2];
		atompre[i + 2]  = dz*dz;
		atompre[i + 3] = atoms[i + 3];
	}

	cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
	CUERR // check and clear any existing errors

	return 0;
}


/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
	dim3 size;
	int i;
	float *atoms;

	srand(54321);			// Ensure that atom placement is repeatable

	atoms = (float *) malloc(count * 4 * sizeof(float));
	*atombuf = atoms;

	// compute grid dimensions in angstroms
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;

	for (i=0; i<count; i++) {
		int addr = i * 4;
		atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
		atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
		atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
		atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
	}  

	return 0;
}

template <typename T>
static int initatoms(T **atombuf, int count, dim3 volsize, float gridspacing) {
	dim3 size;
	int i;
	T *atoms;

	srand(54321);			// Ensure that atom placement is repeatable

	atoms = (T *) malloc(count * 4 * sizeof(T));
	*atombuf = atoms;

	// compute grid dimensions in angstroms
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;

	for (i=0; i<count; i++) {
		int addr = i * 4;
		atoms[addr    ] = (rand() / (T) RAND_MAX) * size.x; 
		atoms[addr + 1] = (rand() / (T) RAND_MAX) * size.y; 
		atoms[addr + 2] = (rand() / (T) RAND_MAX) * size.z; 
		atoms[addr + 3] = ((rand() / (T) RAND_MAX) * 2.0) - 1.0;  // charge
	}  

	return 0;
}

template <typename T>
int copyatomstoconstbuf(T *atoms, int count, float zplane) {
	if (count > MAXATOMS) {
		printf("Atom count exceeds constant buffer storage capacity\n");
		return -1;
	}

	T atompre[4*MAXATOMS];
	int i;
	for (i=0; i<count*4; i+=4) {
		atompre[i    ] = atoms[i    ];
		atompre[i + 1] = atoms[i + 1];
		float dz = zplane - atoms[i + 2];
		atompre[i + 2]  = dz*dz;
		atompre[i + 3] = atoms[i + 3];
	}

	// CUDA_SAFE_CALL(cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0));
	// printf("memcpy size: %d\n", count * 4 * sizeof(float));
	if constexpr (std::is_same<T, int>::value) {
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(atominfo_int, atompre, count * 4 * sizeof(T), 0));
	} else if constexpr (std::is_same<T, float>::value) {
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(T), 0));
	} else {
		printf("error type\n");
		exit(EXIT_FAILURE);
	}
	CUERR // check and clear any existing errors

	return 0;
}

// 初始化参数default
void OriCPKernel::initParams() {
    int cp_blks = 6;
    int cp_iter = 1;
    float *atoms = NULL;
    int atomcount = ATOMCOUNT;
    const float gridspacing = 0.1;					// number of atoms to simulate
    dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
    initatoms(&atoms, atomcount, volsize, gridspacing);

    // allocate and initialize the GPU output array
    int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

    float *ori_output;	
    // float *ptb_output;
    // float *gptb_output;
	if (!this->initialized) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&ori_output, volmemsz));
		CUDA_SAFE_CALL(cudaMemset(ori_output, 0, volmemsz));
	} else {
		CUDA_SAFE_CALL(cudaMemset(this->CPKernelParams->energygrid, 0, volmemsz));
	}
    // cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
    // cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
    // float *host_ori_energy = (float *) malloc(volmemsz);
    // float *host_ptb_energy = (float *) malloc(volmemsz);
    // float *host_gptb_energy = (float *) malloc(volmemsz);

    dim3 cp_grid, cp_block;
    int atomstart = 1;
    int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
    cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
    cp_block.y = BLOCKSIZEY;
    cp_block.z = 1;
    cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
    cp_grid.y = volsize.y / cp_block.y; 
    cp_grid.z = volsize.z / cp_block.z; 

    copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);
	
	if (!this->initialized) {
		this->CPKernelParams = new OriCPParamsStruct<float>();
		this->CPKernelParams->numatoms = runatoms;
		this->CPKernelParams->gridspacing = 0.1;
		this->CPKernelParams->energygrid = ori_output;
		this->launchGridDim = cp_grid;
		this->launchBlockDim = cp_block;

		this->kernelParams.push_back(&this->CPKernelParams->numatoms);
		this->kernelParams.push_back(&this->CPKernelParams->gridspacing);
		this->kernelParams.push_back(&this->CPKernelParams->energygrid);

		this->kernelFunc = (void*)ori_cp;
		this->smem = 0;

		this->initialized = true;
	}
}

void OriCPKernel::initParams_int() {
	printf("OriCPKernel:initParams_int!\n");
    int cp_blks = 6;
    int cp_iter = 1;
    int *atoms = NULL;
    int atomcount = ATOMCOUNT;
    const float gridspacing = 0.1;					// number of atoms to simulate
    dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
    initatoms<int>(&atoms, atomcount, volsize, gridspacing);

    // allocate and initialize the GPU output array
    int volmemsz = sizeof(int) * volsize.x * volsize.y * volsize.z;

    int *ori_output;	
    // float *ptb_output;
    // float *gptb_output;
	if (!this->initialized) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&ori_output, volmemsz));
		CUDA_SAFE_CALL(cudaMemset(ori_output, 0, volmemsz));
	} else {
		CUDA_SAFE_CALL(cudaMemset(this->CPKernelParams->energygrid, 0, volmemsz));
	}
    // cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
    // cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
    // float *host_ori_energy = (float *) malloc(volmemsz);
    // float *host_ptb_energy = (float *) malloc(volmemsz);
    // float *host_gptb_energy = (float *) malloc(volmemsz);

    dim3 cp_grid, cp_block;
    int atomstart = 1;
    int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
    cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
    cp_block.y = BLOCKSIZEY;
    cp_block.z = 1;
    cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
    cp_grid.y = volsize.y / cp_block.y; 
    cp_grid.z = volsize.z / cp_block.z; 

    copyatomstoconstbuf<int>(atoms + 4 * atomstart, runatoms, 0*gridspacing);
	
	if (!this->initialized) {
		this->CPKernelParams_int = new OriCPParamsStruct<int>();
		this->CPKernelParams_int->numatoms = runatoms;
		this->CPKernelParams_int->gridspacing = 0.1;
		this->CPKernelParams_int->energygrid = ori_output;
		this->launchGridDim = cp_grid;
		this->launchBlockDim = cp_block;

		this->kernelParams.push_back(&this->CPKernelParams_int->numatoms);
		this->kernelParams.push_back(&this->CPKernelParams_int->gridspacing);
		this->kernelParams.push_back(&this->CPKernelParams_int->energygrid);

		this->kernelFunc = (void*)ori_cp_d;
		this->smem = 0;

		this->initialized = true;
	}
}

// 虚析构函数实现
OriCPKernel::~OriCPKernel() {
    // free gpu memory
    CUDA_SAFE_CALL(cudaFree(this->CPKernelParams->energygrid));

	// free heap host memory
	free(this->CPKernelParams);

    logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriCPKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // // print dim
    // logger.INFO("[Ori] cp -- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("[Ori] cp -- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
    launchGridDim, launchBlockDim,
    (void**)this->kernelParams.data(), 0, stream));
    
}

void OriCPKernel::loadKernel() {
    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is loading ...");

    this->function = moduleCenter.getFunction(moduleName, kernelName);

    if (this->function == nullptr) {
        logger.ERROR("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " load failed!");
        exit(EXIT_FAILURE);
    }

    return ;
}
