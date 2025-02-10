#pragma once
#include "fft_kernel.h"
#include "Logger.h"
#include "header/fft_header.h"
#include "util.h"
#include "TackerConfig.h"
#include "ModuleCenter.h"

extern Logger logger;
extern ModuleCenter moduleCenter;     

__device__ void GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	__syncthreads(); 
	// asm volatile("bar.sync %0, %1;" : : "r"(sync_id), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	__syncthreads(); 
	// asm volatile("bar.sync %0, %1;" : : "r"(sync_id), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}      

__device__ void GPU_DoFft(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		GPU_exchange( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

extern "C" __global__ void ori_fft(float2* data) {
	float2 *ori_data = data + blockIdx.x*FFT_N;
		float2 v[FFT_R];
		data = ori_data;

		int idxG = threadIdx.x; 
		for (int r=0; r<FFT_R; r++) {  
			v[r] = data[idxG + r*FFT_T];
		} 
		GPU_DoFft( v, threadIdx.x );  
		for (int r=0; r<FFT_R; r++) {
			data[idxG + r*FFT_T] = v[r];
		} 
}

// 构造函数
OriFFTKernel::OriFFTKernel(int id, const std::string& moduleName, const std::string& kernelName) {
    Id = id;
    this->kernelName = kernelName;
    this->moduleName = moduleName;
    // loadKernel();
    initParams();
}

OriFFTKernel::OriFFTKernel(int id){
    Id = id;
    this->kernelName = "fft";
    this->moduleName = "ori_fft"; 
    // loadKernel();
    if (id < 0) {
        initParams_int();
    } else {
        initParams();
    }
}

void OriFFTKernel::initParams() {
    //8*1024*1024;
    int fft_blks = 3;
    int fft_iter = 1;
    int n_bytes = FFT_N * FFT_B * sizeof(float2) * 10; // up scale
    int nthreads = FFT_T;
    srand(54321);

    float *host_shared_source =(float *)malloc(n_bytes);  
    float2 *source    = (float2 *)malloc( n_bytes );
    float2 *host_fft_ori_result    = (float2 *)malloc( n_bytes );

    for(int b=0; b<FFT_B;b++) {	
        for( int i = 0; i < FFT_N; i++ ) {
            source[b*FFT_N+i].x = (rand()/(float)RAND_MAX)*2-1;
            source[b*FFT_N+i].y = (rand()/(float)RAND_MAX)*2-1;
        }
    }

    // allocate device memory
    float2 *fft_ori_source;
    // float *fft_ori_shared_source;
    // cudaMalloc((void**) &fft_ori_shared_source, n_bytes);
    // copy host memory to device
    // cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
    if (!this->initialized) cudaMalloc((void**) &fft_ori_source, n_bytes);
    else fft_ori_source = (float2*)this->FFTKernelParams->data;

    // copy host memory to device
    cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

    dim3 fft_grid;
    dim3 fft_block;
    fft_grid.x = FFT_B;
    fft_block.x = nthreads;

    if (!this->initialized) {
        this->FFTKernelParams = new OriFFTParamsStruct<float2>();
        this->FFTKernelParams->data = fft_ori_source;

        this->kernelParams.push_back(&(this->FFTKernelParams->data));

        this->launchGridDim = fft_grid;
        this->launchBlockDim = fft_block;

        this->smem = 2048;

        this->kernelFunc = (void*)ori_fft;
        this->initialized = true;
    }
}

void OriFFTKernel::initParams_int() {
    //8*1024*1024;
    printf("OriFFTKernel:initParams_int\n");
    int fft_blks = 3;
    int fft_iter = 1;
    int n_bytes = FFT_N * FFT_B * sizeof(int2) * 10; // up scale
    int nthreads = FFT_T;
    srand(54321);

    int *host_shared_source = (int *)malloc(n_bytes);  
    int2 *source    = (int2 *)malloc( n_bytes );
    int2 *host_fft_ori_result    = (int2 *)malloc( n_bytes );

    for(int b=0; b<FFT_B;b++) {	
        for( int i = 0; i < FFT_N; i++ ) {
            source[b*FFT_N+i].x = (rand()/(int)RAND_MAX)*2-1;
            source[b*FFT_N+i].y = (rand()/(int)RAND_MAX)*2-1;
        }
    }

    // allocate device memory
    int2 *fft_ori_source;
    // float *fft_ori_shared_source;
    // cudaMalloc((void**) &fft_ori_shared_source, n_bytes);
    // copy host memory to device
    // cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
    if (!this->initialized) cudaMalloc((void**) &fft_ori_source, n_bytes);
    else fft_ori_source = (int2*)this->FFTKernelParams->data;

    // copy host memory to device
    cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

    dim3 fft_grid;
    dim3 fft_block;
    fft_grid.x = FFT_B;
    fft_block.x = nthreads;

    if (!this->initialized) {
        this->FFTKernelParams_int = new OriFFTParamsStruct<int2>();
        this->FFTKernelParams_int->data = fft_ori_source;

        this->kernelParams.push_back(&(this->FFTKernelParams_int->data));

        this->launchGridDim = fft_grid;
        this->launchBlockDim = fft_block;

        this->smem = 0;

        this->kernelFunc = (void*)ori_fft;
        this->initialized = true;
    }
}

OriFFTKernel::~OriFFTKernel() {
    // free gpu memory
    // for (auto &ptr : cudaFreeList) {
    //     CUDA_SAFE_CALL(cudaFree(ptr));
    // }

    // // free cpu heap memory
    // free(this->FFTKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriFFTKernel::loadKernel() {
    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is loading ...");

    this->function = moduleCenter.getFunction(moduleName, kernelName);

    if (this->function == nullptr) {
        logger.ERROR("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " load failed!");
        exit(EXIT_FAILURE);
    }
    return ;
}

void OriFFTKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, stream));
}