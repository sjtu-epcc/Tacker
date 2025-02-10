#include "Logger.h"
#include "header/nn_header.h"
#include "nn_kernel.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_nn(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng) 
{
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}

OriNNKernel::OriNNKernel(int id) {
    this->Id = id;
    this->kernelName = "nn";
    initParams();
}

OriNNKernel::~OriNNKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->NNKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriNNKernel::initParams() {
 int    i=0;
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;

    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;

    int numRecords = 256 * 2560000;
    if (resultsCount > numRecords) resultsCount = numRecords;

    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;

	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, 0 );
	cudaDeviceSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
	cudaDeviceSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel
    */

	printf("nn kernel: gridDim.x %d , gridDim.y %d, threadsPerBlock %d \n", 
				gridDim.x, gridDim.y, threadsPerBlock);

	this->launchGridDim = gridDim;
	this->launchBlockDim = dim3(threadsPerBlock, 1, 1);
    NNKernelParams = new OriNNParamsStruct();
	this->NNKernelParams->d_locations = d_locations;
	this->NNKernelParams->d_distances = d_distances;
	this->NNKernelParams->numRecords = numRecords;
	this->NNKernelParams->lat = lat;
	this->NNKernelParams->lng = lng;

	this->kernelParams.push_back(&(NNKernelParams->d_locations));
	this->kernelParams.push_back(&(NNKernelParams->d_distances));
	this->kernelParams.push_back(&(NNKernelParams->numRecords));
	this->kernelParams.push_back(&(NNKernelParams->lat));
	this->kernelParams.push_back(&(NNKernelParams->lng));
    
    this->smem = 0;
    this->kernelFunc = (void*)ori_nn;

}

void OriNNKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, 0));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}