#include "Logger.h"
#include "header/path_header.h"
#include "path_kernel.h"
#include "util.h"
#include "TackerConfig.h"

extern Logger logger;

extern "C" __global__ void ori_path(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{
    __shared__ int prev[PATH_BLOCK_SIZE];
    __shared__ int result[PATH_BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
	int small_block_cols = PATH_BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+PATH_BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
    
    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols-1) ? PATH_BLOCK_SIZE-1-(blkXmax-cols+1) : PATH_BLOCK_SIZE-1;

    int W = tx-1;
    int E = tx+1;
    
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i=0; i<iteration ; i++){ 
        computed = false;
        if( IN_RANGE(tx, i+1, PATH_BLOCK_SIZE-i-2) &&  \
                isValid){
                computed = true;
                int left = prev[W];
                int up = prev[tx];
                int right = prev[E];
                int shortest = MIN(left, up);
                shortest = MIN(shortest, right);
                int index = cols*(startStep+i)+xidx;
                result[tx] = shortest + gpuWall[index];

        }
        __syncthreads();
        if(i==iteration-1)
            break;
        if(computed)	 //Assign the computation range
            prev[tx]= result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        gpuResults[xidx]=result[tx];		
    }
}

OriPATHKernel::OriPATHKernel(int id) {
    this->Id = id;
    this->kernelName = "path";
    initParams();
}

OriPATHKernel::~OriPATHKernel(){
    // free gpu memory
    for (auto &ptr : cudaFreeList) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }

    // free cpu heap memory
    free(this->PATHKernelParams);
    
    // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
}

void OriPATHKernel::initParams() {
    /* --------------- pyramid parameters --------------- */
    int rows = 1000000, cols = 1200, pyramid_height = 125;
    int* data;
    int** wall;
    int* result;
    data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = PATH_BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    // printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	// pyramid_height, cols, borderCols, PATH_BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

    int src = 1, dst = 0;

    dim3 dimBlock(PATH_BLOCK_SIZE);
    dim3 dimGrid(blockCols);  
    printf("ori path kernel dimGrid: %d, dimBlock: %d\n", dimGrid.x, dimBlock.x);

    this->launchGridDim = dimBlock;
    this->launchBlockDim = dimGrid;

    PATHKernelParams = new OriPATHParamsStruct();

    this->PATHKernelParams->iteration = pyramid_height;
    this->PATHKernelParams->gpuWall = gpuWall;
    this->PATHKernelParams->gpuSrc = gpuResult[src];
    this->PATHKernelParams->gpuResults = gpuResult[dst];
    this->PATHKernelParams->cols = cols;
    this->PATHKernelParams->rows = rows;
    this->PATHKernelParams->startStep = 0;
    this->PATHKernelParams->border = borderCols;

    this->kernelParams.push_back(&(PATHKernelParams->iteration));
    this->kernelParams.push_back(&(PATHKernelParams->gpuWall));
    this->kernelParams.push_back(&(PATHKernelParams->gpuSrc));
    this->kernelParams.push_back(&(PATHKernelParams->gpuResults));
    this->kernelParams.push_back(&(PATHKernelParams->cols));
    this->kernelParams.push_back(&(PATHKernelParams->rows));
    this->kernelParams.push_back(&(PATHKernelParams->startStep));
    this->kernelParams.push_back(&(PATHKernelParams->border));


    this->smem = 0;
    this->kernelFunc = (void*)ori_path;

}

void OriPATHKernel::executeImpl(cudaStream_t stream) {
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is executing ...");
    // print dim
    // logger.INFO("-- launchGridDim: " + std::to_string(this->launchGridDim.x) + ", " + std::to_string(this->launchGridDim.y) + ", " + std::to_string(this->launchGridDim.z));
    // logger.INFO("-- launchBlockDim: " + std::to_string(this->launchBlockDim.x) + ", " + std::to_string(this->launchBlockDim.y) + ", " + std::to_string(this->launchBlockDim.z));
    
    CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
        launchGridDim, launchBlockDim,
        (void**)this->kernelParams.data(), this->smem, 0));

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}