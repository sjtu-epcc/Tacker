#include "header/path_header.h"

extern "C" __global__ void general_ptb_path(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border,
                int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        } 
        __shared__ int prev[PATH_BLOCK_SIZE];
        __shared__ int result[PATH_BLOCK_SIZE];

        int bx = block_pos;
        int tx= thread_id_x;

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
        asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(256) : "memory"); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
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
            asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(256) : "memory");
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
            asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(256) : "memory"); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        }

        // update the global memory
        // after the last iteration, only threads coordinated within the 
        // small block perform the calculation and switch on ``computed''
        if (computed){
            gpuResults[xidx]=result[tx];		
        }
    }
}


extern "C" __device__ void internal_general_ptb_path(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border,
                int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        } 
        __shared__ int prev[PATH_BLOCK_SIZE];
        __shared__ int result[PATH_BLOCK_SIZE];

        int bx = block_pos;
        int tx= thread_id_x;

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
        asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(256) : "memory"); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
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
            asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(256) : "memory");
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
            asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(256) : "memory"); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        }

        // update the global memory
        // after the last iteration, only threads coordinated within the 
        // small block perform the calculation and switch on ``computed''
        if (computed){
            gpuResults[xidx]=result[tx];		
        }
    }
}