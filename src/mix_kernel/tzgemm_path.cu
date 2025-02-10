#include "header/path_header.h"
#include "header/tzgemm_header.h"

__global__ void path_tzgemm_mix(
        int iteration, 
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols, 
        int rows,
        int startStep,
        int border,
        int path0_grid_dimension_x, int path0_grid_dimension_y, int path0_grid_dimension_z, int path0_block_dimension_x, int path0_block_dimension_y, int path0_block_dimension_z, int path0_ptb_start_block_pos, int path0_ptb_iter_block_step, int path0_ptb_end_block_pos, 
#ifdef AKER_INT8
	int8_t *tzgemm1_A, int8_t *tzgemm1_B, int16_t *tzgemm1_C,
#else
	half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, 
#endif
        int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K, int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        internal_general_ptb_tzgemm(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, 
            tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K,
            tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z,  
            tzgemm1_ptb_start_block_pos, tzgemm1_ptb_iter_block_step, tzgemm1_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        internal_general_ptb_path(
            iteration, gpuWall, gpuSrc, gpuResults, cols, rows, startStep, border,
            path0_grid_dimension_x, path0_grid_dimension_y, path0_grid_dimension_z, path0_block_dimension_x, path0_block_dimension_y, path0_block_dimension_z, 
			path0_ptb_start_block_pos, path0_ptb_iter_block_step, path0_ptb_end_block_pos, 128
        );
    }
}