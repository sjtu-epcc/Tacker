#include "header/lava_header.h"
#include "header/path_header.h"

extern "C" __global__ void mixed_lava_path_kernel(
    par_str d_par_gpu,
        dim_str d_dim_gpu,
        box_str* d_box_gpu,
        FOUR_VECTOR* d_rv_gpu,
        float* d_qv_gpu,
        FOUR_VECTOR* d_fv_gpu, 
        int lava0_grid_dimension_x, int lava0_grid_dimension_y, int lava0_grid_dimension_z, int lava0_block_dimension_x, int lava0_block_dimension_y, int lava0_block_dimension_z, int lava0_ptb_start_block_pos, int lava0_ptb_iter_block_step, int lava0_ptb_end_block_pos,  
        int iteration, 
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols, 
        int rows,
        int startStep,
        int border,
        int path0_grid_dimension_x, int path0_grid_dimension_y, int path0_grid_dimension_z, int path0_block_dimension_x, int path0_block_dimension_y, int path0_block_dimension_z, int path0_ptb_start_block_pos, int path0_ptb_iter_block_step, int path0_ptb_end_block_pos) {
    if (threadIdx.x < 128) {
        internal_general_ptb_lava(d_par_gpu, d_dim_gpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
            lava0_grid_dimension_x, lava0_grid_dimension_y, lava0_grid_dimension_z, lava0_block_dimension_x, lava0_block_dimension_y, lava0_block_dimension_z, 
			lava0_ptb_start_block_pos, lava0_ptb_iter_block_step, lava0_ptb_end_block_pos, 0);
    }
    else if (threadIdx.x < 384) {
        internal_general_ptb_path(iteration, gpuWall, gpuSrc, gpuResults, cols, rows, startStep, border,
            path0_grid_dimension_x, path0_grid_dimension_y, path0_grid_dimension_z, path0_block_dimension_x, path0_block_dimension_y, path0_block_dimension_z, 
			path0_ptb_start_block_pos, path0_ptb_iter_block_step, path0_ptb_end_block_pos, 128);
    }
}