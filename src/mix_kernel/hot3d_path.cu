#include "header/hot3d_header.h"
#include "header/path_header.h"

extern "C" __global__ void mixed_hot3d_path_kernel(
    float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int hot3d0_grid_dimension_x, int hot3d0_grid_dimension_y, int hot3d0_grid_dimension_z, int hot3d0_block_dimension_x, int hot3d0_block_dimension_y, int hot3d0_block_dimension_z, int hot3d0_ptb_start_block_pos, int hot3d0_ptb_iter_block_step, int hot3d0_ptb_end_block_pos, 
        int iteration, 
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols, 
        int rows,
        int startStep,
        int border,
        int path0_grid_dimension_x, int path0_grid_dimension_y, int path0_grid_dimension_z, int path0_block_dimension_x, int path0_block_dimension_y, int path0_block_dimension_z, int path0_ptb_start_block_pos, int path0_ptb_iter_block_step, int path0_ptb_end_block_pos) {
    if (threadIdx.x < 256) {
        internal_general_ptb_hot3d(p, tIn, tOut, sdc,
            nx, ny, nz,
            ce, cw,
            cn, cs,
            ct, cb,
            cc,
            hot3d0_grid_dimension_x, hot3d0_grid_dimension_y, hot3d0_grid_dimension_z, hot3d0_block_dimension_x, hot3d0_block_dimension_y, hot3d0_block_dimension_z, 
			hot3d0_ptb_start_block_pos, hot3d0_ptb_iter_block_step, hot3d0_ptb_end_block_pos, 0);
    }
    else if (threadIdx.x < 512) {
        internal_general_ptb_path(iteration, gpuWall, gpuSrc, gpuResults, cols, rows, startStep, border,
            path0_grid_dimension_x, path0_grid_dimension_y, path0_grid_dimension_z, path0_block_dimension_x, path0_block_dimension_y, path0_block_dimension_z, 
			path0_ptb_start_block_pos, path0_ptb_iter_block_step, path0_ptb_end_block_pos, 256);
    }
}