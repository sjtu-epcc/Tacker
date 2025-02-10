#include "header/hot3d_header.h"
#include "header/lava_header.h"

extern "C" __global__ void mixed_hot3d_lava_kernel(
    float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int hot3d0_grid_dimension_x, int hot3d0_grid_dimension_y, int hot3d0_grid_dimension_z, int hot3d0_block_dimension_x, int hot3d0_block_dimension_y, int hot3d0_block_dimension_z, int hot3d0_ptb_start_block_pos, int hot3d0_ptb_iter_block_step, int hot3d0_ptb_end_block_pos, 
        par_str d_par_gpu,
        dim_str d_dim_gpu,
        box_str* d_box_gpu,
        FOUR_VECTOR* d_rv_gpu,
        float* d_qv_gpu,
        FOUR_VECTOR* d_fv_gpu, 
        int lava0_grid_dimension_x, int lava0_grid_dimension_y, int lava0_grid_dimension_z, int lava0_block_dimension_x, int lava0_block_dimension_y, int lava0_block_dimension_z, int lava0_ptb_start_block_pos, int lava0_ptb_iter_block_step, int lava0_ptb_end_block_pos
        ) {
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
    else if (threadIdx.x < 384) {
        internal_general_ptb_lava(d_par_gpu, d_dim_gpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
            lava0_grid_dimension_x, lava0_grid_dimension_y, lava0_grid_dimension_z, lava0_block_dimension_x, lava0_block_dimension_y, lava0_block_dimension_z, 
			lava0_ptb_start_block_pos, lava0_ptb_iter_block_step, lava0_ptb_end_block_pos, 256);
    }
}