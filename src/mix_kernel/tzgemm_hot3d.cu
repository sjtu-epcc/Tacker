#include "header/hot3d_header.h"
#include "header/tzgemm_header.h"

__global__ void hot3d_tzgemm_mix(
        float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int hot3d0_grid_dimension_x, int hot3d0_grid_dimension_y, int hot3d0_grid_dimension_z, int hot3d0_block_dimension_x, int hot3d0_block_dimension_y, int hot3d0_block_dimension_z, int hot3d0_ptb_start_block_pos, int hot3d0_ptb_iter_block_step, int hot3d0_ptb_end_block_pos, 
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
        internal_general_ptb_hot3d(
            p, tIn, tOut, sdc,
            nx, ny, nz,
            ce, cw,
            cn, cs,
            ct, cb,
            cc,
            hot3d0_grid_dimension_x, hot3d0_grid_dimension_y, hot3d0_grid_dimension_z, hot3d0_block_dimension_x, hot3d0_block_dimension_y, hot3d0_block_dimension_z, 
			hot3d0_ptb_start_block_pos, hot3d0_ptb_iter_block_step, hot3d0_ptb_end_block_pos, 128
        );
    }
}