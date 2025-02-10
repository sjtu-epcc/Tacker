#include "header/lava_header.h"
#include "header/tzgemm_header.h"

__global__ void lava_tzgemm_mix(
        par_str d_par_gpu,
        dim_str d_dim_gpu,
        box_str* d_box_gpu,
        FOUR_VECTOR* d_rv_gpu,
        float* d_qv_gpu,
        FOUR_VECTOR* d_fv_gpu, 
        int lava0_grid_dimension_x, int lava0_grid_dimension_y, int lava0_grid_dimension_z, int lava0_block_dimension_x, int lava0_block_dimension_y, int lava0_block_dimension_z, int lava0_ptb_start_block_pos, int lava0_ptb_iter_block_step, int lava0_ptb_end_block_pos, 
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
    else if (threadIdx.x < 256) {
        internal_general_ptb_lava(
            d_par_gpu, d_dim_gpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
            lava0_grid_dimension_x, lava0_grid_dimension_y, lava0_grid_dimension_z, lava0_block_dimension_x, lava0_block_dimension_y, lava0_block_dimension_z, 
			lava0_ptb_start_block_pos, lava0_ptb_iter_block_step, lava0_ptb_end_block_pos, 128
        );
    }
}