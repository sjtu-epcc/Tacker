// lbm-sgemm-1-3
__global__ void mixed_lbm_sgemm_kernel_1_3(float* lbm0_srcGrid, float* lbm0_dstGrid, int lbm0_grid_dimension_x, int lbm0_grid_dimension_y, int lbm0_grid_dimension_z, int lbm0_block_dimension_x, int lbm0_block_dimension_y, int lbm0_block_dimension_z, int lbm0_ptb_start_block_pos, int lbm0_ptb_iter_block_step, int lbm0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_lbm0(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 0 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 1, lbm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 3, sgemm1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 3, sgemm1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_sgemm2(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 2 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 3, sgemm1_ptb_end_block_pos, 384
        );
    }

}
