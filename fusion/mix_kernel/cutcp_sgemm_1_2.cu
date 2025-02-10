// cutcp-sgemm-1-2
__global__ void mixed_cutcp_sgemm_kernel_1_2(int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 1, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 256
        );
    }

}
