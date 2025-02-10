// cutcp-lbm-5-1
__global__ void mixed_cutcp_lbm_kernel_5_1(int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 5, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cutcp1(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 1 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 5, cutcp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cutcp2(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 2 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 5, cutcp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cutcp3(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 3 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 5, cutcp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cutcp4(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 4 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 5, cutcp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 640
        );
    }

}
