// cp-cutcp-1-6
__global__ void mixed_cp_cutcp_kernel_1_6(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, int cutcp1_binDim_x, int cutcp1_binDim_y, float4* cutcp1_binZeroAddr, float cutcp1_h, float cutcp1_cutoff2, float cutcp1_inv_cutoff2, float* cutcp1_regionZeroAddr, int cutcp1_zRegionIndex_t, int cutcp1_grid_dimension_x, int cutcp1_grid_dimension_y, int cutcp1_grid_dimension_z, int cutcp1_block_dimension_x, int cutcp1_block_dimension_y, int cutcp1_block_dimension_z, int cutcp1_ptb_start_block_pos, int cutcp1_ptb_iter_block_step, int cutcp1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 1, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cutcp0(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 0 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cutcp1(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 1 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cutcp2(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 2 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cutcp3(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 3 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_cutcp4(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 4 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_cutcp5(
            cutcp1_binDim_x, cutcp1_binDim_y, cutcp1_binZeroAddr, cutcp1_h, cutcp1_cutoff2, cutcp1_inv_cutoff2, cutcp1_regionZeroAddr, cutcp1_zRegionIndex_t, cutcp1_grid_dimension_x, cutcp1_grid_dimension_y, cutcp1_grid_dimension_z, cutcp1_block_dimension_x, cutcp1_block_dimension_y, cutcp1_block_dimension_z, cutcp1_ptb_start_block_pos + 5 * cutcp1_ptb_iter_block_step, cutcp1_ptb_iter_block_step * 6, cutcp1_ptb_end_block_pos, 768
        );
    }

}
