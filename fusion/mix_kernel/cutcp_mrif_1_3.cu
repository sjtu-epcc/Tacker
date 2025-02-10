// cutcp-mrif-1-3
__global__ void mixed_cutcp_mrif_kernel_1_3(int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, int mrif1_numK, int mrif1_kGlobalIndex, float* mrif1_x, float* mrif1_y, float* mrif1_z, float* mrif1_outR, float* mrif1_outI, int mrif1_grid_dimension_x, int mrif1_grid_dimension_y, int mrif1_grid_dimension_z, int mrif1_block_dimension_x, int mrif1_block_dimension_y, int mrif1_block_dimension_z, int mrif1_ptb_start_block_pos, int mrif1_ptb_iter_block_step, int mrif1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 1, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_mrif0(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 0 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_mrif1(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 1 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_mrif2(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 2 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 640
        );
    }

}
