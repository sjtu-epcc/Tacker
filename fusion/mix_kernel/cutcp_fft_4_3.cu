// cutcp-fft-4-3
__global__ void mixed_cutcp_fft_kernel_4_3(int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, float2* fft1_data, int fft1_grid_dimension_x, int fft1_grid_dimension_y, int fft1_grid_dimension_z, int fft1_block_dimension_x, int fft1_block_dimension_y, int fft1_block_dimension_z, int fft1_ptb_start_block_pos, int fft1_ptb_iter_block_step, int fft1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 4, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cutcp1(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 1 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 4, cutcp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cutcp2(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 2 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 4, cutcp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cutcp3(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 3 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 4, cutcp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_fft0(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 0 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_fft1(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 1 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_fft2(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 2 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 768
        );
    }

}
