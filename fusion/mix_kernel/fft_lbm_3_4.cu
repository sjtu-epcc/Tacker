// fft-lbm-3-4
__global__ void mixed_fft_lbm_kernel_3_4(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_fft0(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 0 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_fft1(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 1 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_fft2(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 2 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 4, lbm1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_lbm1(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 1 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 4, lbm1_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_lbm2(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 2 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 4, lbm1_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_lbm3(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 3 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 4, lbm1_ptb_end_block_pos, 768
        );
    }

}
