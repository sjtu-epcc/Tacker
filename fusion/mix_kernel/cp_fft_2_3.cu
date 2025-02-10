// cp-fft-2-3
__global__ void mixed_cp_fft_kernel_2_3(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float2* fft1_data, int fft1_grid_dimension_x, int fft1_grid_dimension_y, int fft1_grid_dimension_z, int fft1_block_dimension_x, int fft1_block_dimension_y, int fft1_block_dimension_z, int fft1_ptb_start_block_pos, int fft1_ptb_iter_block_step, int fft1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_fft0(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 0 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_fft1(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 1 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_fft2(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 2 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 3, fft1_ptb_end_block_pos, 512
        );
    }

}
