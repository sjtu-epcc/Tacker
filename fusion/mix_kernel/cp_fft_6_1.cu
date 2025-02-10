// cp-fft-6-1
__global__ void mixed_cp_fft_kernel_6_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float2* fft1_data, int fft1_grid_dimension_x, int fft1_grid_dimension_y, int fft1_grid_dimension_z, int fft1_block_dimension_x, int fft1_block_dimension_y, int fft1_block_dimension_z, int fft1_ptb_start_block_pos, int fft1_ptb_iter_block_step, int fft1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cp4(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 4 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_cp5(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 5 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_fft0(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 0 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 1, fft1_ptb_end_block_pos, 768
        );
    }

}
