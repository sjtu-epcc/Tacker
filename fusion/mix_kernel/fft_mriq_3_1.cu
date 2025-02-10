// fft-mriq-3-1
__global__ void mixed_fft_mriq_kernel_3_1(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, int mriq1_numK, int mriq1_kGlobalIndex, float* mriq1_x, float* mriq1_y, float* mriq1_z, float* mriq1_Qr, float* mriq1_Qi, int mriq1_grid_dimension_x, int mriq1_grid_dimension_y, int mriq1_grid_dimension_z, int mriq1_block_dimension_x, int mriq1_block_dimension_y, int mriq1_block_dimension_z, int mriq1_ptb_start_block_pos, int mriq1_ptb_iter_block_step, int mriq1_ptb_end_block_pos){
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
    else if (threadIdx.x < 640) {
        general_ptb_mriq0(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 0 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 1, mriq1_ptb_end_block_pos, 384
        );
    }

}
