// fft-mrif-3-2
__global__ void mixed_fft_mrif_kernel_3_2(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, int mrif1_numK, int mrif1_kGlobalIndex, float* mrif1_x, float* mrif1_y, float* mrif1_z, float* mrif1_outR, float* mrif1_outI, int mrif1_grid_dimension_x, int mrif1_grid_dimension_y, int mrif1_grid_dimension_z, int mrif1_block_dimension_x, int mrif1_block_dimension_y, int mrif1_block_dimension_z, int mrif1_ptb_start_block_pos, int mrif1_ptb_iter_block_step, int mrif1_ptb_end_block_pos){
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
        general_ptb_mrif0(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 0 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 2, mrif1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_mrif1(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 1 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 2, mrif1_ptb_end_block_pos, 640
        );
    }

}
