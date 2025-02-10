// mrif-mriq-3-1
__global__ void mixed_mrif_mriq_kernel_3_1(int mrif0_numK, int mrif0_kGlobalIndex, float* mrif0_x, float* mrif0_y, float* mrif0_z, float* mrif0_outR, float* mrif0_outI, int mrif0_grid_dimension_x, int mrif0_grid_dimension_y, int mrif0_grid_dimension_z, int mrif0_block_dimension_x, int mrif0_block_dimension_y, int mrif0_block_dimension_z, int mrif0_ptb_start_block_pos, int mrif0_ptb_iter_block_step, int mrif0_ptb_end_block_pos, int mriq1_numK, int mriq1_kGlobalIndex, float* mriq1_x, float* mriq1_y, float* mriq1_z, float* mriq1_Qr, float* mriq1_Qi, int mriq1_grid_dimension_x, int mriq1_grid_dimension_y, int mriq1_grid_dimension_z, int mriq1_block_dimension_x, int mriq1_block_dimension_y, int mriq1_block_dimension_z, int mriq1_ptb_start_block_pos, int mriq1_ptb_iter_block_step, int mriq1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        general_ptb_mrif0(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 0 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_mrif1(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 1 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_mrif2(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 2 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 1024) {
        general_ptb_mriq0(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 0 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 1, mriq1_ptb_end_block_pos, 768
        );
    }

}
