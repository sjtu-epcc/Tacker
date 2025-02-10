// mrif-sgemm-1-2
__global__ void mixed_mrif_sgemm_kernel_1_2(int mrif0_numK, int mrif0_kGlobalIndex, float* mrif0_x, float* mrif0_y, float* mrif0_z, float* mrif0_outR, float* mrif0_outI, int mrif0_grid_dimension_x, int mrif0_grid_dimension_y, int mrif0_grid_dimension_z, int mrif0_block_dimension_x, int mrif0_block_dimension_y, int mrif0_block_dimension_z, int mrif0_ptb_start_block_pos, int mrif0_ptb_iter_block_step, int mrif0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        general_ptb_mrif0(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 0 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 1, mrif0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 2, sgemm1_ptb_end_block_pos, 384
        );
    }

}
