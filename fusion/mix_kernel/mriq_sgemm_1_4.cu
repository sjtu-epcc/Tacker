// mriq-sgemm-1-4
__global__ void mixed_mriq_sgemm_kernel_1_4(int mriq0_numK, int mriq0_kGlobalIndex, float* mriq0_x, float* mriq0_y, float* mriq0_z, float* mriq0_Qr, float* mriq0_Qi, int mriq0_grid_dimension_x, int mriq0_grid_dimension_y, int mriq0_grid_dimension_z, int mriq0_block_dimension_x, int mriq0_block_dimension_y, int mriq0_block_dimension_z, int mriq0_ptb_start_block_pos, int mriq0_ptb_iter_block_step, int mriq0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        general_ptb_mriq0(
            mriq0_numK, mriq0_kGlobalIndex, mriq0_x, mriq0_y, mriq0_z, mriq0_Qr, mriq0_Qi, mriq0_grid_dimension_x, mriq0_grid_dimension_y, mriq0_grid_dimension_z, mriq0_block_dimension_x, mriq0_block_dimension_y, mriq0_block_dimension_z, mriq0_ptb_start_block_pos + 0 * mriq0_ptb_iter_block_step, mriq0_ptb_iter_block_step * 1, mriq0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_sgemm1(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 1 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_sgemm2(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 2 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_sgemm3(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 3 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 4, sgemm1_ptb_end_block_pos, 640
        );
    }

}
