// lbm-mrif-2-3
__global__ void mixed_lbm_mrif_kernel_2_3(float* lbm0_srcGrid, float* lbm0_dstGrid, int lbm0_grid_dimension_x, int lbm0_grid_dimension_y, int lbm0_grid_dimension_z, int lbm0_block_dimension_x, int lbm0_block_dimension_y, int lbm0_block_dimension_z, int lbm0_ptb_start_block_pos, int lbm0_ptb_iter_block_step, int lbm0_ptb_end_block_pos, int mrif1_numK, int mrif1_kGlobalIndex, float* mrif1_x, float* mrif1_y, float* mrif1_z, float* mrif1_outR, float* mrif1_outI, int mrif1_grid_dimension_x, int mrif1_grid_dimension_y, int mrif1_grid_dimension_z, int mrif1_block_dimension_x, int mrif1_block_dimension_y, int mrif1_block_dimension_z, int mrif1_ptb_start_block_pos, int mrif1_ptb_iter_block_step, int mrif1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_lbm0(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 0 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 2, lbm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_lbm1(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 1 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 2, lbm0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_mrif0(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 0 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_mrif1(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 1 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 1024) {
        general_ptb_mrif2(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 2 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 768
        );
    }

}
