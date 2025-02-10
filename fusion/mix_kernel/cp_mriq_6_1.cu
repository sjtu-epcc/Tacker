// cp-mriq-6-1
__global__ void mixed_cp_mriq_kernel_6_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, int mriq1_numK, int mriq1_kGlobalIndex, float* mriq1_x, float* mriq1_y, float* mriq1_z, float* mriq1_Qr, float* mriq1_Qi, int mriq1_grid_dimension_x, int mriq1_grid_dimension_y, int mriq1_grid_dimension_z, int mriq1_block_dimension_x, int mriq1_block_dimension_y, int mriq1_block_dimension_z, int mriq1_ptb_start_block_pos, int mriq1_ptb_iter_block_step, int mriq1_ptb_end_block_pos){
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
    else if (threadIdx.x < 1024) {
        general_ptb_mriq0(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 0 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 1, mriq1_ptb_end_block_pos, 768
        );
    }

}
