// mriq-stencil-1-3
__global__ void mixed_mriq_stencil_kernel_1_3(int mriq0_numK, int mriq0_kGlobalIndex, float* mriq0_x, float* mriq0_y, float* mriq0_z, float* mriq0_Qr, float* mriq0_Qi, int mriq0_grid_dimension_x, int mriq0_grid_dimension_y, int mriq0_grid_dimension_z, int mriq0_block_dimension_x, int mriq0_block_dimension_y, int mriq0_block_dimension_z, int mriq0_ptb_start_block_pos, int mriq0_ptb_iter_block_step, int mriq0_ptb_end_block_pos, float stencil1_c0, float stencil1_c1, float* stencil1_A0, float* stencil1_Anext, int stencil1_nx, int stencil1_ny, int stencil1_nz, int stencil1_grid_dimension_x, int stencil1_grid_dimension_y, int stencil1_grid_dimension_z, int stencil1_block_dimension_x, int stencil1_block_dimension_y, int stencil1_block_dimension_z, int stencil1_ptb_start_block_pos, int stencil1_ptb_iter_block_step, int stencil1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        general_ptb_mriq0(
            mriq0_numK, mriq0_kGlobalIndex, mriq0_x, mriq0_y, mriq0_z, mriq0_Qr, mriq0_Qi, mriq0_grid_dimension_x, mriq0_grid_dimension_y, mriq0_grid_dimension_z, mriq0_block_dimension_x, mriq0_block_dimension_y, mriq0_block_dimension_z, mriq0_ptb_start_block_pos + 0 * mriq0_ptb_iter_block_step, mriq0_ptb_iter_block_step * 1, mriq0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_stencil0(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 0 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_stencil1(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 1 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_stencil2(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 2 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 512
        );
    }

}
