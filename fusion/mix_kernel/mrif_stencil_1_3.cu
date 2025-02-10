// mrif-stencil-1-3
__global__ void mixed_mrif_stencil_kernel_1_3(int mrif0_numK, int mrif0_kGlobalIndex, float* mrif0_x, float* mrif0_y, float* mrif0_z, float* mrif0_outR, float* mrif0_outI, int mrif0_grid_dimension_x, int mrif0_grid_dimension_y, int mrif0_grid_dimension_z, int mrif0_block_dimension_x, int mrif0_block_dimension_y, int mrif0_block_dimension_z, int mrif0_ptb_start_block_pos, int mrif0_ptb_iter_block_step, int mrif0_ptb_end_block_pos, float stencil1_c0, float stencil1_c1, float* stencil1_A0, float* stencil1_Anext, int stencil1_nx, int stencil1_ny, int stencil1_nz, int stencil1_grid_dimension_x, int stencil1_grid_dimension_y, int stencil1_grid_dimension_z, int stencil1_block_dimension_x, int stencil1_block_dimension_y, int stencil1_block_dimension_z, int stencil1_ptb_start_block_pos, int stencil1_ptb_iter_block_step, int stencil1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        general_ptb_mrif0(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 0 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 1, mrif0_ptb_end_block_pos, 0
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
