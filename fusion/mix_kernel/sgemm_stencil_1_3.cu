// sgemm-stencil-1-3
__global__ void mixed_sgemm_stencil_kernel_1_3(float* sgemm0_A, float* sgemm0_B, float* sgemm0_C, int sgemm0_NORMAL_M, int sgemm0_NORMAL_N, int sgemm0_NORMAL_K, int sgemm0_grid_dimension_x, int sgemm0_grid_dimension_y, int sgemm0_grid_dimension_z, int sgemm0_block_dimension_x, int sgemm0_block_dimension_y, int sgemm0_block_dimension_z, int sgemm0_ptb_start_block_pos, int sgemm0_ptb_iter_block_step, int sgemm0_ptb_end_block_pos, float stencil1_c0, float stencil1_c1, float* stencil1_A0, float* stencil1_Anext, int stencil1_nx, int stencil1_ny, int stencil1_nz, int stencil1_grid_dimension_x, int stencil1_grid_dimension_y, int stencil1_grid_dimension_z, int stencil1_block_dimension_x, int stencil1_block_dimension_y, int stencil1_block_dimension_z, int stencil1_ptb_start_block_pos, int stencil1_ptb_iter_block_step, int stencil1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_sgemm0(
            sgemm0_A, sgemm0_B, sgemm0_C, sgemm0_NORMAL_M, sgemm0_NORMAL_N, sgemm0_NORMAL_K, sgemm0_grid_dimension_x, sgemm0_grid_dimension_y, sgemm0_grid_dimension_z, sgemm0_block_dimension_x, sgemm0_block_dimension_y, sgemm0_block_dimension_z, sgemm0_ptb_start_block_pos + 0 * sgemm0_ptb_iter_block_step, sgemm0_ptb_iter_block_step * 1, sgemm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_stencil0(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 0 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_stencil1(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 1 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_stencil2(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 2 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 384
        );
    }

}
