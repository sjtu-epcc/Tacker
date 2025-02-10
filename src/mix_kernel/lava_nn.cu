#include "header/lava_header.h"
#include "header/nn_header.h"

extern "C" __global__ void mixed_lava_nn_kernel(
    par_str d_par_gpu,
        dim_str d_dim_gpu,
        box_str* d_box_gpu,
        FOUR_VECTOR* d_rv_gpu,
        float* d_qv_gpu,
        FOUR_VECTOR* d_fv_gpu, 
        int lava0_grid_dimension_x, int lava0_grid_dimension_y, int lava0_grid_dimension_z, int lava0_block_dimension_x, int lava0_block_dimension_y, int lava0_block_dimension_z, int lava0_ptb_start_block_pos, int lava0_ptb_iter_block_step, int lava0_ptb_end_block_pos,  
        LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
        int nn0_grid_dimension_x, int nn0_grid_dimension_y, int nn0_grid_dimension_z, int nn0_block_dimension_x, int nn0_block_dimension_y, int nn0_block_dimension_z, int nn0_ptb_start_block_pos, int nn0_ptb_iter_block_step, int nn0_ptb_end_block_pos) {
    if (threadIdx.x < 128) {
        internal_general_ptb_lava(d_par_gpu, d_dim_gpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
            lava0_grid_dimension_x, lava0_grid_dimension_y, lava0_grid_dimension_z, lava0_block_dimension_x, lava0_block_dimension_y, lava0_block_dimension_z, 
			lava0_ptb_start_block_pos, lava0_ptb_iter_block_step, lava0_ptb_end_block_pos, 0);
    }
    else if (threadIdx.x < 256) {
        internal_general_ptb_nn(d_locations, d_distances, numRecords, lat, lng,
            nn0_grid_dimension_x, nn0_grid_dimension_y, nn0_grid_dimension_z, nn0_block_dimension_x, nn0_block_dimension_y, nn0_block_dimension_z, 
			nn0_ptb_start_block_pos, nn0_ptb_iter_block_step, nn0_ptb_end_block_pos, 127);
    }
}