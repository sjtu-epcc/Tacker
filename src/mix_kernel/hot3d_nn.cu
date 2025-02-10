#include "header/hot3d_header.h"
#include "header/nn_header.h"

extern "C" __global__ void mixed_hot3d_nn_kernel(
    float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int hot3d0_grid_dimension_x, int hot3d0_grid_dimension_y, int hot3d0_grid_dimension_z, int hot3d0_block_dimension_x, int hot3d0_block_dimension_y, int hot3d0_block_dimension_z, int hot3d0_ptb_start_block_pos, int hot3d0_ptb_iter_block_step, int hot3d0_ptb_end_block_pos, 
        LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
        int nn0_grid_dimension_x, int nn0_grid_dimension_y, int nn0_grid_dimension_z, int nn0_block_dimension_x, int nn0_block_dimension_y, int nn0_block_dimension_z, int nn0_ptb_start_block_pos, int nn0_ptb_iter_block_step, int nn0_ptb_end_block_pos) {
    if (threadIdx.x < 256) {
        internal_general_ptb_hot3d(p, tIn, tOut, sdc,
            nx, ny, nz,
            ce, cw,
            cn, cs,
            ct, cb,
            cc,
            hot3d0_grid_dimension_x, hot3d0_grid_dimension_y, hot3d0_grid_dimension_z, hot3d0_block_dimension_x, hot3d0_block_dimension_y, hot3d0_block_dimension_z, 
			hot3d0_ptb_start_block_pos, hot3d0_ptb_iter_block_step, hot3d0_ptb_end_block_pos, 0);
    }
    else if (threadIdx.x < 384) {
        internal_general_ptb_nn(d_locations, d_distances, numRecords, lat, lng,
            nn0_grid_dimension_x, nn0_grid_dimension_y, nn0_grid_dimension_z, nn0_block_dimension_x, nn0_block_dimension_y, nn0_block_dimension_z, 
			nn0_ptb_start_block_pos, nn0_ptb_iter_block_step, nn0_ptb_end_block_pos, 256);
    }
}