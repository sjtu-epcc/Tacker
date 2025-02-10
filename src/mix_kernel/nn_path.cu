#include "header/path_header.h"
#include "header/nn_header.h"

__global__ void mixed_nn_path_kernel(
    LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
        int nn0_grid_dimension_x, int nn0_grid_dimension_y, int nn0_grid_dimension_z, int nn0_block_dimension_x, int nn0_block_dimension_y, int nn0_block_dimension_z, int nn0_ptb_start_block_pos, int nn0_ptb_iter_block_step, int nn0_ptb_end_block_pos,
        int iteration, 
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols, 
        int rows,
        int startStep,
        int border,
        int path0_grid_dimension_x, int path0_grid_dimension_y, int path0_grid_dimension_z, int path0_block_dimension_x, int path0_block_dimension_y, int path0_block_dimension_z, int path0_ptb_start_block_pos, int path0_ptb_iter_block_step, int path0_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        internal_general_ptb_nn(
            d_locations, d_distances, numRecords, lat, lng,
            nn0_grid_dimension_x, nn0_grid_dimension_y, nn0_grid_dimension_z, nn0_block_dimension_x, nn0_block_dimension_y, nn0_block_dimension_z, 
			nn0_ptb_start_block_pos, nn0_ptb_iter_block_step, nn0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        internal_general_ptb_path(
            iteration, gpuWall, gpuSrc, gpuResults, cols, rows, startStep, border,
            path0_grid_dimension_x, path0_grid_dimension_y, path0_grid_dimension_z, path0_block_dimension_x, path0_block_dimension_y, path0_block_dimension_z, 
			path0_ptb_start_block_pos, path0_ptb_iter_block_step, path0_ptb_end_block_pos, 128
        );
    }
}