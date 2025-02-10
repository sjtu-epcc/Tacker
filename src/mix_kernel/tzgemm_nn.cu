#include "header/nn_header.h"
#include "header/tzgemm_header.h"

__global__ void nn_tzgemm_mix(
        LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
        int nn0_grid_dimension_x, int nn0_grid_dimension_y, int nn0_grid_dimension_z, int nn0_block_dimension_x, int nn0_block_dimension_y, int nn0_block_dimension_z, int nn0_ptb_start_block_pos, int nn0_ptb_iter_block_step, int nn0_ptb_end_block_pos, 
#ifdef AKER_INT8
	int8_t *tzgemm1_A, int8_t *tzgemm1_B, int16_t *tzgemm1_C,
#else
	half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, 
#endif
        int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K, int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        internal_general_ptb_tzgemm(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, 
            tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K,
            tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z,  
            tzgemm1_ptb_start_block_pos, tzgemm1_ptb_iter_block_step, tzgemm1_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        internal_general_ptb_nn(
            d_locations, d_distances, numRecords, lat, lng,
            nn0_grid_dimension_x, nn0_grid_dimension_y, nn0_grid_dimension_z, nn0_block_dimension_x, nn0_block_dimension_y, nn0_block_dimension_z, 
			nn0_ptb_start_block_pos, nn0_ptb_iter_block_step, nn0_ptb_end_block_pos, 128
        );
    }
}