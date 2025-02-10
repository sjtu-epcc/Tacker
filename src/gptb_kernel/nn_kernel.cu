#include "header/nn_header.h"

extern "C" __global__ void general_ptb_nn(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
              int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) 
{   
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }
        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        int globalId = block_dimension_x * ( block_id_x * block_id_y + block_id_x ) + thread_id_x; // more efficient
        LatLong *latLong = d_locations+globalId;
        if (globalId < numRecords) {
            float *dist=d_distances+globalId;
            *dist = (float)sqrtf((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
        }
    }
}

extern "C" __forceinline__ __device__ void
internal_general_ptb_nn(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
              int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) 
{   
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }
        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        int globalId = block_dimension_x * ( block_id_x * block_id_y + block_id_x ) + thread_id_x; // more efficient
        LatLong *latLong = d_locations+globalId;
        if (globalId < numRecords) {
            float *dist=d_distances+globalId;
            *dist = (float)sqrtf((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
        }
    }
}