#include "header/cp_header.h"

// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.

extern "C" __global__ void g_general_ptb_cp(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        unsigned int xindex  = __umul24(block_id_x, block_dimension_x) * UNROLLX
                            + thread_id_x;
        unsigned int yindex  = __umul24(block_id_y, block_dimension_y) + thread_id_y;
        unsigned int outaddr = (__umul24(grid_dimension_x, block_dimension_x) * UNROLLX) * yindex
                                + xindex;

        float coory = gridspacing * yindex;
        float coorx = gridspacing * xindex;

        float energyvalx1=0.0f;
        float energyvalx2=0.0f;
        float energyvalx3=0.0f;
        float energyvalx4=0.0f;
        float energyvalx5=0.0f;
        float energyvalx6=0.0f;
        float energyvalx7=0.0f;
        float energyvalx8=0.0f;

        float gridspacing_u = gridspacing * BLOCKSIZEX;

        int atomid;
        for (atomid=0; atomid<numatoms; atomid++) {
            float dy = coory - atominfo[atomid].y;
            float dyz2 = (dy * dy) + atominfo[atomid].z;

            float dx1 = coorx - atominfo[atomid].x;
            float dx2 = dx1 + gridspacing_u;
            float dx3 = dx2 + gridspacing_u;
            float dx4 = dx3 + gridspacing_u;
            float dx5 = dx4 + gridspacing_u;
            float dx6 = dx5 + gridspacing_u;
            float dx7 = dx6 + gridspacing_u;
            float dx8 = dx7 + gridspacing_u;

            energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dyz2));
            energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dyz2));
            energyvalx3 += atominfo[atomid].w * (1.0f / sqrtf(dx3*dx3 + dyz2));
            energyvalx4 += atominfo[atomid].w * (1.0f / sqrtf(dx4*dx4 + dyz2));
            energyvalx5 += atominfo[atomid].w * (1.0f / sqrtf(dx5*dx5 + dyz2));
            energyvalx6 += atominfo[atomid].w * (1.0f / sqrtf(dx6*dx6 + dyz2));
            energyvalx7 += atominfo[atomid].w * (1.0f / sqrtf(dx7*dx7 + dyz2));
            energyvalx8 += atominfo[atomid].w * (1.0f / sqrtf(dx8*dx8 + dyz2));
        }

        energygrid[outaddr]   += energyvalx1;
        energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
        energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
        energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
        energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
        energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
        energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
        energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
    }
}

extern "C" __global__ void g_general_ptb_cp_int(int numatoms, float gridspacing, int * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        unsigned int xindex  = __umul24(block_id_x, block_dimension_x) * UNROLLX
                            + thread_id_x;
        unsigned int yindex  = __umul24(block_id_y, block_dimension_y) + thread_id_y;
        unsigned int outaddr = (__umul24(grid_dimension_x, block_dimension_x) * UNROLLX) * yindex
                                + xindex;

        int coory = gridspacing * yindex;
        int coorx = gridspacing * xindex;

        int energyvalx1=0;
        int energyvalx2=0;
        int energyvalx3=0;
        int energyvalx4=0;
        int energyvalx5=0;
        int energyvalx6=0;
        int energyvalx7=0;
        int energyvalx8=0;

        int gridspacing_u = gridspacing * BLOCKSIZEX;

        int atomid;
        for (atomid=0; atomid<numatoms; atomid++) {
            int dy = coory - atominfo[atomid].y;
            int dyz2 = (dy * dy) + atominfo[atomid].z;

            int dx1 = coorx - atominfo[atomid].x;
            int dx2 = dx1 + gridspacing_u;
            int dx3 = dx2 + gridspacing_u;
            int dx4 = dx3 + gridspacing_u;
            int dx5 = dx4 + gridspacing_u;
            int dx6 = dx5 + gridspacing_u;
            int dx7 = dx6 + gridspacing_u;
            int dx8 = dx7 + gridspacing_u;

            energyvalx1 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx1*dx1 + dyz2));
            energyvalx2 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx2*dx2 + dyz2));
            energyvalx3 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx3*dx3 + dyz2));
            energyvalx4 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx4*dx4 + dyz2));
            energyvalx5 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx5*dx5 + dyz2));
            energyvalx6 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx6*dx6 + dyz2));
            energyvalx7 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx7*dx7 + dyz2));
            energyvalx8 += atominfo_int[atomid].w * (1000 / (int)sqrtf(dx8*dx8 + dyz2));
        }

        energygrid[outaddr]   += energyvalx1;
        energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
        energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
        energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
        energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
        energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
        energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
        energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
    }
}