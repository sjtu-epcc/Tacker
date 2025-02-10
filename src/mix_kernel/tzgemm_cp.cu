#include "header/cp_header.h"
#include "header/tzgemm_header.h"
#include <mma.h>
using namespace nvcuda; 

__device__ void cp_tzgemm_cp0(int numatoms, float gridspacing, float * energygrid, 
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

__device__ void cp_tzgemm_cp1(int numatoms, float gridspacing, float * energygrid, 
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

__device__ void cp_tzgemm_cp2(int numatoms, float gridspacing, float * energygrid, 
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

__device__ void cp_tzgemm_cp0_int(int numatoms, float gridspacing, int * energygrid, 
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
            int dy = coory - atominfo_int[atomid].y;
            int dyz2 = (dy * dy) + atominfo_int[atomid].z;

            int dx1 = coorx - atominfo_int[atomid].x;
            int dx2 = dx1 + gridspacing_u;
            int dx3 = dx2 + gridspacing_u;
            int dx4 = dx3 + gridspacing_u;
            int dx5 = dx4 + gridspacing_u;
            int dx6 = dx5 + gridspacing_u;
            int dx7 = dx6 + gridspacing_u;
            int dx8 = dx7 + gridspacing_u;

            energyvalx1 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx1*dx1 + dyz2));
            energyvalx2 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx2*dx2 + dyz2));
            energyvalx3 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx3*dx3 + dyz2));
            energyvalx4 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx4*dx4 + dyz2));
            energyvalx5 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx5*dx5 + dyz2));
            energyvalx6 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx6*dx6 + dyz2));
            energyvalx7 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx7*dx7 + dyz2));
            energyvalx8 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx8*dx8 + dyz2));
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

__device__ void cp_tzgemm_cp1_int(int numatoms, float gridspacing, int * energygrid, 
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
            int dy = coory - atominfo_int[atomid].y;
            int dyz2 = (dy * dy) + atominfo_int[atomid].z;

            int dx1 = coorx - atominfo_int[atomid].x;
            int dx2 = dx1 + gridspacing_u;
            int dx3 = dx2 + gridspacing_u;
            int dx4 = dx3 + gridspacing_u;
            int dx5 = dx4 + gridspacing_u;
            int dx6 = dx5 + gridspacing_u;
            int dx7 = dx6 + gridspacing_u;
            int dx8 = dx7 + gridspacing_u;

            energyvalx1 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx1*dx1 + dyz2));
            energyvalx2 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx2*dx2 + dyz2));
            energyvalx3 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx3*dx3 + dyz2));
            energyvalx4 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx4*dx4 + dyz2));
            energyvalx5 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx5*dx5 + dyz2));
            energyvalx6 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx6*dx6 + dyz2));
            energyvalx7 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx7*dx7 + dyz2));
            energyvalx8 += atominfo_int[atomid].w * (1 / (int)sqrtf(dx8*dx8 + dyz2));
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

__device__ inline void cp_tzgemm_tzgemm0(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += ptb_iter_block_step) {
		if (block_pos >= ptb_end_block_pos) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


			// These fragments will accumulate the result of A and B matrix fragment
			// multiplications along the K_GLOBAL dimension.
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					wmma::fill_fragment(c[i][j], 0.0f);
				}
			}

			// Select what warp copies what matrix to shared memory.
			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
			const half *warp_ptr = 
				warpId < (WARPS_PER_BLOCK / 2) 
					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

			// Go through the global K dimension by a fixed step at a time.
			#pragma unroll
			for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
				// Copy slices of the A and B matrices to shared memory.
				// The first half of the warps in the CTA copy the A matrix, 
				// the rest copy the B matrix.
				size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2)
						? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
						: (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

				// First half of the warp copies the first row / column of the matrix,
				// the second half of the warp copies the next.
				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
					+ (laneId % CHUNK_COPY_LINE_LANES);

				// Shift the second half of the warp to the next row / column in the
				// shared memory.
				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

				#pragma unroll
				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
					// Copy 16 bytes at once in each lane.
					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

					// Advance the global memory pointer and the shared memory index.
					lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
				}

				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;

				// Compute a grid of C matrix tiles in each warp.
				#pragma unroll
				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

					#pragma unroll
					for (int i = 0; i < WARP_COL_TILES; i++) {
						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

						#pragma unroll
						for (int j = 0; j < WARP_ROW_TILES; j++) {
							if (i == 0) {
								// Load the B matrix fragment once, because it is going to be
								// reused against the other A matrix fragments.
								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
							}
							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
						}
					}
				}
				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
			}

			// Store the D fragments to shared memory.
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					// Uniform, point-wise transformations of ALL fragment elements by ALL
					// threads in the warp are well-defined even though element indices
					// within fragment storage are not defined.
					#pragma unroll
					for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
				}
			}

			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
		}
}

__device__ void cp_tzgemm_tzgemm1(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += ptb_iter_block_step) {
		if (block_pos >= ptb_end_block_pos) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


			// These fragments will accumulate the result of A and B matrix fragment
			// multiplications along the K_GLOBAL dimension.
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					wmma::fill_fragment(c[i][j], 0.0f);
				}
			}

			// Select what warp copies what matrix to shared memory.
			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
			const half *warp_ptr = 
				warpId < (WARPS_PER_BLOCK / 2) 
					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

			// Go through the global K dimension by a fixed step at a time.
			#pragma unroll
			for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
				// Copy slices of the A and B matrices to shared memory.
				// The first half of the warps in the CTA copy the A matrix, 
				// the rest copy the B matrix.
				size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2)
						? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
						: (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

				// First half of the warp copies the first row / column of the matrix,
				// the second half of the warp copies the next.
				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
					+ (laneId % CHUNK_COPY_LINE_LANES);

				// Shift the second half of the warp to the next row / column in the
				// shared memory.
				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

				#pragma unroll
				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
					// Copy 16 bytes at once in each lane.
					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

					// Advance the global memory pointer and the shared memory index.
					lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
				}

				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

				// Compute a grid of C matrix tiles in each warp.
				#pragma unroll
				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

					#pragma unroll
					for (int i = 0; i < WARP_COL_TILES; i++) {
						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

						#pragma unroll
						for (int j = 0; j < WARP_ROW_TILES; j++) {
							if (i == 0) {
								// Load the B matrix fragment once, because it is going to be
								// reused against the other A matrix fragments.
								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
							}
							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
						}
					}
				}
				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
			}

			// Store the D fragments to shared memory.
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					// Uniform, point-wise transformations of ALL fragment elements by ALL
					// threads in the warp are well-defined even though element indices
					// within fragment storage are not defined.
					#pragma unroll
					for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
				}
			}

			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
		}
}

__device__ void cp_tzgemm_tzgemm2(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += ptb_iter_block_step) {
		if (block_pos >= ptb_end_block_pos) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


			// These fragments will accumulate the result of A and B matrix fragment
			// multiplications along the K_GLOBAL dimension.
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					wmma::fill_fragment(c[i][j], 0.0f);
				}
			}

			// Select what warp copies what matrix to shared memory.
			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
			const half *warp_ptr = 
				warpId < (WARPS_PER_BLOCK / 2) 
					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

			// Go through the global K dimension by a fixed step at a time.
			#pragma unroll
			for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
				// Copy slices of the A and B matrices to shared memory.
				// The first half of the warps in the CTA copy the A matrix, 
				// the rest copy the B matrix.
				size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2)
						? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
						: (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

				// First half of the warp copies the first row / column of the matrix,
				// the second half of the warp copies the next.
				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
					+ (laneId % CHUNK_COPY_LINE_LANES);

				// Shift the second half of the warp to the next row / column in the
				// shared memory.
				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

				#pragma unroll
				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
					// Copy 16 bytes at once in each lane.
					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

					// Advance the global memory pointer and the shared memory index.
					lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
				}

				asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");;

				// Compute a grid of C matrix tiles in each warp.
				#pragma unroll
				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

					#pragma unroll
					for (int i = 0; i < WARP_COL_TILES; i++) {
						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

						#pragma unroll
						for (int j = 0; j < WARP_ROW_TILES; j++) {
							if (i == 0) {
								// Load the B matrix fragment once, because it is going to be
								// reused against the other A matrix fragments.
								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
							}
							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
						}
					}
				}
				asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");;
			}

			// Store the D fragments to shared memory.
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					// Uniform, point-wise transformations of ALL fragment elements by ALL
					// threads in the warp are well-defined even though element indices
					// within fragment storage are not defined.
					#pragma unroll
					for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
				}
			}

			asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");;

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");;
		}
}

// __global__ void cp_tzgemm_mix(
//         int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, 
//         int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, 
//             half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K,
//             int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
//     // if (threadIdx.x == 0 && blockIdx.x == 0){
// 	// 	printf("cp blk range: %d - %d, tzgemm blk range: %d - %d\n", cp0_ptb_start_block_pos, cp0_ptb_end_block_pos, tzgemm1_ptb_start_block_pos, tzgemm1_ptb_end_block_pos);
// 	// }
// 	if (threadIdx.x < 128) {
//         cp_tzgemm_cp0(
//             cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 0
//         );
//     }
// 	else if (threadIdx.x < 256) {
// 		cp_tzgemm_cp1(
// 			cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 128
// 		);
// 	}
// 	else if (threadIdx.x < 384) {
// 		cp_tzgemm_tzgemm0(
//             tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 0 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 2, tzgemm1_ptb_end_block_pos, 256
//         );
// 	}
// 	else {
// 		cp_tzgemm_tzgemm1(
// 			tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 1 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 2, tzgemm1_ptb_end_block_pos, 384
// 		);
// 	}

// }

__global__ void cp_tzgemm_mix(
        int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, 
        int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, 
            half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K,
            int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    // if (threadIdx.x == 0 && blockIdx.x == 0){
	// 	printf("cp blk range: %d - %d, tzgemm blk range: %d - %d\n", cp0_ptb_start_block_pos, cp0_ptb_end_block_pos, tzgemm1_ptb_start_block_pos, tzgemm1_ptb_end_block_pos);
	// }
	if (threadIdx.x < 128) {
        cp_tzgemm_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos, cp0_ptb_iter_block_step, cp0_ptb_end_block_pos, 0
        );
    }
	// else if (threadIdx.x < 256) {
	// 	cp_tzgemm_cp1(
	// 		cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 128
	// 	);
	// }
	else if (threadIdx.x < 256) {
		cp_tzgemm_tzgemm0(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos, tzgemm1_ptb_iter_block_step, tzgemm1_ptb_end_block_pos, 128
        );
	}
	// else {
	// 	cp_tzgemm_tzgemm1(
	// 		tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 1 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 2, tzgemm1_ptb_end_block_pos, 384
	// 	);
	// }

}

// 对应的general_ptb_tzgemm ：ptxas info    : Used 107 registers, 18432 bytes smem, 428 bytes cmem[0]
// ptxas info    : Used 107 registers, 18432 bytes smem, 480 bytes cmem[0]
extern "C" __global__ void cp_tzgemm_mix_int(
        int cp0_numatoms, float cp0_gridspacing, int* cp0_energygrid, 
        int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, 
            half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K,
            int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
	if (threadIdx.x < 128) {
        cp_tzgemm_tzgemm0(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos, tzgemm1_ptb_iter_block_step, tzgemm1_ptb_end_block_pos, 0
        );
    }
	else if (threadIdx.x < 256) {
		cp_tzgemm_cp0_int(
			cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos, cp0_ptb_iter_block_step * 1, cp0_ptb_end_block_pos, 128
		);
    }
	// else if (threadIdx.x < 384) {
	// 	cp_tzgemm_cp1_int(
	// 		cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 256
	// 	);
    //     // printf("tzgemm-384-branch...\n");
	// }
	// else if (threadIdx.x < 256) {
	// 	cp_tzgemm_tzgemm1(
	// 		tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 1 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 1, tzgemm1_ptb_end_block_pos, 128
	// 	);
    //     // printf("tzgemm-else-branch...\n");
	// } 
	//  else {
	// 	cp_tzgemm_tzgemm2(
	// 		tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 2 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 3, tzgemm1_ptb_end_block_pos, 384
	// 	);
    // }

}
// 3个branch会超