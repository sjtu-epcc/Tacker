#include "header/stencil_header.h"

extern "C" __global__ void general_ptb_stencil(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    //shared memeory
    __shared__ float sh_A0[tile_x * tile_y * 2];

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        //thread coarsening along x direction
        const int i = block_id_x*block_dimension_x*2+thread_id_x;
        const int i2= block_id_x*block_dimension_x*2+thread_id_x+block_dimension_x;
        const int j = block_id_y*block_dimension_y+thread_id_y;
        const int sh_id=thread_id_x + thread_id_y*block_dimension_x*2;
        const int sh_id2=thread_id_x +block_dimension_x+ thread_id_y*block_dimension_x*2;

        sh_A0[sh_id]=0.0f;
        sh_A0[sh_id2]=0.0f;
        // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
__syncthreads();

        //get available region for load and store
        const bool w_region =  i>0 && j>0 &&(i<(nx-1)) &&(j<(ny-1)) ;
        const bool w_region2 =  j>0 &&(i2<nx-1) &&(j<ny-1) ;
        const bool x_l_bound = (thread_id_x==0);
        const bool x_h_bound = ((thread_id_x+block_dimension_x)==(block_dimension_x*2-1));
        const bool y_l_bound = (thread_id_y==0);
        const bool y_h_bound = (thread_id_y==(block_dimension_y-1));

        //register for bottom and top planes
        //because of thread coarsening, we need to doulbe registers
        float bottom=0.0f,bottom2=0.0f,top=0.0f,top2=0.0f;

        //load data for bottom and current 
        if((i<nx) &&(j<ny))
        {
            bottom=A0[Index3D (nx, ny, i, j, 0)];
            sh_A0[sh_id]=A0[Index3D (nx, ny, i, j, 1)];
        }
        if((i2<nx) &&(j<ny))
        {
            bottom2=A0[Index3D (nx, ny, i2, j, 0)];
            sh_A0[sh_id2]=A0[Index3D (nx, ny, i2, j, 1)];
        }

        // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
__syncthreads();
        
        for(int k=1;k<nz-1;k++)
        {
            float a_left_right,a_up,a_down;		
            
            //load required data on xy planes
            //if it on shared memory, load from shared memory
            //if not, load from global memory
            if((i<nx) &&(j<ny))
                top=A0[Index3D (nx, ny, i, j, k+1)];
                
            if(w_region)
            {
                a_up        =y_h_bound?A0[Index3D (nx, ny, i, j+1, k )]:sh_A0[sh_id+2*block_dimension_x];
                a_down      =y_l_bound?A0[Index3D (nx, ny, i, j-1, k )]:sh_A0[sh_id-2*block_dimension_x];
                a_left_right=x_l_bound?A0[Index3D (nx, ny, i-1, j, k )]:sh_A0[sh_id-1];
        
                Anext[Index3D (nx, ny, i, j, k)] = (top + bottom + a_up + a_down + sh_A0[sh_id+1] +a_left_right)*c1
                                            -  sh_A0[sh_id]*c0;		
            }
            
            //load another block 
            if((i2<nx) &&(j<ny))
                top2=A0[Index3D (nx, ny, i2, j, k+1)];
                
            if(w_region2)
            {
                a_up        =y_h_bound?A0[Index3D (nx, ny, i2, j+1, k )]:sh_A0[sh_id2+2*block_dimension_x];
                a_down      =y_l_bound?A0[Index3D (nx, ny, i2, j-1, k )]:sh_A0[sh_id2-2*block_dimension_x];
                a_left_right=x_h_bound?A0[Index3D (nx, ny, i2+1, j, k )]:sh_A0[sh_id2+1];

                Anext[Index3D (nx, ny, i2, j, k)] = (top2 + bottom2 + a_up + a_down + a_left_right +sh_A0[sh_id2-1])*c1
                                            -  sh_A0[sh_id2]*c0;
            }

            //swap data
            // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
__syncthreads();
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
            // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
__syncthreads();
        }
    }
}