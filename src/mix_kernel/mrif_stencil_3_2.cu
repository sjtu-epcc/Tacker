#include "header/mrif_header.h"
#include "header/stencil_header.h"

// mrif
__device__ void mrif_stencil_mrif0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void mrif_stencil_mrif1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void mrif_stencil_mrif2(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

// stencil
__device__ void mrif_stencil_stencil0(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{
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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
            // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
        }
    }
}

__device__ void mrif_stencil_stencil1(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{
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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
        }
    }
}

// mrif-stencil-3-2
__global__ void mixed_mrif_stencil_kernel_3_2(int mrif0_numK, int mrif0_kGlobalIndex, float* mrif0_x, float* mrif0_y, float* mrif0_z, float* mrif0_outR, float* mrif0_outI, int mrif0_grid_dimension_x, int mrif0_grid_dimension_y, int mrif0_grid_dimension_z, int mrif0_block_dimension_x, int mrif0_block_dimension_y, int mrif0_block_dimension_z, int mrif0_ptb_start_block_pos, int mrif0_ptb_iter_block_step, int mrif0_ptb_end_block_pos, 
    float stencil1_c0, float stencil1_c1, float* stencil1_A0, float* stencil1_Anext, int stencil1_nx, int stencil1_ny, int stencil1_nz, int stencil1_grid_dimension_x, int stencil1_grid_dimension_y, int stencil1_grid_dimension_z, int stencil1_block_dimension_x, int stencil1_block_dimension_y, int stencil1_block_dimension_z, int stencil1_ptb_start_block_pos, int stencil1_ptb_iter_block_step, int stencil1_ptb_end_block_pos){
    if (threadIdx.x < 256) {
        mrif_stencil_mrif0(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 0 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 512) {
        mrif_stencil_mrif1(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 1 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 768) {
        mrif_stencil_mrif2(
            mrif0_numK, mrif0_kGlobalIndex, mrif0_x, mrif0_y, mrif0_z, mrif0_outR, mrif0_outI, mrif0_grid_dimension_x, mrif0_grid_dimension_y, mrif0_grid_dimension_z, mrif0_block_dimension_x, mrif0_block_dimension_y, mrif0_block_dimension_z, mrif0_ptb_start_block_pos + 2 * mrif0_ptb_iter_block_step, mrif0_ptb_iter_block_step * 3, mrif0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 896) {
        mrif_stencil_stencil0(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 0 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 2, stencil1_ptb_end_block_pos, 768
        );
    }
    else if (threadIdx.x < 1024) {
        mrif_stencil_stencil1(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 1 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 2, stencil1_ptb_end_block_pos, 896
        );
    }

}
