#include "header/cp_header.h"
#include "header/fft_header.h"

// cp
__device__ void cp_fft_cp0(int numatoms, float gridspacing, float * energygrid, 
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

__device__ void cp_fft_cp1(int numatoms, float gridspacing, float * energygrid, 
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


__device__ void cp_fft_cp2(int numatoms, float gridspacing, float * energygrid, 
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

// fft
__device__ void G_GPU_exchange_cp_fft_fft0( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_cp_fft_fft0(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_cp_fft_fft0( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void cp_fft_fft0(float2* data, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	
	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

	// // ori
	// int thread_id_x = threadIdx.x - thread_step;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

		// // ori
		// int block_id_x = block_pos;
        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

		float2 *ori_data = data + block_id_x * FFT_N;
		float2 v[FFT_R];
		// data = ori_data;

		int idxG = thread_id_x; 
		for (int r=0; r<FFT_R; r++) {  
			v[r] = ori_data[idxG + r*FFT_T];
		} 
		G_GPU_DoFft_cp_fft_fft0( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

// cp-fft-3-1
extern "C" __global__ void mixed_cp_fft_kernel_3_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float2* fft1_data, int fft1_grid_dimension_x, int fft1_grid_dimension_y, int fft1_grid_dimension_z, int fft1_block_dimension_x, int fft1_block_dimension_y, int fft1_block_dimension_z, int fft1_ptb_start_block_pos, int fft1_ptb_iter_block_step, int fft1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        cp_fft_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        cp_fft_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        cp_fft_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        cp_fft_fft0(
            fft1_data, fft1_grid_dimension_x, fft1_grid_dimension_y, fft1_grid_dimension_z, fft1_block_dimension_x, fft1_block_dimension_y, fft1_block_dimension_z, fft1_ptb_start_block_pos + 0 * fft1_ptb_iter_block_step, fft1_ptb_iter_block_step * 1, fft1_ptb_end_block_pos, 384
        );
    }

}
