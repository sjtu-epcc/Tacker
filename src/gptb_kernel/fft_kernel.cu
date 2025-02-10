#include "header/fft_header.h"

__device__ void G_GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	__syncthreads(); 

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	__syncthreads(); 


	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

extern "C" __global__ void g_general_ptb_fft(float2* data, 
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
		G_GPU_DoFft( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}


extern __device__ void G_GPU_exchange_fft_tzgemm_fft1_int( int2* v, int stride, int idxD, int incD, 
	int idxS, int incS);

extern __device__ void G_GPU_DoFft_fft_tzgemm_fft1_int(int2* v, int j, int stride);

extern "C" __global__ void g_general_ptb_fft_int(int2* data, 
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

		int2 *ori_data = data + block_id_x * FFT_N;
		int2 v[FFT_R];
		// data = ori_data;

		int idxG = thread_id_x; 
		for (int r=0; r<FFT_R; r++) {  
			v[r] = ori_data[idxG + r*FFT_T];
		} 
		G_GPU_DoFft_fft_tzgemm_fft1_int( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}