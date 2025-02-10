#include "header/fft_header.h"
#include "header/stencil_header.h"

// fft
__device__ void G_GPU_exchange_fft_stencil_fft0( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_stencil_fft0(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_stencil_fft0( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_stencil_fft0(float2* data, 
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
		G_GPU_DoFft_fft_stencil_fft0( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_stencil_fft1( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_stencil_fft1(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_stencil_fft1( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_stencil_fft1(float2* data, 
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
		G_GPU_DoFft_fft_stencil_fft1( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}
// step_size == launch param == ptb worker num == SM_NUM * ptb_per_sm_number

__device__ void G_GPU_exchange_fft_stencil_fft2( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_stencil_fft2(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_stencil_fft2( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_stencil_fft2(float2* data, 
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
		G_GPU_DoFft_fft_stencil_fft2( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_stencil_fft3( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_stencil_fft3(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_stencil_fft3( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_stencil_fft3(float2* data, 
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
		G_GPU_DoFft_fft_stencil_fft3( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_stencil_fft4( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_stencil_fft4(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_stencil_fft4( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_stencil_fft4(float2* data, 
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
		G_GPU_DoFft_fft_stencil_fft4( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

// stencil
__device__ void fft_stencil_stencil0(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
            // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
        }
    }
}

__device__ void fft_stencil_stencil1(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
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
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
        }
    }
}

__device__ void fft_stencil_stencil2(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
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
asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(128) : "memory");
        }
    }
}



// fft-stencil-5-3
__global__ void mixed_fft_stencil_kernel_5_3(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, float stencil1_c0, float stencil1_c1, float* stencil1_A0, float* stencil1_Anext, int stencil1_nx, int stencil1_ny, int stencil1_nz, int stencil1_grid_dimension_x, int stencil1_grid_dimension_y, int stencil1_grid_dimension_z, int stencil1_block_dimension_x, int stencil1_block_dimension_y, int stencil1_block_dimension_z, int stencil1_ptb_start_block_pos, int stencil1_ptb_iter_block_step, int stencil1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        fft_stencil_fft0(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 0 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 5, fft0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        fft_stencil_fft1(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 1 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 5, fft0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        fft_stencil_fft2(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 2 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 5, fft0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        fft_stencil_fft3(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 3 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 5, fft0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        fft_stencil_fft4(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 4 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 5, fft0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        fft_stencil_stencil0(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 0 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        fft_stencil_stencil1(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 1 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 768
        );
    }
    else if (threadIdx.x < 1024) {
        fft_stencil_stencil2(
            stencil1_c0, stencil1_c1, stencil1_A0, stencil1_Anext, stencil1_nx, stencil1_ny, stencil1_nz, stencil1_grid_dimension_x, stencil1_grid_dimension_y, stencil1_grid_dimension_z, stencil1_block_dimension_x, stencil1_block_dimension_y, stencil1_block_dimension_z, stencil1_ptb_start_block_pos + 2 * stencil1_ptb_iter_block_step, stencil1_ptb_iter_block_step * 3, stencil1_ptb_end_block_pos, 896
        );
    }

}
