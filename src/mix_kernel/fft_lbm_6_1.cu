#include "header/lbm_header.h"
#include "header/fft_header.h"

// fft
__device__ void G_GPU_exchange_fft_lbm_fft0( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_lbm_fft0(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft0( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft0(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft0( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_lbm_fft1( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_lbm_fft1(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft1( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft1(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft1( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}
// step_size == launch param == ptb worker num == SM_NUM * ptb_per_sm_number

__device__ void G_GPU_exchange_fft_lbm_fft2( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_lbm_fft2(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft2( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft2(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft2( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_lbm_fft3( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_lbm_fft3(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft3( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft3(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft3( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_lbm_fft4( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_lbm_fft4(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft4( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft4(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft4( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_lbm_fft5( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	// __syncthreads(); 
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void G_GPU_DoFft_fft_lbm_fft5(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_lbm_fft5( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_lbm_fft5(float2* data, 
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
		G_GPU_DoFft_fft_lbm_fft5( v, thread_id_x );  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

// lbm
__device__ void fft_lbm_lbm0( float* srcGrid, float* dstGrid,
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z,
    int block_dimension_x, int block_dimension_y, int block_dimension_z,
    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
	//Using some predefined macros here.  Consider this the declaration 
    //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

    // // ori
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_step) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = ((threadIdx.x - thread_step) / block_dimension_x) / block_dimension_y;

    SWEEP_VAR
	float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
	float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
	float tempSB, tempET, tempEB, tempWT, tempWB;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        // ori
        // int block_id_x = block_pos % grid_dimension_x;
        // int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = (block_pos / grid_dimension_x) / grid_dimension_y;

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        // float *d_temp = srcGrid;
        // srcGrid = dstGrid;
        // dstGrid = d_temp;

        SWEEP_X = thread_id_x;
        SWEEP_Y = block_id_x;
        SWEEP_Z = block_id_y;

        //Load all of the input fields
        //This is a gather operation of the SCATTER preprocessor variable
            // is undefined in layout_config.h, or a "local" read otherwise
        tempC = SRC_C(srcGrid);
        tempN = SRC_N(srcGrid);
        tempS = SRC_S(srcGrid);
        tempE = SRC_E(srcGrid);
        tempW = SRC_W(srcGrid);
        tempT = SRC_T(srcGrid);
        tempB = SRC_B(srcGrid);
        tempNE= SRC_NE(srcGrid);
        tempNW= SRC_NW(srcGrid);
        tempSE = SRC_SE(srcGrid);
        tempSW = SRC_SW(srcGrid);
        tempNT = SRC_NT(srcGrid);
        tempNB = SRC_NB(srcGrid);
        tempST = SRC_ST(srcGrid);
        tempSB = SRC_SB(srcGrid);
        tempET = SRC_ET(srcGrid);
        tempEB = SRC_EB(srcGrid);
        tempWT = SRC_WT(srcGrid);
        tempWB = SRC_WB(srcGrid);

        //Test whether the cell is fluid or obstacle
        if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
            //Swizzle the inputs: reflect any fluid coming into this cell 
            // back to where it came from
            temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
            temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
            temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
            temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
            temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
            temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
            temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
            temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
            temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
        }
        else {
            //The math meat of LBM: ignore for optimization
            float ux, uy, uz, rho, u2;
            float temp1, temp2, temp_base;
            rho = tempC + tempN
                + tempS + tempE
                + tempW + tempT
                + tempB + tempNE
                + tempNW + tempSE
                + tempSW + tempNT
                + tempNB + tempST
                + tempSB + tempET
                + tempEB + tempWT
                + tempWB;

            ux = + tempE - tempW
                + tempNE - tempNW
                + tempSE - tempSW
                + tempET + tempEB
                - tempWT - tempWB;
            uy = + tempN - tempS
                + tempNE + tempNW
                - tempSE - tempSW
                + tempNT + tempNB
                - tempST - tempSB;
            uz = + tempT - tempB
                + tempNT - tempNB
                + tempST - tempSB
                + tempET - tempEB
                + tempWT - tempWB;

            ux /= rho;
            uy /= rho;
            uz /= rho;
            if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
                ux = 0.005f;
                uy = 0.002f;
                uz = 0.000f;
            }
            u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
            temp_base = OMEGA*rho;
            temp1 = DFL1*temp_base;


            //Put the output values for this cell in the shared memory
            temp_base = OMEGA*rho;
            temp1 = DFL1*temp_base;
            temp2 = 1.0f-OMEGA;
            tempC = temp2*tempC + temp1*(                                 - u2);
                temp1 = DFL2*temp_base;	
            tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
            tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
            tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
            tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
            tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
            tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
            temp1 = DFL3*temp_base;
            tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
            tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
            tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
            tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
            tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
            tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
            tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
            tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
            tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
            tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
            tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
            tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
        }

        //Write the results computed above
        //This is a scatter operation of the SCATTER preprocessor variable
            // is defined in layout_config.h, or a "local" write otherwise
        DST_C ( dstGrid ) = tempC;

        DST_N ( dstGrid ) = tempN; 
        DST_S ( dstGrid ) = tempS;
        DST_E ( dstGrid ) = tempE;
        DST_W ( dstGrid ) = tempW;
        DST_T ( dstGrid ) = tempT;
        DST_B ( dstGrid ) = tempB;

        DST_NE( dstGrid ) = tempNE;
        DST_NW( dstGrid ) = tempNW;
        DST_SE( dstGrid ) = tempSE;
        DST_SW( dstGrid ) = tempSW;
        DST_NT( dstGrid ) = tempNT;
        DST_NB( dstGrid ) = tempNB;
        DST_ST( dstGrid ) = tempST;
        DST_SB( dstGrid ) = tempSB;
        DST_ET( dstGrid ) = tempET;
        DST_EB( dstGrid ) = tempEB;
        DST_WT( dstGrid ) = tempWT;
        DST_WB( dstGrid ) = tempWB;
    }
}

// fft-lbm-6-1
__global__ void mixed_fft_lbm_kernel_6_1(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        fft_lbm_fft0(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 0 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        fft_lbm_fft1(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 1 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        fft_lbm_fft2(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 2 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        fft_lbm_fft3(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 3 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        fft_lbm_fft4(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 4 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        fft_lbm_fft5(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 5 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 6, fft0_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        fft_lbm_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 768
        );
    }

}
