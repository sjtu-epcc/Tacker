#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
using namespace nvcuda; 
#include "header/mriq_header.h"
#include "header/fft_header.h"

// fft
__device__ void G_GPU_exchange_fft_mriq_fft0( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_mriq_fft0(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_mriq_fft0( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_mriq_fft0(float2* data, 
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
		G_GPU_DoFft_fft_mriq_fft0( v, thread_id_x, 1);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

__device__ void G_GPU_exchange_fft_mriq_fft1( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_mriq_fft1(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_mriq_fft1( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_mriq_fft1(float2* data, 
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
		G_GPU_DoFft_fft_mriq_fft1( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}
// step_size == launch param == ptb worker num == SM_NUM * ptb_per_sm_number

__device__ void G_GPU_exchange_fft_mriq_fft2( float2* v, int stride, int idxD, int incD, 
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

__device__ void G_GPU_DoFft_fft_mriq_fft2(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		G_GPU_exchange_fft_mriq_fft2( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

__device__ void fft_mriq_fft2(float2* data, 
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
		G_GPU_DoFft_fft_mriq_fft2( v, thread_id_x);  
		for (int r=0; r<FFT_R; r++) {
			ori_data[idxG + r*FFT_T] = v[r];
		}
	}
}

// mriq
__device__ void fft_mriq_mriq0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2_MRIQ * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2_MRIQ * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void fft_mriq_mriq1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2_MRIQ * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2_MRIQ * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

// fft-mriq-3-2
__global__ void mixed_fft_mriq_kernel_3_2(float2* fft0_data, int fft0_grid_dimension_x, int fft0_grid_dimension_y, int fft0_grid_dimension_z, int fft0_block_dimension_x, int fft0_block_dimension_y, int fft0_block_dimension_z, int fft0_ptb_start_block_pos, int fft0_ptb_iter_block_step, int fft0_ptb_end_block_pos, int mriq1_numK, int mriq1_kGlobalIndex, float* mriq1_x, float* mriq1_y, float* mriq1_z, float* mriq1_Qr, float* mriq1_Qi, int mriq1_grid_dimension_x, int mriq1_grid_dimension_y, int mriq1_grid_dimension_z, int mriq1_block_dimension_x, int mriq1_block_dimension_y, int mriq1_block_dimension_z, int mriq1_ptb_start_block_pos, int mriq1_ptb_iter_block_step, int mriq1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        fft_mriq_fft0(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 0 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        fft_mriq_fft1(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 1 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        fft_mriq_fft2(
            fft0_data, fft0_grid_dimension_x, fft0_grid_dimension_y, fft0_grid_dimension_z, fft0_block_dimension_x, fft0_block_dimension_y, fft0_block_dimension_z, fft0_ptb_start_block_pos + 2 * fft0_ptb_iter_block_step, fft0_ptb_iter_block_step * 3, fft0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 640) {
        fft_mriq_mriq0(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 0 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 2, mriq1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 896) {
        fft_mriq_mriq1(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 1 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 2, mriq1_ptb_end_block_pos, 640
        );
    }

}
