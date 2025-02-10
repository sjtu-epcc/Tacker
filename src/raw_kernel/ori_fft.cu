#include "header/fft_header.h"

inline __device__ void GPU_FFT2(float2* v){
	float2 vt = v[0];
	v[0] = vt + v[1];
	v[1] = vt - v[1];
}

inline __device__ void GPU_FFT2(float2 &v1, float2 &v2) { 
    float2 v0 = v1;  
    v1 = v0 + v2; 
    v2 = v0 - v2; 
}

inline __device__ void GPU_FFT4(float2 &v0,float2 &v1,float2 &v2,float2 &v3) { 
    GPU_FFT2(v0, v2);
    GPU_FFT2(v1, v3);
    v3 = v3 * exp_1_4;
    GPU_FFT2(v0, v1);
    GPU_FFT2(v2, v3);    
}

inline __device__ void GPU_FFT4(float2* v) {
    GPU_FFT4(v[0],v[1],v[2],v[3] );
}

__device__ int GPU_expand(int idxL, int N1, int N2 ){ 
	return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

__device__ void GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS){ 
	__shared__ float work[FFT_T*FFT_R*2];//FFT_T*FFT_R*2
	float* sr = work;
	float* si = work+FFT_T*FFT_R;  
	__syncthreads(); 
	// asm volatile("bar.sync %0, %1;" : : "r"(sync_id), "r"(128) : "memory");
	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	__syncthreads(); 
	// asm volatile("bar.sync %0, %1;" : : "r"(sync_id), "r"(128) : "memory");

	for( int r=0; r<FFT_R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}      

__device__ void GPU_DoFft(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<FFT_N; Ns*=FFT_R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*FFT_R); 
		for( int r=0; r<FFT_R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,FFT_R); 
		int idxS = GPU_expand(j,FFT_N/FFT_R,FFT_R); 
		GPU_exchange( v,stride, idxD,Ns, idxS,FFT_N/FFT_R);
	}      
}

extern "C" __global__ void ori_fft(float2* data) {
	float2 *ori_data = data + blockIdx.x*FFT_N;
		float2 v[FFT_R];
		data = ori_data;

		int idxG = threadIdx.x; 
		for (int r=0; r<FFT_R; r++) {  
			v[r] = data[idxG + r*FFT_T];
		} 
		GPU_DoFft( v, threadIdx.x );  
		for (int r=0; r<FFT_R; r++) {
			data[idxG + r*FFT_T] = v[r];
		} 
}