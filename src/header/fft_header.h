/*** 
 * @Author: diagonal
 * @Date: 2023-11-14 19:44:49
 * @LastEditors: diagonal
 * @LastEditTime: 2023-11-23 13:24:57
 * @FilePath: /tacker/ptb_kernels/header/fft_header.h
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */
#pragma once
// #define B 1024
#define FFT_B 10240

#define FFT_N 4*4*4*4
#define FFT_R 2

#define FFT_T  FFT_N/FFT_R 

inline __device__ float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline __device__ float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline __device__ float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline __device__ float2 operator*( float2 a, float b ) { return make_float2( b*a.x , b*a.y); }
// inline float2 operator|( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
// inline float2 operator&( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_float2(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_float2(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_float2( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_float2(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_float2(  0, -1 )
#define exp_3_8   make_float2( -1, -1 )//requires post-multiply by 1/sqrt(2)

#include "pets_common.h"
#define FFT_GRID_DIM (SM_NUM * 2)

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

inline __device__ int GPU_expand(int idxL, int N1, int N2 ){ 
	return (idxL/N1)*N1*N2 + (idxL%N1); 
}   

