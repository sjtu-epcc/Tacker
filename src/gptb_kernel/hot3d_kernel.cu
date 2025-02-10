#include "header/hot3d_header.h"

extern "C" __global__ void general_ptb_hot3d(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) 
{

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }
        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        float amb_temp = 80.0;

        int i = block_dimension_x * block_id_x + thread_id_x;  
        int j = block_dimension_y * block_id_y + thread_id_y;

        int c = i + j * nx;
        int xy = nx * ny;

        int W = (i == 0)        ? c : c - 1;
        int E = (i == nx-1)     ? c : c + 1;
        int N = (j == 0)        ? c : c - nx;
        int S = (j == ny-1)     ? c : c + nx;

        float temp1, temp2, temp3;
        temp1 = temp2 = tIn[c];
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;

        for (int k = 1; k < nz-1; ++k) {
            temp1 = temp2;
            temp2 = temp3;
            temp3 = tIn[c+xy];
            tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
                + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
            c += xy;
            W += xy;
            E += xy;
            N += xy;
            S += xy;
        }
        temp1 = temp2;
        temp2 = temp3;
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    }
}

extern "C" __device__ void internal_general_ptb_hot3d(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, 
        int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) 
{

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }
        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        float amb_temp = 80.0;

        int i = block_dimension_x * block_id_x + thread_id_x;  
        int j = block_dimension_y * block_id_y + thread_id_y;

        int c = i + j * nx;
        int xy = nx * ny;

        int W = (i == 0)        ? c : c - 1;
        int E = (i == nx-1)     ? c : c + 1;
        int N = (j == 0)        ? c : c - nx;
        int S = (j == ny-1)     ? c : c + nx;

        float temp1, temp2, temp3;
        temp1 = temp2 = tIn[c];
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;

        for (int k = 1; k < nz-1; ++k) {
            temp1 = temp2;
            temp2 = temp3;
            temp3 = tIn[c+xy];
            tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
                + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
            c += xy;
            W += xy;
            E += xy;
            N += xy;
            S += xy;
        }
        temp1 = temp2;
        temp2 = temp3;
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    }
}