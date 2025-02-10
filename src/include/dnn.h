#pragma once
#include "header/pets_common.h"
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <sstream>
#include <mma.h>
#include <iostream>
#include <curand.h>
#include <unordered_set>
#include "header/tzgemm_header.h"
#include "json.h"
using namespace nvcuda;
#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

extern std::unordered_set<int> gemm_ks;

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

__inline__ __global__ void im2col_gpu_kernel(int n, float* data_im,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    float* data_col, int data_im_size, int data_col_size) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            int h_index = i / width_col;
            int h_col = h_index % height_col;
            int w_col = i % width_col;
            int c_im = h_index / height_col;
            int c_col = c_im * kernel_h * kernel_w;
            int h_offset = h_col * stride_h - pad_h;
            int w_offset = w_col * stride_w - pad_w;
            float* data_col_ptr = data_col;
            if (((c_col * height_col + h_col) * width_col + w_col) >= data_col_size) {
                // printf("c_col: %d, data_col_size: %d\n", c_col, data_col_size);
                continue;
            }
            data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
            float* data_im_ptr = data_im;
            if (((c_im * height + h_offset) * width + w_offset) >= data_im_size) {
                // printf("c_im: %d, data_im_size: %d\n", c_im, data_im_size);
                continue;
            }
            data_im_ptr += (c_im * height + h_offset) * width + w_offset;
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int h_im = h_offset + i * dilation_h;
                    int w_im = w_offset + j * dilation_w;
                    if (h_col >= height_col || w_col >= width_col || h_col < 0 || w_col < 0 || (data_col_ptr - data_col) >= data_col_size) {
                        // printf("h_col: %d, w_col: %d, height_col: %d, width_col: %d, data_col_ptr - data_col: %d\n", h_col, w_col, height_col, width_col, data_col_ptr - data_col);
                        continue;
                    }
                    if ((data_im_ptr - data_im) + i * dilation_h * width + j * dilation_w >= data_im_size) continue;
                    *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                        data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                    if (((data_col_ptr - data_col)+ height_col * width_col) >= data_col_size) {
                        // printf("data_col_ptr - data_col: %d, data_col_size: %d\n", data_col_ptr - data_col, data_col_size);
                        continue;
                    }
                    data_col_ptr += height_col * width_col;
                }
            }
        }
}

__inline__ __global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__inline__ __global__ void convertFp32ToInt8(int8_t *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}


// __inline__ __global__ void ptb_tzgemm(half *A, half *B, float *C, 
// 		// float alpha, float beta,
// 		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
// 		int grid_dimension_x, int block_dimension_x) {

// 	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
// 	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

// 	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
// 	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
// 	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

// 	float alpha = alpha_g;
// 	float beta = beta_g;

// 	unsigned int block_pos = blockIdx.x;
//     int thread_id_x = threadIdx.x;

// 	// Warp and lane identification.
// 	const unsigned int warpId = thread_id_x / WARP_SIZE;
// 	const unsigned int laneId = thread_id_x % WARP_SIZE;

// 	// Offset in shared memory from which the B matrix is stored.
// 	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
// 	// This pointer is used to access the C and D matrix tiles this warp computes.
// 	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
// 								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
// 								(warpId % 2) * SHMEM_OFFSET;

// 	// This pointer is used to stream the C and D matrices block-wide tile to and
// 	// from shared memory.
// 	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

// 	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
// 	// each tile computation. Technically this is not generally correct (may
// 	// result in a loss of precision). Zero still needs to be specially handled
// 	// though.
// 	beta /= alpha;

// 	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
// 	// matrix to the right and down, and selects the next tile to compute. Once
// 	// there's no such tile, all warps in this CTA exit.
// 	for (;; block_pos += gridDim.x) {
// 		if (block_pos >= grid_dimension_x) {
//             return;
//         }

// 		const unsigned int block_tile_i =
// 			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
// 		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
// 		// This warp's pointer to the C matrix data to copy memory from to shared
// 		// memory.
// 		const size_t gmem_idx =
// 			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


//         // These fragments will accumulate the result of A and B matrix fragment
//         // multiplications along the K_GLOBAL dimension.
//         wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
//         #pragma unroll
//         for (int i = 0; i < WARP_COL_TILES; i++) {
//             #pragma unroll
//             for (int j = 0; j < WARP_ROW_TILES; j++) {
//                 wmma::fill_fragment(c[i][j], 0.0f);
//             }
//         }

//         // Select what warp copies what matrix to shared memory.
//         // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
//         const half *warp_ptr = 
//             warpId < (WARPS_PER_BLOCK / 2) 
//                 ? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
//                 : (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

//         // Go through the global K dimension by a fixed step at a time.
//         #pragma unroll
//         for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
//             // Copy slices of the A and B matrices to shared memory.
//             // The first half of the warps in the CTA copy the A matrix, 
//             // the rest copy the B matrix.
//             size_t shmem_idx =
//                 warpId < (WARPS_PER_BLOCK / 2)
//                     ? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
//                     : (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

//             // First half of the warp copies the first row / column of the matrix,
//             // the second half of the warp copies the next.
//             int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
//                 + (laneId % CHUNK_COPY_LINE_LANES);

//             // Shift the second half of the warp to the next row / column in the
//             // shared memory.
//             shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

//             #pragma unroll
//             for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
//                 // Copy 16 bytes at once in each lane.
//                 *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
//                     *lane_ptr;

//                 // Advance the global memory pointer and the shared memory index.
//                 lane_ptr =
//                     (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
//                 shmem_idx += CHUNK_COPY_LINES_PER_WARP;
//             }

//             __syncthreads();

//             // Compute a grid of C matrix tiles in each warp.
//             #pragma unroll
//             for (int k_step = 0; k_step < CHUNK_K; k_step++) {
//                 wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
//                 wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

//                 #pragma unroll
//                 for (int i = 0; i < WARP_COL_TILES; i++) {
//                     size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
//                     const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
//                     wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

//                     #pragma unroll
//                     for (int j = 0; j < WARP_ROW_TILES; j++) {
//                         if (i == 0) {
//                             // Load the B matrix fragment once, because it is going to be
//                             // reused against the other A matrix fragments.
//                             size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
//                             const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
//                             wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
//                         }
//                         wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
//                     }
//                 }
//             }
//             __syncthreads();
//         }

//         // Store the D fragments to shared memory.
//         #pragma unroll
//         for (int i = 0; i < WARP_COL_TILES; i++) {
//             #pragma unroll
//             for (int j = 0; j < WARP_ROW_TILES; j++) {
//                 // Uniform, point-wise transformations of ALL fragment elements by ALL
//                 // threads in the warp are well-defined even though element indices
//                 // within fragment storage are not defined.
//                 #pragma unroll
//                 for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

//                 float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
//                 wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
//             }
//         }

//         __syncthreads();

//         // Now that shared memory contains all the D tiles, stream them to global
//         // memory.
//         float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

//         #pragma unroll
//         for (int i = 0; i < 16; i++) {
//             *((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
//                 *((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
//         }
//         __syncthreads();
//     }
// }

#ifdef AKER_INT8
__inline__ __global__ void ptb_tzgemm(int8_t *A, int8_t *B, int16_t *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x) {
    
	__shared__ int shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
#else
__inline__ __global__ void ptb_tzgemm(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x) {
    __shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
#endif
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
    #ifdef AKER_INT8
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0];
    #else
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;
    #endif
	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
    #ifdef AKER_INT8
	int16_t *shmem_warp_stream_ptr = (int16_t *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;
    #else
    float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;
    #endif

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += gridDim.x) {
		if (block_pos >= grid_dimension_x) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        #ifdef AKER_INT8
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c[WARP_COL_TILES][WARP_ROW_TILES];
        #else
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
        #endif

        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                #ifdef AKER_INT8
                wmma::fill_fragment(c[i][j], 0);
                #else
                wmma::fill_fragment(c[i][j], 0.0f);
                #endif
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        #ifdef AKER_INT8
        const int8_t *warp_ptr = (int8_t*)(
            warpId < (WARPS_PER_BLOCK / 2) 
                ? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2));
        #else
        const half *warp_ptr = 
            warpId < (WARPS_PER_BLOCK / 2) 
                ? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);
        #endif

        // Go through the global K dimension by a fixed step at a time.
        #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, 
            // the rest copy the B matrix.
            size_t shmem_idx =
                warpId < (WARPS_PER_BLOCK / 2)
                    ? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                    : (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            #ifdef AKER_INT8
            short4 *lane_ptr = (short4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
                + (laneId % CHUNK_COPY_LINE_LANES);
            #else
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
                + (laneId % CHUNK_COPY_LINE_LANES);
            #endif

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            // 这里有对齐问题
            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                #ifdef AKER_INT8
                // Copy 16 bytes at once in each lane.
                *((short4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                    *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr =
                    (short4 *)((int8_t *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
                #else
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                    *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr =
                    (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
                #endif
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                #ifdef AKER_INT8
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b[WARP_ROW_TILES];
                #else
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];
                #endif

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
                    #ifdef AKER_INT8
                    const int8_t *tile_ptr = (int8_t*)&shmem[shmem_idx_a][k_step * WMMA_K];
                    #else
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
                    #endif
                    wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
                            #ifdef AKER_INT8
                            const int8_t *tile_ptr = (int8_t*)&shmem[shmem_idx_b][k_step * WMMA_K];
                            #else
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
                            #endif
                            wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
                        }
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // Store the D fragments to shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                #pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

                // int *tile_ptr = (shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N);
                #ifdef AKER_INT8
                int *tile_ptr = (shmem_warp_tile_ptr);
                #else
                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
                #endif
                // wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
                // wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE / 2, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        #ifdef AKER_INT8
        int16_t *dst_gmem_warp_stream_ptr = &C[gmem_idx];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            *((short4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((short4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        #else
        float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            *((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        #endif
        __syncthreads();
    }
}
// #endif


extern long long MAX_ORI_WMMA_A;
extern long long MAX_ORI_WMMA_B;
extern long long MAX_ORI_WMMA_C;
extern int MAX_M_GLOBAL;
extern int MAX_N_GLOBAL;
extern int MAX_K_GLOBAL;
extern int MAX_COL_BUFFER;
extern int MAX_BOTTOM;
extern bool im2col_malloced;
extern bool gemm_malloced;
extern float *bottom;
extern float *col_buffer;
extern float *ori_host_A;
extern float *ori_host_B;
#ifdef AKER_INT8
extern int8_t *ori_wmma_A;
extern int8_t *ori_wmma_B;
extern int16_t *ori_wmma_C;
#else
extern half *ori_wmma_A;
extern half *ori_wmma_B;
extern float *ori_wmma_C;
#endif
extern float *cublas_wmma_C;
extern float *ori_wmma_results1;
extern float *ori_wmma_results2;

extern int batch_size;

extern std::string MODEL_NAME;

__inline__ cudnnStatus_t mycudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    // printf("batch_size: %d\n", batch_size);
    // cudaEvent_t startKERNEL, stopKERNEL;
	// cudaErrCheck(cudaEventCreate(&startKERNEL));
	// cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // float milliseconds = 0;

    // cudaErrCheck(cudaEventRecord(startKERNEL));

    // img2col参数
    int input_n;
	int input_c;
	int input_h;
	int input_w;
	int output_n;
	int output_c;
	int output_h;
	int output_w;

	int col_n;
	int col_c;
	int col_h;
	int col_w;

    int kernel_k;
    int kernel_c;
    int kernel_h;
	int kernel_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
    int dilation_h;
	int dilation_w;

    // get input tensor args from xDesc
    cudnnDataType_t dataType;

    int a,b,c,d;

    CUDNN_SAFE_CALL(cudnnGetTensor4dDescriptor(xDesc, &dataType, &input_n, &input_c, &input_h, &input_w, &a, &b, &c, &d));
    // printf("input_n: %d, input_c: %d, input_h: %d, input_w: %d\n", input_n, input_c, input_h, input_w);
    // get Filter args from wDesc
    cudnnTensorFormat_t format;
    CUDNN_SAFE_CALL(cudnnGetFilter4dDescriptor(wDesc, &dataType, &format, &kernel_k, &kernel_c, &kernel_h, &kernel_w));
    // printf("kernel_k: %d, kernel_c: %d, kernel_h: %d, kernel_w: %d\n", kernel_k, kernel_c, kernel_h, kernel_w);

    // get output tensor args from yDesc
    CUDNN_SAFE_CALL(cudnnGetTensor4dDescriptor(yDesc, &dataType, &output_n, &output_c, &output_h, &output_w, &a, &b, &c, &d));
    // printf("output_n: %d, output_c: %d, output_h: %d, output_w: %d\n", output_n, output_c, output_h, output_w);

    cudnnConvolutionMode_t mode;
    // get convolution args from convDesc
    CUDNN_SAFE_CALL(cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w, &mode, &dataType));
    // printf("pad_h: %d, pad_w: %d, stride_h: %d, stride_w: %d, dilation_h: %d, dilation_w: %d\n", pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    col_n = input_n;
    col_c = input_c;
    batch_size = input_n != 1 ? input_n : batch_size;
    // printf("--------------------------------mycudnnConvolutionForward-%d\n", batch_size);

    int height_col = (input_h + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (input_w + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = input_n * input_c * height_col * width_col;

    assert(input_n == output_n);
    assert(output_h == height_col);
    assert(output_w == width_col);
    assert(kernel_c == input_c);

    col_h = height_col;
    col_w = width_col;

    // printf("MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);
    MAX_COL_BUFFER = max(MAX_COL_BUFFER, col_n * col_c * col_h * col_w);
    MAX_BOTTOM = max(MAX_BOTTOM, input_n * input_c * input_h * input_w);
    // printf("[MY-CUDNN]MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);
    
    if (!im2col_malloced) {
        MAX_COL_BUFFER = geti(2, MODEL_NAME.c_str(), "MAX_COL_BUFFER");
        MAX_BOTTOM = geti(2, MODEL_NAME.c_str(), "MAX_BOTTOM");
        // printf("[MY-CUDNN]malloc MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);
        cudaErrCheck(cudaMalloc((void**)&bottom, MAX_BOTTOM * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&col_buffer, MAX_COL_BUFFER * sizeof(float)));
        // curandGenerator_t gen;
        // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        // curandErrCheck(curandGenerateUniform(gen, bottom, MAX_BOTTOM * sizeof(float)));
        // curandErrCheck(curandGenerateUniform(gen, col_buffer, MAX_COL_BUFFER * sizeof(float)));
        // cudaErrCheck(cudaMemset(bottom, 1.0f, input_n * input_c * input_h * input_w * sizeof(float))); // input_n
        // cudaErrCheck(cudaMemset(col_buffer, 1.0f, col_n * col_c * col_h * col_w * sizeof(float))); // col_n
        cudaErrCheck(cudaMemset(bottom, 1.0f, MAX_COL_BUFFER * sizeof(float)));
        cudaErrCheck(cudaMemset(col_buffer, 1.0f, MAX_COL_BUFFER * sizeof(float)));
        im2col_malloced = true;
    }
    // printf("MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);

    // curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    // curandErrCheck(curandGenerateUniform(gen, bottom, input_n * input_c * input_h * input_w));
    // curandErrCheck(curandGenerateUniform(gen, col_buffer, col_n * col_c * col_h * col_w));
    // cudaErrCheck(cudaMemset(bottom, 1.0f, MAX_COL_BUFFER * sizeof(float)));
    //     cudaErrCheck(cudaMemset(col_buffer, 1.0f, MAX_COL_BUFFER * sizeof(float)));

    // 调用 img2col
    dim3 im_grid;
	dim3 im_block;
    im_block.x = 256;
	im_grid.x = int(num_kernels / 256);
	im_grid.x = SM_NUM * geti(2, MODEL_NAME.c_str(), "MY_IM2COL_BLK_PER_SM");
    // printf("im_grid.x: %d\n", im_grid.x);

    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // print args
    // printf("num_kernels: %d, input_n: %d, input_c: %d, input_h: %d, input_w: %d, kernel_h: %d, kernel_w: %d, pad_h: %d, pad_w: %d, stride_h: %d, stride_w: %d, dilation_h: %d, dilation_w: %d, height_col: %d, width_col: %d, col_buffer: %p, bottom: %p\n", num_kernels, input_n, input_c, input_h, input_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, col_buffer, bottom);
    // printf("col_buffer_size: %d, bottom_size: %d\n", col_n * col_c * col_h * col_w, input_n * input_c * input_h * input_w);
    // launch im2col
    checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, input_n * input_c * input_h * input_w, col_n * col_c * col_h * col_w))); // input_n, col_n
    
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
	// cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	// cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
	// printf("im2col_gpu_kernel took %f ms\n", milliseconds);
    // milliseconds = 0;


    // call gemm
    int M_INPUT = input_n * height_col * width_col; // 1 * 112 * 112, no batch_size because not affect ptb block content
    int N_INPUT = kernel_k; // 64
    int K_INPUT = kernel_h * kernel_w * kernel_c;  // 7 * 7 * 3, right see https://zhuanlan.zhihu.com/p/276023990

    // printf("M_INPUT: %d, N_INPUT: %d, K_INPUT: %d\n", M_INPUT, N_INPUT, K_INPUT);

    // assert (M_INPUT * N_INPUT == output_n * output_c * output_h * output_w);


    int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

    gemm_ks.insert(K_GLOBAL);

    // M_INPUT *= batch_size;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = SM_NUM * geti(2, MODEL_NAME.c_str(), "MY_WMMA_BLK_PER_SM");
	wmma_block.x = THREADS_PER_BLOCK;

    // M_GLOBAL /= batch_size;
    long long cur_ori_wmma_A = (long long)sizeof(half) * M_GLOBAL * K_GLOBAL;
    long long cur_ori_wmma_B = (long long)sizeof(half) * N_GLOBAL * K_GLOBAL;
    long long cur_ori_wmma_C = (long long)sizeof(float) * M_GLOBAL * N_GLOBAL;
    MAX_ORI_WMMA_A = max(MAX_ORI_WMMA_A, cur_ori_wmma_A);
    MAX_ORI_WMMA_B = max(MAX_ORI_WMMA_B, cur_ori_wmma_B);
    MAX_ORI_WMMA_C = max(MAX_ORI_WMMA_C, cur_ori_wmma_C);
    // printf("[MY-CUDNN]MAX_ORI_WMMA_A: %lld, MAX_ORI_WMMA_B: %lld, MAX_ORI_WMMA_C: %lld\n", MAX_ORI_WMMA_A, MAX_ORI_WMMA_B, MAX_ORI_WMMA_C);
    if (!gemm_malloced) {
        MAX_ORI_WMMA_A = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_A");
        MAX_ORI_WMMA_B = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_B");
        MAX_ORI_WMMA_C = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_C");
        // printf("[mycudnn]try to malloc ori_wmma_A->%f MB, ori_wmma_B->%f MB, ori_wmma_C->%f MB\n", MAX_ORI_WMMA_A / 1024.0 / 1024.0, MAX_ORI_WMMA_B / 1024.0 / 1024.0, MAX_ORI_WMMA_C / 1024.0 / 1024.0);
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), MAX_ORI_WMMA_A * 2));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), MAX_ORI_WMMA_B * 2));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), MAX_ORI_WMMA_A));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), MAX_ORI_WMMA_B));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), MAX_ORI_WMMA_C));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cublas_wmma_C), MAX_ORI_WMMA_C));
        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, ori_host_A, MAX_ORI_WMMA_A / sizeof(half)));
        curandErrCheck(curandGenerateUniform(gen, ori_host_B, MAX_ORI_WMMA_B / sizeof(half)));
        #ifdef AKER_INT8
        convertFp32ToInt8 <<< (MAX_ORI_WMMA_A / sizeof(half) + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, MAX_ORI_WMMA_A / sizeof(half));
        convertFp32ToInt8 <<< (MAX_ORI_WMMA_B / sizeof(half) + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, MAX_ORI_WMMA_B / sizeof(half));
        #else
        convertFp32ToFp16 <<< (MAX_ORI_WMMA_A / sizeof(half) + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, MAX_ORI_WMMA_A / sizeof(half));
        convertFp32ToFp16 <<< (MAX_ORI_WMMA_B / sizeof(half) + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, MAX_ORI_WMMA_B / sizeof(half));
        #endif
        // cudaErrCheck(cudaMemset(ori_wmma_A, 2.0f, MAX_ORI_WMMA_A));
        // cudaErrCheck(cudaMemset(ori_wmma_B, 3.0f, MAX_ORI_WMMA_B));
        cudaErrCheck(cudaMemset(ori_wmma_C, 0, MAX_ORI_WMMA_C));
        cudaErrCheck(cudaMemset(cublas_wmma_C, 0, MAX_ORI_WMMA_C));

        ori_wmma_results1 = (float *)malloc(MAX_ORI_WMMA_C);
        for (int i = 0; i < MAX_ORI_WMMA_C / sizeof(float); i++) {
            ori_wmma_results1[i] = 0.0f;
        }
        ori_wmma_results2 = (float *)malloc(MAX_ORI_WMMA_C);
        for (int i = 0; i < MAX_ORI_WMMA_C / sizeof(float) ; i++) {
            ori_wmma_results2[i] = 0.0f;
        }
        gemm_malloced = true;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    // printf("M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("MAX_ORI_WMMA_A: %lld, MAX_ORI_WMMA_B: %lld, MAX_ORI_WMMA_C: %lld\n", MAX_ORI_WMMA_A, MAX_ORI_WMMA_B, MAX_ORI_WMMA_C);

    assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);


    // cudaErrCheck(cudaEventRecord(startKERNEL))
	// curandErrCheck(curandGenerateUniform(gen, ori_host_A, M_GLOBAL * K_GLOBAL));
    // curandErrCheck(curandGenerateUniform(gen, ori_host_B, N_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));
	// convertFp32ToFp16 <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    // convertFp32ToFp16 <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
    // cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // M_GLOBAL *= batch_size;
    // printf("Running with gemm...\n");
    // printf("gemm M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("gemm block dim: %d, grid dim: %d\n", wmma_block_dim_x, wmma_grid_dim_x);
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, 
		M_GLOBAL, N_GLOBAL, K_GLOBAL,
		// alpha, beta,
		wmma_grid_dim_x, wmma_block_dim_x)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("ptb_tzgemm took %f ms\n", milliseconds);
    // milliseconds = 0;

    // printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	// printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	// printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

    // printf("--------------------------------\n");
    return CUDNN_STATUS_SUCCESS;
}



__inline__ cublasStatus_t mycublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    int M_INPUT = m;
    int N_INPUT = n;
    int K_INPUT = k;

    int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

    gemm_ks.insert(K_GLOBAL);

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = SM_NUM * geti(2, MODEL_NAME.c_str(), "MY_WMMA_BLK_PER_SM");
	wmma_block.x = THREADS_PER_BLOCK;

    MAX_ORI_WMMA_A = max(MAX_ORI_WMMA_A, (long long)sizeof(half) * M_GLOBAL * K_GLOBAL);
    MAX_ORI_WMMA_B = max(MAX_ORI_WMMA_B, (long long)sizeof(half) * N_GLOBAL * K_GLOBAL);
    MAX_ORI_WMMA_C = max(MAX_ORI_WMMA_C, (long long)sizeof(float) * M_GLOBAL * N_GLOBAL);
    // printf("[MY-SGEMM]MAX_ORI_WMMA_A: %lld, MAX_ORI_WMMA_B: %lld, MAX_ORI_WMMA_C: %lld\n", MAX_ORI_WMMA_A, MAX_ORI_WMMA_B, MAX_ORI_WMMA_C);
    if (!gemm_malloced) {
        MAX_ORI_WMMA_A = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_A");
        MAX_ORI_WMMA_B = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_B");
        MAX_ORI_WMMA_C = geti(2, MODEL_NAME.c_str(), "MAX_ORI_WMMA_C");
        // printf("[mycublas]try to malloc ori_wmma_A->%f MB, ori_wmma_B->%f MB, ori_wmma_C->%f MB\n", MAX_ORI_WMMA_A / 1024.0 / 1024.0, MAX_ORI_WMMA_B / 1024.0 / 1024.0, MAX_ORI_WMMA_C / 1024.0 / 1024.0);
        // printf("MAX_ORI_WMMA_A: %lld, MAX_ORI_WMMA_B: %lld, MAX_ORI_WMMA_C: %lld\n", MAX_ORI_WMMA_A, MAX_ORI_WMMA_B, MAX_ORI_WMMA_C);
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), MAX_ORI_WMMA_A));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), MAX_ORI_WMMA_B));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), MAX_ORI_WMMA_C));
        cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, MAX_ORI_WMMA_A));
        cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, MAX_ORI_WMMA_B));
        cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, MAX_ORI_WMMA_C));
        gemm_malloced = true;
    }
    // printf("M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("MAX_ORI_WMMA_A: %lld, MAX_ORI_WMMA_B: %lld, MAX_ORI_WMMA_C: %lld\n", MAX_ORI_WMMA_A, MAX_ORI_WMMA_B, MAX_ORI_WMMA_C);
    assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);
    
    // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, sizeof(float) * M_GLOBAL * N_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_A, 1.0f, sizeof(half) * M_GLOBAL * K_GLOBAL));
    // cudaErrCheck(cudaMemset(ori_wmma_B, 1.0f, sizeof(half) * N_GLOBAL * K_GLOBAL));
	// convertFp32ToFp16 <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    // convertFp32ToFp16 <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
    // cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // cudaEvent_t startKERNEL, stopKERNEL;
	// cudaErrCheck(cudaEventCreate(&startKERNEL));
	// cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // float milliseconds = 0;
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C,
        M_GLOBAL, N_GLOBAL, K_GLOBAL,
        // alpha, beta,
        wmma_grid_dim_x, wmma_block_dim_x)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("ptb_tzgemm time:%f\n", milliseconds);
    return CUBLAS_STATUS_SUCCESS;
}