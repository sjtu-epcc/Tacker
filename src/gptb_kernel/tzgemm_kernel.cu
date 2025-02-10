#include "tzgemm_kernel.h"

// extern "C" __global__ void general_ptb_tzgemm(half *A, half *B, float *C, 
// 		// float alpha, float beta,
// 		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
// 		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
// 		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
// 		internal_general_ptb_tzgemm(
// 			A, B, C, 
// 			// alpha, beta,
// 			M_GLOBAL, N_GLOBAL, K_GLOBAL,
// 			grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z,  
// 					ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base
// 		);
// 	}
// __forceinline__ __device__ void internal_general_ptb_tzgemm(half *A, half *B, float *C, 
// 		// float alpha, float beta,
// 		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
// 		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
// 		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

// 	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
// 	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

// 	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
// 	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
// 	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

// 	float alpha = alpha_g;
// 	float beta = beta_g;

// 	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

//     int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;

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
// 	for (;; block_pos += ptb_iter_block_step) {
// 		if (block_pos >= ptb_end_block_pos) {
//             return;
//         }

// 		const unsigned int block_tile_i =
// 			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
// 		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
// 		// This warp's pointer to the C matrix data to copy memory from to shared
// 		// memory.
// 		const size_t gmem_idx =
// 			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


// 			// These fragments will accumulate the result of A and B matrix fragment
// 			// multiplications along the K_GLOBAL dimension.
// 			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
// 			#pragma unroll
// 			for (int i = 0; i < WARP_COL_TILES; i++) {
// 				#pragma unroll
// 				for (int j = 0; j < WARP_ROW_TILES; j++) {
// 					wmma::fill_fragment(c[i][j], 0.0f);
// 				}
// 			}

// 			// Select what warp copies what matrix to shared memory.
// 			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
// 			const half *warp_ptr = 
// 				warpId < (WARPS_PER_BLOCK / 2) 
// 					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
// 					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

// 			// Go through the global K dimension by a fixed step at a time.
// 			#pragma unroll
// 			for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
// 				// Copy slices of the A and B matrices to shared memory.
// 				// The first half of the warps in the CTA copy the A matrix, 
// 				// the rest copy the B matrix.
// 				size_t shmem_idx =
// 					warpId < (WARPS_PER_BLOCK / 2)
// 						? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
// 						: (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

// 				// First half of the warp copies the first row / column of the matrix,
// 				// the second half of the warp copies the next.
// 				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
// 					+ (laneId % CHUNK_COPY_LINE_LANES);

// 				// Shift the second half of the warp to the next row / column in the
// 				// shared memory.
// 				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

// 				#pragma unroll
// 				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
// 					// Copy 16 bytes at once in each lane.
// 					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
// 						*lane_ptr;

// 					// Advance the global memory pointer and the shared memory index.
// 					lane_ptr =
// 						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
// 					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
// 				}

// 				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

// 				// Compute a grid of C matrix tiles in each warp.
// 				#pragma unroll
// 				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
// 					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
// 					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

// 					#pragma unroll
// 					for (int i = 0; i < WARP_COL_TILES; i++) {
// 						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
// 						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
// 						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

// 						#pragma unroll
// 						for (int j = 0; j < WARP_ROW_TILES; j++) {
// 							if (i == 0) {
// 								// Load the B matrix fragment once, because it is going to be
// 								// reused against the other A matrix fragments.
// 								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
// 								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
// 								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
// 							}
// 							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
// 						}
// 					}
// 				}
// 				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
// 			}

// 			// Store the D fragments to shared memory.
// 			#pragma unroll
// 			for (int i = 0; i < WARP_COL_TILES; i++) {
// 				#pragma unroll
// 				for (int j = 0; j < WARP_ROW_TILES; j++) {
// 					// Uniform, point-wise transformations of ALL fragment elements by ALL
// 					// threads in the warp are well-defined even though element indices
// 					// within fragment storage are not defined.
// 					#pragma unroll
// 					for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

// 					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
// 					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
// 				}
// 			}

// 			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

// 			// Now that shared memory contains all the D tiles, stream them to global
// 			// memory.
// 			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

// 			#pragma unroll
// 			for (int i = 0; i < 16; i++) {
// 				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
// 					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
// 			}
// 			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
// 		}
// }

#ifdef AKER_INT8
__forceinline__ __device__ void internal_general_ptb_tzgemm(int8_t *A, int8_t *B, int16_t *C, 
#else
__forceinline__ __device__ void internal_general_ptb_tzgemm(half *A, half *B, float *C, 
#endif
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
	#ifdef AKER_INT8
	__shared__ int shmem_[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	#else
	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	#endif
	// // extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base);

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
    #ifdef AKER_INT8
    int *shmem_warp_tile_ptr = (int *)&shmem_[0][0];
    #else
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;
    #endif
	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
    #ifdef AKER_INT8
	int16_t *shmem_warp_stream_ptr = (int16_t *)&shmem_[0][0] + warpId * SHMEM_STRIDE * WMMA_M;
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
	for (;; block_pos += ptb_iter_block_step) {
		if (block_pos >= ptb_end_block_pos) {
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
                *((short4 *)&shmem_[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
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

            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

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
                    const int8_t *tile_ptr = (int8_t*)&shmem_[shmem_idx_a][k_step * WMMA_K];
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
                            const int8_t *tile_ptr = (int8_t*)&shmem_[shmem_idx_b][k_step * WMMA_K];
                            #else
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
                            #endif
                            wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
                        }
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
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

        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

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
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        #endif
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
    }
}

extern "C" __global__ void general_ptb_tzgemm(
#ifdef AKER_INT8
	int8_t *A, int8_t *B, int16_t *C,
#else
	half *A, half *B, float *C, 
#endif
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		// int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10,
		// int a11, int a12, int a13, int a14, int a15, int a16, int a17, int a18, int a19, int a20,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
		internal_general_ptb_tzgemm(
			A, B, C, 
			// alpha, beta,
			M_GLOBAL, N_GLOBAL, K_GLOBAL,
			grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z,  
					ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base
		);
	}