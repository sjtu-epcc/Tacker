#include "header/tzgemm_header.h"
#include "header/stencil_header.h"


__device__ void stencil_tzgemm_tzgemm0(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

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
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					wmma::fill_fragment(c[i][j], 0.0f);
				}
			}

			// Select what warp copies what matrix to shared memory.
			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
			const half *warp_ptr = 
				warpId < (WARPS_PER_BLOCK / 2) 
					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

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
				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
					+ (laneId % CHUNK_COPY_LINE_LANES);

				// Shift the second half of the warp to the next row / column in the
				// shared memory.
				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

				#pragma unroll
				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
					// Copy 16 bytes at once in each lane.
					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

					// Advance the global memory pointer and the shared memory index.
					lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
				}

				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;

				// Compute a grid of C matrix tiles in each warp.
				#pragma unroll
				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

					#pragma unroll
					for (int i = 0; i < WARP_COL_TILES; i++) {
						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

						#pragma unroll
						for (int j = 0; j < WARP_ROW_TILES; j++) {
							if (i == 0) {
								// Load the B matrix fragment once, because it is going to be
								// reused against the other A matrix fragments.
								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
							}
							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
						}
					}
				}
				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
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

					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
				}
			}

			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
		}
}

__device__ void stencil_tzgemm_stencil0(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
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
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
        
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
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
            bottom=sh_A0[sh_id];
            sh_A0[sh_id]=top;
            bottom2=sh_A0[sh_id2];
            sh_A0[sh_id2]=top2;
            // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");;
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
        }
    }
}

// asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

__global__ void stencil_tzgemm_mix(
        float stencil0_c0, float stencil0_c1, float* stencil0_A0, float* stencil0_Anext, int stencil0_nx, int stencil0_ny, int stencil0_nz, int stencil0_grid_dimension_x, int stencil0_grid_dimension_y, int stencil0_grid_dimension_z, int stencil0_block_dimension_x, int stencil0_block_dimension_y, int stencil0_block_dimension_z, int stencil0_ptb_start_block_pos, int stencil0_ptb_iter_block_step, int stencil0_ptb_end_block_pos, 
         half* tzgemm1_A, half* tzgemm1_B, float* tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K, int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        stencil_tzgemm_tzgemm0(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, 
            tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K,
            tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z,  
            tzgemm1_ptb_start_block_pos, tzgemm1_ptb_iter_block_step, tzgemm1_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        stencil_tzgemm_stencil0(
            stencil0_c0, stencil0_c1, stencil0_A0, stencil0_Anext, stencil0_nx, stencil0_ny, stencil0_nz, 
            stencil0_grid_dimension_x, stencil0_grid_dimension_y, stencil0_grid_dimension_z, stencil0_block_dimension_x, stencil0_block_dimension_y, stencil0_block_dimension_z, 
			stencil0_ptb_start_block_pos, stencil0_ptb_iter_block_step, stencil0_ptb_end_block_pos, 128
        );
    }
}