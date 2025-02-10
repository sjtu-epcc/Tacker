#include "header/lbm_header.h"
#include "header/tzgemm_header.h"
#include <mma.h>
using namespace nvcuda; 

__device__ void lbm_tzgemm_lbm0( float* srcGrid, float* dstGrid,
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

__device__ void lbm_tzgemm_tzgemm0(half *A, half *B, float *C, 
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

__global__ void lbm_tzgemm_mix(float* lbm0_srcGrid, float* lbm0_dstGrid, 
        int lbm0_grid_dimension_x, int lbm0_grid_dimension_y, int lbm0_grid_dimension_z, int lbm0_block_dimension_x, int lbm0_block_dimension_y, int lbm0_block_dimension_z, int lbm0_ptb_start_block_pos, int lbm0_ptb_iter_block_step, int lbm0_ptb_end_block_pos, 
        half* tzgemm1_A, half* tzgemm1_B, float* tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K, 
        int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        lbm_tzgemm_lbm0(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 0 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 1, lbm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        lbm_tzgemm_tzgemm0(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 0 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 1, tzgemm1_ptb_end_block_pos, 128
        );
    }
}