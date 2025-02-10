#include "header/cutcp_header.h"
#include "header/tzgemm_header.h"
#include <mma.h>
using namespace nvcuda; 
__device__ void cutcp_tzgemm_cutcp0(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex_t,
    int grid_dimension_x,
    int grid_dimension_y,
    int grid_dimension_z,
    int block_dimension_x,
    int block_dimension_y,
    int block_dimension_z,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos,
    int thread_base
    ) {
    // unsigned int block_pos = blockIdx.x + SM_NUM * 2; // TODO: why SM_NUM * 2?
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) / (block_dimension_y * block_dimension_z);
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_z) % block_dimension_y;
    int thread_id_z = (threadIdx.x - thread_base) % block_dimension_z;

    // // ori
    // int thread_id_x = (threadIdx.x - thread_step) / (block_dimension_y * block_dimension_z);
    // int thread_id_y = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) / block_dimension_z;
    // int thread_id_z = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) % block_dimension_z;

    /* thread id */
	const int tid = (thread_id_z * block_dimension_y + thread_id_y) * block_dimension_x + thread_id_x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos / grid_dimension_z) % grid_dimension_y;
        int block_id_z = block_pos % grid_dimension_z;


        int xRegionIndex = block_id_x;
        int yRegionIndex = block_id_y;
        int zRegionIndex = block_id_z;
    
        /* neighbor index */
        int nbrid;

        /* this is the start of the sub-region indexed by tid */
        myRegionAddr = regionZeroAddr + ((zRegionIndex*grid_dimension_y + yRegionIndex)*grid_dimension_x + xRegionIndex)*REGION_SIZE;

        /* spatial coordinate of this lattice point */
        float x = (8 * xRegionIndex + thread_id_x) * h;
        float y = (8 * yRegionIndex + thread_id_y) * h;
        float z = (8 * zRegionIndex + thread_id_z) * h;

        int totalbins = 0;
        int numbins;

        /* bin number determined by center of region */
        myBinIndex.x = (int) floorf((8 * xRegionIndex + 4) * h * BIN_INVLEN);
        myBinIndex.y = (int) floorf((8 * yRegionIndex + 4) * h * BIN_INVLEN);
        myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

        /* first neighbor in list for me to cache */
        nbrid = (tid >> 4);

        numbins = BIN_CACHE_MAXLEN;

        float energy0 = 0.f;
        float energy1 = 0.f;
        float energy2 = 0.f;
        float energy3 = 0.f;


        for (totalbins = 0;  totalbins < NbrListLen;  totalbins += numbins) {
            int bincnt;

            /* start of where to write in shared memory */
            int startoff = BIN_SIZE * (tid >> 4);

            /* each half-warp to cache up to 4 atom bins */
            for (bincnt = 0;  bincnt < 4 && nbrid < NbrListLen;  bincnt++, nbrid += 8) {
                int i = myBinIndex.x + NbrList[nbrid].x;
                int j = myBinIndex.y + NbrList[nbrid].y;
                int k = myBinIndex.z + NbrList[nbrid].z;

                /* determine global memory location of atom bin */
                float *p_global = ((float *) binZeroAddr) + (((__mul24(k, binDim_y) + j)*binDim_x + i) << BIN_SHIFT);

                /* coalesced read from global memory -
                * retain same ordering in shared memory for now */
                int binIndex = startoff + (bincnt << (3 + BIN_SHIFT));
                int tidmask = tid & 15;

                AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
                AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
            }
        //    __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");

            /* no warp divergence */
            if (totalbins + BIN_CACHE_MAXLEN > NbrListLen) {
                numbins = NbrListLen - totalbins;
            }

            int stopbin = (numbins << BIN_SHIFT);
            for (bincnt = 0; bincnt < stopbin; bincnt+=BIN_SIZE) {
                for (int i = 0;  i < BIN_DEPTH;  i++) {
                    int off = bincnt + (i<<2);

                    float aq = AtomBinCache[off + 3];
                    if (0.f == aq) 
                        break;  /* no more atoms in bin */

                    float dx = AtomBinCache[off    ] - x;
                    float dz = AtomBinCache[off + 2] - z;
                    float dxdz2 = dx*dx + dz*dz;
                    float dy = AtomBinCache[off + 1] - y;
                    float r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy0 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy1 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy2 += aq * rsqrtf(r2) * s * s;
                    }
                    dy -= 2.0f*h;
                    r2 = dy*dy + dxdz2;

                    if (r2 < cutoff2) {
                        float s = (1.f - r2 * inv_cutoff2);
                        energy3 += aq * rsqrtf(r2) * s * s;
                    }
                } /* end loop over atoms in bin */
            } /* end loop over cached atom bins */
        //    __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void cutcp_tzgemm_tzgemm0(half *A, half *B, float *C, 
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

				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

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
				asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
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

			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");;
		}
}

__global__ void cutcp_tzgemm_mix(
        int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, 
        int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, 
            half *tzgemm1_A, half *tzgemm1_B, float *tzgemm1_C, int tzgemm1_NORMAL_M, int tzgemm1_NORMAL_N, int tzgemm1_NORMAL_K,
            int tzgemm1_grid_dimension_x, int tzgemm1_grid_dimension_y, int tzgemm1_grid_dimension_z, int tzgemm1_block_dimension_x, int tzgemm1_block_dimension_y, int tzgemm1_block_dimension_z, int tzgemm1_ptb_start_block_pos, int tzgemm1_ptb_iter_block_step, int tzgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        cutcp_tzgemm_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 1, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        cutcp_tzgemm_tzgemm0(
            tzgemm1_A, tzgemm1_B, tzgemm1_C, tzgemm1_NORMAL_M, tzgemm1_NORMAL_N, tzgemm1_NORMAL_K, tzgemm1_grid_dimension_x, tzgemm1_grid_dimension_y, tzgemm1_grid_dimension_z, tzgemm1_block_dimension_x, tzgemm1_block_dimension_y, tzgemm1_block_dimension_z, tzgemm1_ptb_start_block_pos + 0 * tzgemm1_ptb_iter_block_step, tzgemm1_ptb_iter_block_step * 1, tzgemm1_ptb_end_block_pos, 128
        );
    }

}
