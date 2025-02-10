#include "header/atom.h"
#include <mma.h>
using namespace nvcuda;

#include "header/cutcp_header.h"
#include "header/sgemm_header.h"

__device__ void cutcp_sgemm_cutcp0(
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


__device__ void cutcp_sgemm_sgemm0(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
	        int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		        int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {
    
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    // // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = (threadIdx.x - thread_base); // TODO: check
        int m = block_id_x * TILE_M + mid;
        int n = block_id_y * TILE_N + thread_id_x;
        

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            // __syncthreads();
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
        }
        int t = ldc * block_id_y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

// cutcp-sgemm-1-1
__global__ void mixed_cutcp_sgemm_kernel_1_1(int cutcp0_binDim_x, int cutcp0_binDim_y, float4* cutcp0_binZeroAddr, float cutcp0_h, float cutcp0_cutoff2, float cutcp0_inv_cutoff2, float* cutcp0_regionZeroAddr, int cutcp0_zRegionIndex_t, int cutcp0_grid_dimension_x, int cutcp0_grid_dimension_y, int cutcp0_grid_dimension_z, int cutcp0_block_dimension_x, int cutcp0_block_dimension_y, int cutcp0_block_dimension_z, int cutcp0_ptb_start_block_pos, int cutcp0_ptb_iter_block_step, int cutcp0_ptb_end_block_pos, float* sgemm1_A, float* sgemm1_B, float* sgemm1_C, int sgemm1_NORMAL_M, int sgemm1_NORMAL_N, int sgemm1_NORMAL_K, int sgemm1_grid_dimension_x, int sgemm1_grid_dimension_y, int sgemm1_grid_dimension_z, int sgemm1_block_dimension_x, int sgemm1_block_dimension_y, int sgemm1_block_dimension_z, int sgemm1_ptb_start_block_pos, int sgemm1_ptb_iter_block_step, int sgemm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        cutcp_sgemm_cutcp0(
            cutcp0_binDim_x, cutcp0_binDim_y, cutcp0_binZeroAddr, cutcp0_h, cutcp0_cutoff2, cutcp0_inv_cutoff2, cutcp0_regionZeroAddr, cutcp0_zRegionIndex_t, cutcp0_grid_dimension_x, cutcp0_grid_dimension_y, cutcp0_grid_dimension_z, cutcp0_block_dimension_x, cutcp0_block_dimension_y, cutcp0_block_dimension_z, cutcp0_ptb_start_block_pos + 0 * cutcp0_ptb_iter_block_step, cutcp0_ptb_iter_block_step * 1, cutcp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        cutcp_sgemm_sgemm0(
            sgemm1_A, sgemm1_B, sgemm1_C, sgemm1_NORMAL_M, sgemm1_NORMAL_N, sgemm1_NORMAL_K, sgemm1_grid_dimension_x, sgemm1_grid_dimension_y, sgemm1_grid_dimension_z, sgemm1_block_dimension_x, sgemm1_block_dimension_y, sgemm1_block_dimension_z, sgemm1_ptb_start_block_pos + 0 * sgemm1_ptb_iter_block_step, sgemm1_ptb_iter_block_step * 1, sgemm1_ptb_end_block_pos, 128
        );
    }

}
