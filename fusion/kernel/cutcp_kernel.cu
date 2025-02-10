
__global__ void ori_cutcp(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex_t,
    int iteration
    )
{
	__shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
    // __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

	const int xRegionIndex = blockIdx.x;
	const int yRegionIndex = blockIdx.y;
    const int zRegionIndex = blockIdx.z;
	/* thread id */
	const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

    for (int loop = 0; loop < iteration; loop++) {
        /* neighbor index */
        int nbrid;

        /* this is the start of the sub-region indexed by tid */
        myRegionAddr = regionZeroAddr + ((zRegionIndex*gridDim.y + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;

        /* spatial coordinate of this lattice point */
        float x = (8 * xRegionIndex + threadIdx.x) * h;
        float y = (8 * yRegionIndex + threadIdx.y) * h;
        float z = (8 * zRegionIndex + threadIdx.z) * h;

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
            __syncthreads();

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
            __syncthreads();
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}


__global__ void ptb_cutcp(
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
    int iteration
    ) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x / (block_dimension_y * block_dimension_z);
    int thread_id_y = (threadIdx.x % (block_dimension_y * block_dimension_z)) / block_dimension_z;
    int thread_id_z = (threadIdx.x % (block_dimension_y * block_dimension_z)) % block_dimension_z;
    // 这里等于 threadIdx.x % block_dimension_z

    /* thread id */
	const int tid = (thread_id_z * block_dimension_y + thread_id_y) * block_dimension_x + thread_id_x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos % (grid_dimension_y * grid_dimension_z)) / grid_dimension_z;
        int block_id_z = (block_pos % (grid_dimension_y * grid_dimension_z)) % grid_dimension_z;

        for (int loop = 0; loop < iteration; loop++) {

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
               __syncthreads();

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
               __syncthreads();
            } /* end loop over neighbor list */

            /* store into global memory */
            myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
        }
    }
}


__global__ void ptb2_cutcp(
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
    int iteration
    ) {
    unsigned int block_pos = blockIdx.x;
    /* thread id */
	const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos % (grid_dimension_y * grid_dimension_z)) / grid_dimension_z;
        int block_id_z = (block_pos % (grid_dimension_y * grid_dimension_z)) % grid_dimension_z;

        for (int loop = 0; loop < iteration; loop++) {

            int xRegionIndex = block_id_x;
            int yRegionIndex = block_id_y;
            int zRegionIndex = block_id_z;
        
            /* neighbor index */
            int nbrid;

            /* this is the start of the sub-region indexed by tid */
            myRegionAddr = regionZeroAddr + ((zRegionIndex*grid_dimension_y + yRegionIndex)*grid_dimension_x + xRegionIndex)*REGION_SIZE;

            /* spatial coordinate of this lattice point */
            float x = (8 * xRegionIndex + threadIdx.x) * h;
            float y = (8 * yRegionIndex + threadIdx.y) * h;
            float z = (8 * zRegionIndex + threadIdx.z) * h;

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
               __syncthreads();

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
               __syncthreads();
            } /* end loop over neighbor list */

            /* store into global memory */
            myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
        }
    }
}


__device__ void mix_cutcp0(
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
    int thread_step,
    int iteration
    ) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) / (block_dimension_y * block_dimension_z);
    int thread_id_y = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) / block_dimension_z;
    int thread_id_z = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) % block_dimension_z;

    /* thread id */
	const int tid = (thread_id_z * block_dimension_y + thread_id_y) * block_dimension_x + thread_id_x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += CUTCP_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos % (grid_dimension_y * grid_dimension_z)) / grid_dimension_z;
        int block_id_z = (block_pos % (grid_dimension_y * grid_dimension_z)) % grid_dimension_z;

        for (int loop = 0; loop < iteration; loop++) {

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
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

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
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
            } /* end loop over neighbor list */

            /* store into global memory */
            myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
        }
    }
}


__device__ void mix_cutcp1(
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
    int thread_step,
    int iteration
    ) {
    unsigned int block_pos = blockIdx.x + 68 * 2;

    int thread_id_x = (threadIdx.x - thread_step) / (block_dimension_y * block_dimension_z);
    int thread_id_y = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) / block_dimension_z;
    int thread_id_z = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) % block_dimension_z;

    /* thread id */
	const int tid = (thread_id_z * block_dimension_y + thread_id_y) * block_dimension_x + thread_id_x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += SM_NUM * 4) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos % (grid_dimension_y * grid_dimension_z)) / grid_dimension_z;
        int block_id_z = (block_pos % (grid_dimension_y * grid_dimension_z)) % grid_dimension_z;

        for (int loop = 0; loop < iteration; loop++) {

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
                asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");

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
                asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");
            } /* end loop over neighbor list */

            /* store into global memory */
            myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
        }
    }
}


__device__ void mix_cutcp2(
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
    int thread_step,
    int iteration
    ) {
    unsigned int block_pos = blockIdx.x + SM_NUM * 2;
    int thread_id_x = (threadIdx.x - thread_step) / (block_dimension_y * block_dimension_z);
    int thread_id_y = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) / block_dimension_z;
    int thread_id_z = ((threadIdx.x - thread_step) % (block_dimension_y * block_dimension_z)) % block_dimension_z;

    /* thread id */
	const int tid = (thread_id_z * block_dimension_y + thread_id_y) * block_dimension_x + thread_id_x;

    __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
	// __shared__ float *myRegionAddr;
	// __shared__ int3 myBinIndex;

    float *myRegionAddr;
	int3 myBinIndex;

    for (;; block_pos += SM_NUM * 3) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }

        int block_id_x = block_pos / (grid_dimension_y * grid_dimension_z);
        int block_id_y = (block_pos % (grid_dimension_y * grid_dimension_z)) / grid_dimension_z;
        int block_id_z = (block_pos % (grid_dimension_y * grid_dimension_z)) % grid_dimension_z;

        for (int loop = 0; loop < iteration; loop++) {

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
                asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");

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
                asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
            } /* end loop over neighbor list */

            /* store into global memory */
            myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
            myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
        }
    }
}


__device__ void general_ptb_cutcp0(
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

__device__ void general_ptb_cutcp1(
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
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void general_ptb_cutcp2(
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
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void general_ptb_cutcp3(
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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(4), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void general_ptb_cutcp4(
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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(5), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void general_ptb_cutcp5(
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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");

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
asm volatile("bar.sync %0, %1;" : : "r"(6), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

__device__ void general_ptb_cutcp6(
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
            asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");

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
            asm volatile("bar.sync %0, %1;" : : "r"(7), "r"(128) : "memory");
        } /* end loop over neighbor list */

        /* store into global memory */
        myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
        myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
    }
}

extern "C" __global__ void g_general_ptb_cutcp(
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