
void createDataStructs(int numK, int numX, 
                       float*& realRhoPhi, float*& imagRhoPhi, 
                       float*& outR, float*& outI) {
  realRhoPhi = (float* ) calloc(numK, sizeof(float));
  imagRhoPhi = (float* ) calloc(numK, sizeof(float));
  outR = (float*) calloc (numX, sizeof (float));
  outI = (float*) calloc (numX, sizeof (float));
}


void inputData(int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI,
               float** dR, float** dI) {
    int numK, numX;
    FILE* fid = fopen("/home/jxdeng/workspace/tacker/0_mybench/file_t/mrif_input.bin", "r");
    fread (&numK, sizeof (int), 1, fid);
    *_numK = numK;
    fread (&numX, sizeof (int), 1, fid);
    *_numX = numX;
    *kx = (float *) memalign(16, numK * sizeof (float));
    fread (*kx, sizeof (float), numK, fid);
    *ky = (float *) memalign(16, numK * sizeof (float));
    fread (*ky, sizeof (float), numK, fid);
    *kz = (float *) memalign(16, numK * sizeof (float));
    fread (*kz, sizeof (float), numK, fid);
    *x = (float *) memalign(16, numX * sizeof (float));
    fread (*x, sizeof (float), numX, fid);
    *y = (float *) memalign(16, numX * sizeof (float));
    fread (*y, sizeof (float), numX, fid);
    *z = (float *) memalign(16, numX * sizeof (float));
    fread (*z, sizeof (float), numX, fid);
    *phiR = (float *) memalign(16, numK * sizeof (float));
    fread (*phiR, sizeof (float), numK, fid);
    *phiI = (float *) memalign(16, numK * sizeof (float));
    fread (*phiI, sizeof (float), numK, fid);
    *dR = (float *) memalign(16, numK * sizeof (float));
    fread (*dR, sizeof (float), numK, fid);
    *dI = (float *) memalign(16, numK * sizeof (float));
    fread (*dI, sizeof (float), numK, fid);
    fclose (fid); 
}


__global__ void ComputeRhoPhiGPU(int numK,
        float* phiR, float* phiI, 
        float* dR, float* dI, 
        float* realRhoPhi, float* imagRhoPhi) {
    int indexK = blockIdx.x*KERNEL_RHO_PHI_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float rPhiR = phiR[indexK];
        float rPhiI = phiI[indexK];
        float rDR = dR[indexK];
        float rDI = dI[indexK];
        realRhoPhi[indexK] = rPhiR * rDR + rPhiI * rDI;
        imagRhoPhi[indexK] = rPhiR * rDI - rPhiI * rDR;
    }
}


__global__ void ori_mrif(int numK, int kGlobalIndex,
        float* x, float* y, float* z, 
        float* outR, float* outI,
        int iteration) {
    for (int loop = 0; loop < iteration; loop++) {
        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {
            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;
            // Determine the element of the X arrays computed by this thread
            int xIndex = blockIdx.x * KERNEL_FH_THREADS_PER_BLOCK + threadIdx.x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                        kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 *
                    (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (;(kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 *
                    (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 *
                    (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 *
                    (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 *
                    (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
    }
}


__global__ void ptb_mrif(int numK, int kGlobalIndex,
        float* x, float* y, float* z, 
        float* outR, float* outI,
        int grid_dimension_x, int block_dimension_x,
        int iteration) {

    int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
        int block_id_x = block_pos;

        for (int loop = 0; loop < iteration; loop++) {
            for (int FHGrid = 0; FHGrid < 1; FHGrid++) {
                kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;

                float sX;
                float sY;
                float sZ;
                float sOutR;
                float sOutI;
                // Determine the element of the X arrays computed by this thread
                int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

                sX = x[xIndex];
                sY = y[xIndex];
                sZ = z[xIndex];
                sOutR = outR[xIndex];
                sOutI = outI[xIndex];

                // Loop over all elements of K in constant mem to compute a partial value
                // for X.
                int kIndex = 0;
                int kCnt = numK - kGlobalIndex;
                if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                    for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                            kIndex++, kGlobalIndex++) {
                        float expArg = PIx2 *
                        (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                        float cosArg = cos(expArg);
                        float sinArg = sin(expArg);
                        sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                        sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                    }
                }

                for (;(kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                        kIndex += 4, kGlobalIndex += 4) {
                    float expArg = PIx2 *
                        (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                    int kIndex1 = kIndex + 1;
                    float expArg1 = PIx2 *
                        (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                    float cosArg1 = cos(expArg1);
                    float sinArg1 = sin(expArg1);
                    sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                    sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                    int kIndex2 = kIndex + 2;
                    float expArg2 = PIx2 *
                        (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                    float cosArg2 = cos(expArg2);
                    float sinArg2 = sin(expArg2);
                    sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                    sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                    int kIndex3 = kIndex + 3;
                    float expArg3 = PIx2 *
                        (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                    float cosArg3 = cos(expArg3);
                    float sinArg3 = sin(expArg3);
                    sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                    sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
                }

                outR[xIndex] = sOutR;
                outI[xIndex] = sOutI;
            }
        }
    }
}


__device__ void mix_mrif(int numK, int kGlobalIndex,
              float* x, float* y, float* z, 
              float* outR, float* outI, 
			  int grid_dimension_x, int block_dimension_x, int thread_step,
			  int iteration) {

	unsigned int block_pos = blockIdx.x;
	int thread_id_x = threadIdx.x - thread_step;

	 for (;; block_pos += MRIF_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;
	
		for (int loop = 0; loop < iteration; loop++) {

			for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

				kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
				float sX;
				float sY;
				float sZ;
				float sOutR;
				float sOutI;

				// Determine the element of the X arrays computed by this thread
				int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

				sX = x[xIndex];
				sY = y[xIndex];
				sZ = z[xIndex];
				sOutR = outR[xIndex];
				sOutI = outI[xIndex];

				// Loop over all elements of K in constant mem to compute a partial value
				// for X.
				int kIndex = 0;
				int kCnt = numK - kGlobalIndex;
				if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
					for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
						kIndex++, kGlobalIndex++) {
						float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
						float cosArg = cos(expArg);
						float sinArg = sin(expArg);
						sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
						sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
					}
				}

				for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
						kIndex += 4, kGlobalIndex += 4) {
					float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
					float cosArg = cos(expArg);
					float sinArg = sin(expArg);
					sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
					sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

					int kIndex1 = kIndex + 1;
					float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
					float cosArg1 = cos(expArg1);
					float sinArg1 = sin(expArg1);
					sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
					sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

					int kIndex2 = kIndex + 2;
					float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
					float cosArg2 = cos(expArg2);
					float sinArg2 = sin(expArg2);
					sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
					sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

					int kIndex3 = kIndex + 3;
					float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
					float cosArg3 = cos(expArg3);
					float sinArg3 = sin(expArg3);
					sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
					sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
				}

				outR[xIndex] = sOutR;
				outI[xIndex] = sOutI;
			}
		}
	}
}

__device__ void general_ptb_mrif0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif2(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif3(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif4(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif5(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif6(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif7(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__device__ void general_ptb_mrif8(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}

__global__ void g_general_ptb_mrif(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
	
	int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

	for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);
	

        for (int FHGrid = 0; FHGrid < 1; FHGrid++) {

            kGlobalIndex = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
            float sX;
            float sY;
            float sZ;
            float sOutR;
            float sOutI;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_FH_THREADS_PER_BLOCK + thread_id_x;

            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sOutR = outR[xIndex];
            sOutI = outI[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            int kCnt = numK - kGlobalIndex;
            if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
                for (kIndex = 0; (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
                    kIndex++, kGlobalIndex++) {
                    float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                    float cosArg = cos(expArg);
                    float sinArg = sin(expArg);
                    sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                    sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = PIx2 * (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
                float cosArg = cos(expArg);
                float sinArg = sin(expArg);
                sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
                sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
                float cosArg1 = cos(expArg1);
                float sinArg1 = sin(expArg1);
                sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
                sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = PIx2 * (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
                float cosArg2 = cos(expArg2);
                float sinArg2 = sin(expArg2);
                sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
                sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = PIx2 * (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
                float cosArg3 = cos(expArg3);
                float sinArg3 = sin(expArg3);
                sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
                sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}