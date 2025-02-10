/*** 
 * @Author: diagonal
 * @Date: 2023-11-14 19:44:50
 * @LastEditors: diagonal
 * @LastEditTime: 2023-11-19 12:57:35
 * @FilePath: /tacker/ptb_kernels/kernel/mriq_kernel.cu
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */

void inputData(int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI) {
    int numK, numX;
    FILE* fid = fopen("/home/jxdeng/workspace/tacker/0_mybench/file_t/mriq_input.bin", "r");
    if (fid == NULL) {
        fprintf(stderr, "Cannot open input file\n");
        exit(-1);
    }

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
    fclose (fid); 
}

__global__ void ori_ComputePhiMag(float* phiR, float* phiI, float* phiMag, int numK) {
    int indexK = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}


__global__ void ori_mriq(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi, int iteration) {
    for (int loop = 0; loop < iteration; loop++) {
        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = blockIdx.x * KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}


__global__ void ptb_mriq(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi,
         int grid_dimension_x, int block_dimension_x, int iteration) {
    int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;
        for (int loop = 0; loop < iteration; loop++) {
            for (int QGrid = 0; QGrid < 1; QGrid++) {
                kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

                float sX;
                float sY;
                float sZ;
                float sQr;
                float sQi;

                // Determine the element of the X arrays computed by this thread
                int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

                // Read block's X values from global mem to shared mem
                sX = x[xIndex];
                sY = y[xIndex];
                sZ = z[xIndex];
                sQr = Qr[xIndex];
                sQi = Qi[xIndex];

                // Loop over all elements of K in constant mem to compute a partial value
                // for X.
                int kIndex = 0;
                // if (numK % 2) {
                //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
                //     sQr += ck[0].PhiMag * cos(expArg);
                //     sQi += ck[0].PhiMag * sin(expArg);
                //     kIndex++;
                //     kGlobalIndex++;
                // }

                for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 2, kGlobalIndex += 2) {
                    float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                                ck[kIndex].Kz * sZ);
                    sQr += ck[kIndex].PhiMag * cos(expArg);
                    sQi += ck[kIndex].PhiMag * sin(expArg);

                    int kIndex1 = kIndex + 1;
                    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                                ck[kIndex1].Kz * sZ);
                    sQr += ck[kIndex1].PhiMag * cos(expArg1);
                    sQi += ck[kIndex1].PhiMag * sin(expArg1);
                }

                Qr[xIndex] = sQr;
                Qi[xIndex] = sQi;
            }
        }
    }
}


__global__ void ptb2_mriq(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi,
         int grid_dimension_x, int iteration) {
    int block_pos = blockIdx.x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;
        for (int loop = 0; loop < iteration; loop++) {
            for (int QGrid = 0; QGrid < 1; QGrid++) {
                kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

                float sX;
                float sY;
                float sZ;
                float sQr;
                float sQi;

                // Determine the element of the X arrays computed by this thread
                int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

                // Read block's X values from global mem to shared mem
                sX = x[xIndex];
                sY = y[xIndex];
                sZ = z[xIndex];
                sQr = Qr[xIndex];
                sQi = Qi[xIndex];

                // Loop over all elements of K in constant mem to compute a partial value
                // for X.
                int kIndex = 0;
                // if (numK % 2) {
                //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
                //     sQr += ck[0].PhiMag * cos(expArg);
                //     sQi += ck[0].PhiMag * sin(expArg);
                //     kIndex++;
                //     kGlobalIndex++;
                // }

                for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 2, kGlobalIndex += 2) {
                    float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                                ck[kIndex].Kz * sZ);
                    sQr += ck[kIndex].PhiMag * cos(expArg);
                    sQi += ck[kIndex].PhiMag * sin(expArg);

                    int kIndex1 = kIndex + 1;
                    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                                ck[kIndex1].Kz * sZ);
                    sQr += ck[kIndex1].PhiMag * cos(expArg1);
                    sQi += ck[kIndex1].PhiMag * sin(expArg1);
                }

                Qr[xIndex] = sQr;
                Qi[xIndex] = sQi;
            }
        }
    }
}

__device__ void mix_mriq(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi,
         int grid_dimension_x, int block_dimension_x, int thread_step, int iteration) {
    int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x - thread_step;

    for (;; block_pos += MRIQ_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;
        for (int loop = 0; loop < iteration; loop++) {
            for (int QGrid = 0; QGrid < 1; QGrid++) {
                kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

                float sX;
                float sY;
                float sZ;
                float sQr;
                float sQi;

                // Determine the element of the X arrays computed by this thread
                int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

                // Read block's X values from global mem to shared mem
                sX = x[xIndex];
                sY = y[xIndex];
                sZ = z[xIndex];
                sQr = Qr[xIndex];
                sQi = Qi[xIndex];

                // Loop over all elements of K in constant mem to compute a partial value
                // for X.
                int kIndex = 0;
                // if (numK % 2) {
                //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
                //     sQr += ck[0].PhiMag * cos(expArg);
                //     sQi += ck[0].PhiMag * sin(expArg);
                //     kIndex++;
                //     kGlobalIndex++;
                // }

                for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 2, kGlobalIndex += 2) {
                    float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                                ck[kIndex].Kz * sZ);
                    sQr += ck[kIndex].PhiMag * cos(expArg);
                    sQi += ck[kIndex].PhiMag * sin(expArg);

                    int kIndex1 = kIndex + 1;
                    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                                ck[kIndex1].Kz * sZ);
                    sQr += ck[kIndex1].PhiMag * cos(expArg1);
                    sQi += ck[kIndex1].PhiMag * sin(expArg1);
                }

                Qr[xIndex] = sQr;
                Qi[xIndex] = sQi;
            }
        }
    }
}


__device__ void general_ptb_mriq0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq2(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq3(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq4(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq5(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq6(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq7(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

__device__ void general_ptb_mriq8(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}

extern "C" __global__ void g_general_ptb_mriq(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

        for (int QGrid = 0; QGrid < 1; QGrid++) {
            kGlobalIndex = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

            float sX;
            float sY;
            float sZ;
            float sQr;
            float sQi;

            // Determine the element of the X arrays computed by this thread
            int xIndex = block_id_x * KERNEL_Q_THREADS_PER_BLOCK + thread_id_x;

            // Read block's X values from global mem to shared mem
            sX = x[xIndex];
            sY = y[xIndex];
            sZ = z[xIndex];
            sQr = Qr[xIndex];
            sQi = Qi[xIndex];

            // Loop over all elements of K in constant mem to compute a partial value
            // for X.
            int kIndex = 0;
            // if (numK % 2) {
            //     float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2 * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2 * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}