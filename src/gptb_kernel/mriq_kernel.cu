#include "header/mriq_header.h"

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
            //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = PIx2_MRIQ * (ck[kIndex].Kx * sX + ck[kIndex].Ky * sY +
                            ck[kIndex].Kz * sZ);
                sQr += ck[kIndex].PhiMag * cos(expArg);
                sQi += ck[kIndex].PhiMag * sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = PIx2_MRIQ * (ck[kIndex1].Kx * sX + ck[kIndex1].Ky * sY +
                            ck[kIndex1].Kz * sZ);
                sQr += ck[kIndex1].PhiMag * cos(expArg1);
                sQi += ck[kIndex1].PhiMag * sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}


extern "C" __global__ void g_general_ptb_mriq_int(int numK, int kGlobalIndex, int* x, int* y, int* z, int* Qr , int* Qi,
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

            int sX;
            int sY;
            int sZ;
            int sQr;
            int sQi;

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
            //     float expArg = PIx2_MRIQ * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
            //     sQr += ck[0].PhiMag * cos(expArg);
            //     sQi += ck[0].PhiMag * sin(expArg);
            //     kIndex++;
            //     kGlobalIndex++;
            // }

            for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                kIndex += 2, kGlobalIndex += 2) {
                float expArg = (int)PIx2_MRIQ * (ck_int[kIndex].Kx * sX + ck_int[kIndex].Ky * sY +
                            ck_int[kIndex].Kz * sZ);
                sQr += ck_int[kIndex].PhiMag * (int)cos(expArg);
                sQi += ck_int[kIndex].PhiMag * (int)sin(expArg);

                int kIndex1 = kIndex + 1;
                float expArg1 = (int)PIx2_MRIQ * (ck_int[kIndex1].Kx * sX + ck_int[kIndex1].Ky * sY +
                            ck_int[kIndex1].Kz * sZ);
                sQr += ck_int[kIndex1].PhiMag * (int)cos(expArg1);
                sQi += ck_int[kIndex1].PhiMag * (int)sin(expArg1);
            }

            Qr[xIndex] = sQr;
            Qi[xIndex] = sQi;
        }
    }
}