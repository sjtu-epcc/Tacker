#include "header/mrif_header.h"

// extern __constant__ __device__ mrif_kValues c[KERNEL_FH_K_ELEMS_PER_GRID];


extern "C" __global__ void g_general_ptb_mrif(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
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


extern "C" __global__ void g_general_ptb_mrif_int(int numK, int kGlobalIndex, int* x, int* y, int* z, int* outR, int* outI, 
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
            int sX;
            int sY;
            int sZ;
            int sOutR;
            int sOutI;

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
                    float expArg = (int)PIx2 * (c_int[kIndex].Kx * sX + c_int[kIndex].Ky * sY + c_int[kIndex].Kz * sZ);
                    int cosArg = cos((expArg));
                    int sinArg = sin((expArg));
                    sOutR += c_int[kIndex].RhoPhiR * cosArg - c_int[kIndex].RhoPhiI * sinArg;
                    sOutI += c_int[kIndex].RhoPhiI * cosArg + c_int[kIndex].RhoPhiR * sinArg;
                }
            }

            for (; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
                    kIndex += 4, kGlobalIndex += 4) {
                float expArg = (int)PIx2 * (c_int[kIndex].Kx * sX + c_int[kIndex].Ky * sY + c_int[kIndex].Kz * sZ);
                int cosArg = cos(expArg);
                int sinArg = sin(expArg);
                sOutR += c_int[kIndex].RhoPhiR * cosArg - c_int[kIndex].RhoPhiI * sinArg;
                sOutI += c_int[kIndex].RhoPhiI * cosArg + c_int[kIndex].RhoPhiR * sinArg;

                int kIndex1 = kIndex + 1;
                float expArg1 = (int)PIx2 * (c_int[kIndex1].Kx * sX + c_int[kIndex1].Ky * sY + c_int[kIndex1].Kz * sZ);
                int cosArg1 = cos(expArg1);
                int sinArg1 = sin(expArg1);
                sOutR += c_int[kIndex1].RhoPhiR * cosArg1 - c_int[kIndex1].RhoPhiI * sinArg1;
                sOutI += c_int[kIndex1].RhoPhiI * cosArg1 + c_int[kIndex1].RhoPhiR * sinArg1;

                int kIndex2 = kIndex + 2;
                float expArg2 = (int)PIx2 * (c_int[kIndex2].Kx * sX + c_int[kIndex2].Ky * sY + c_int[kIndex2].Kz * sZ);
                int cosArg2 = cos(expArg2);
                int sinArg2 = sin(expArg2);
                sOutR += c_int[kIndex2].RhoPhiR * cosArg2 - c_int[kIndex2].RhoPhiI * sinArg2;
                sOutI += c_int[kIndex2].RhoPhiI * cosArg2 + c_int[kIndex2].RhoPhiR * sinArg2;

                int kIndex3 = kIndex + 3;
                float expArg3 = (int)PIx2 * (c_int[kIndex3].Kx * sX + c_int[kIndex3].Ky * sY + c_int[kIndex3].Kz * sZ);
                int cosArg3 = cos(expArg3);
                int sinArg3 = sin(expArg3);
                sOutR += c_int[kIndex3].RhoPhiR * cosArg3 - c_int[kIndex3].RhoPhiI * sinArg3;
                sOutI += c_int[kIndex3].RhoPhiI * cosArg3 + c_int[kIndex3].RhoPhiR * sinArg3;    
            }

            outR[xIndex] = sOutR;
            outI[xIndex] = sOutI;
        }
	}
}