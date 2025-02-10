#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
using namespace nvcuda; 
#include "header/lbm_header.h"
#include "header/mriq_header.h"
// lbm
__device__ void lbm_mriq_lbm0( float* srcGrid, float* dstGrid,
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

// mriq
__device__ void lbm_mriq_mriq0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

__device__ void lbm_mriq_mriq1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi,
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

// lbm-mriq-1-2
__global__ void mixed_lbm_mriq_kernel_1_2(float* lbm0_srcGrid, float* lbm0_dstGrid, int lbm0_grid_dimension_x, int lbm0_grid_dimension_y, int lbm0_grid_dimension_z, int lbm0_block_dimension_x, int lbm0_block_dimension_y, int lbm0_block_dimension_z, int lbm0_ptb_start_block_pos, int lbm0_ptb_iter_block_step, int lbm0_ptb_end_block_pos, int mriq1_numK, int mriq1_kGlobalIndex, float* mriq1_x, float* mriq1_y, float* mriq1_z, float* mriq1_Qr, float* mriq1_Qi, int mriq1_grid_dimension_x, int mriq1_grid_dimension_y, int mriq1_grid_dimension_z, int mriq1_block_dimension_x, int mriq1_block_dimension_y, int mriq1_block_dimension_z, int mriq1_ptb_start_block_pos, int mriq1_ptb_iter_block_step, int mriq1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        lbm_mriq_lbm0(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 0 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 1, lbm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        lbm_mriq_mriq0(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 0 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 2, mriq1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 640) {
        lbm_mriq_mriq1(
            mriq1_numK, mriq1_kGlobalIndex, mriq1_x, mriq1_y, mriq1_z, mriq1_Qr, mriq1_Qi, mriq1_grid_dimension_x, mriq1_grid_dimension_y, mriq1_grid_dimension_z, mriq1_block_dimension_x, mriq1_block_dimension_y, mriq1_block_dimension_z, mriq1_ptb_start_block_pos + 1 * mriq1_ptb_iter_block_step, mriq1_ptb_iter_block_step * 2, mriq1_ptb_end_block_pos, 384
        );
    }

}
