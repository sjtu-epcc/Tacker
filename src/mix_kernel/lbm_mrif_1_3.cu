#include "header/lbm_header.h"
#include "header/mrif_header.h"
// lbm
__device__ void lbm_mrif_lbm0( float* srcGrid, float* dstGrid,
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

// mrif
__device__ void lbm_mrif_mrif0(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
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

__device__ void lbm_mrif_mrif1(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
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

__device__ void lbm_mrif_mrif2(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, 
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

// lbm-mrif-1-3
__global__ void mixed_lbm_mrif_kernel_1_3(float* lbm0_srcGrid, float* lbm0_dstGrid, int lbm0_grid_dimension_x, int lbm0_grid_dimension_y, int lbm0_grid_dimension_z, int lbm0_block_dimension_x, int lbm0_block_dimension_y, int lbm0_block_dimension_z, int lbm0_ptb_start_block_pos, int lbm0_ptb_iter_block_step, int lbm0_ptb_end_block_pos, int mrif1_numK, int mrif1_kGlobalIndex, float* mrif1_x, float* mrif1_y, float* mrif1_z, float* mrif1_outR, float* mrif1_outI, int mrif1_grid_dimension_x, int mrif1_grid_dimension_y, int mrif1_grid_dimension_z, int mrif1_block_dimension_x, int mrif1_block_dimension_y, int mrif1_block_dimension_z, int mrif1_ptb_start_block_pos, int mrif1_ptb_iter_block_step, int mrif1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        lbm_mrif_lbm0(
            lbm0_srcGrid, lbm0_dstGrid, lbm0_grid_dimension_x, lbm0_grid_dimension_y, lbm0_grid_dimension_z, lbm0_block_dimension_x, lbm0_block_dimension_y, lbm0_block_dimension_z, lbm0_ptb_start_block_pos + 0 * lbm0_ptb_iter_block_step, lbm0_ptb_iter_block_step * 1, lbm0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 384) {
        lbm_mrif_mrif0(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 0 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 640) {
        lbm_mrif_mrif1(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 1 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 896) {
        lbm_mrif_mrif2(
            mrif1_numK, mrif1_kGlobalIndex, mrif1_x, mrif1_y, mrif1_z, mrif1_outR, mrif1_outI, mrif1_grid_dimension_x, mrif1_grid_dimension_y, mrif1_grid_dimension_z, mrif1_block_dimension_x, mrif1_block_dimension_y, mrif1_block_dimension_z, mrif1_ptb_start_block_pos + 2 * mrif1_ptb_iter_block_step, mrif1_ptb_iter_block_step * 3, mrif1_ptb_end_block_pos, 640
        );
    }

}
