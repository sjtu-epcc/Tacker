'''
Author: diagonal
Date: 2023-11-18 22:08:26
LastEditors: diagonal
LastEditTime: 2023-11-19 12:59:29
FilePath: /tacker/mix_kernels/code/mrif_code.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
mrif_header_code = """
#include "header/mrif_header.h"
#include "kernel/mrif_kernel.cu"
"""

mrif_variables_code = """
    // mrif variables
    // ---------------------------------------------------------------------------------------
        int mrif_blks = 3;
        int mrif_iter = 1;
        int mrif_numX, mrif_numK;		                /* Number of X and K values */
        int original_numK;		            /* Number of K values in input file */
        float *mrif_base_kx, *mrif_base_ky, *mrif_base_kz;		        /* K trajectory (3D vectors) */
        float *mrif_base_x, *mrif_base_y, *mrif_base_z;		            /* X coordinates (3D vectors) */
        float *mrif_base_phiR, *mrif_base_phiI;		            /* Phi values (complex) */
        float *mrif_base_dR, *mrif_base_dI;		                /* D values (complex) */
        float *mrif_base_realRhoPhi, *mrif_base_imagRhoPhi;     /* RhoPhi values (complex) */
        mrif_kValues* mrif_kVals;		                /* Copy of X and RhoPhi.  Its
                                            * data layout has better cache
                                            * performance. */
        inputData(
            &original_numK, &mrif_numX,
            &mrif_base_kx, &mrif_base_ky, &mrif_base_kz,
            &mrif_base_x, &mrif_base_y, &mrif_base_z,
            &mrif_base_phiR, &mrif_base_phiI,
            &mrif_base_dR, &mrif_base_dI);
        mrif_numK = original_numK;

        // createDataStructs(mrif_numK, mrif_numX, mrif_base_realRhoPhi, mrif_base_imagRhoPhi, base_outR, base_outI);
        mrif_kVals = (mrif_kValues *)calloc(mrif_numK, sizeof (mrif_kValues));
        mrif_base_realRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));
        mrif_base_imagRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));

        // kernel 1
        float *ori_phiR, *ori_phiI;
        float *ori_dR, *ori_dI;
        float *ori_realRhoPhi, *ori_imagRhoPhi;
        // kernel 2
        float *ori_x, *ori_y, *ori_z;
        float *ori_outI, *ori_outR;
        float *host_ori_outI;		            /* Output signal (complex) */

        // kernel 2
        float *ptb_x, *ptb_y, *ptb_z;
        float *ptb_outI, *ptb_outR;
        float *host_ptb_outI;		            /* Output signal (complex) */

        // gptb kernel
        float *gptb_x, *gptb_y, *gptb_z;
        float *gptb_outI, *gptb_outR;
        float *host_gptb_outI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&ori_phiR, mrif_numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, mrif_numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, mrif_numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, mrif_base_phiR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, mrif_base_phiI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, mrif_base_dR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, mrif_base_dI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&ori_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ori_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&ptb_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&gptb_x, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_y, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_z, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(gptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&gptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outI, 0, mrif_numX * sizeof(float)));

        host_ori_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_ptb_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_gptb_outI = (float*) calloc (mrif_numX, sizeof (float));
    // ---------------------------------------------------------------------------------------
"""

mrif_solo_running_code = """
    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
        // computeRhoPhi_GPU(mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
        dim3 mrif_grid1;
        dim3 mrif_block1;
        mrif_grid1.x = mrif_numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_grid1.y = 1;
        mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_block1.y = 1;
        printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \\n", 
                    mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrif_grid1, mrif_block1 >>> (mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
        cudaErrCheck(cudaMemcpy(mrif_base_realRhoPhi, ori_realRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(mrif_base_imagRhoPhi, ori_imagRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < mrif_numK; k++) {
            mrif_kVals[k].Kx = mrif_base_kx[k];
            mrif_kVals[k].Ky = mrif_base_ky[k];
            mrif_kVals[k].Kz = mrif_base_kz[k];
            mrif_kVals[k].RhoPhiR = mrif_base_realRhoPhi[k];
            mrif_kVals[k].RhoPhiI = mrif_base_imagRhoPhi[k];
        }
    // ---------------------------------------------------------------------------------------
    
    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 mrif_grid2, ori_mrif_grid2;
        dim3 mrif_block2, ori_mrif_block2;
        mrif_grid2.x = mrif_numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrif_grid2.y = 1;
        mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrif_block2.y = 1;
        ori_mrif_grid2 = mrif_grid2;
        ori_mrif_block2 = mrif_block2;
        printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \\n", 
                    mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        mrif_kValues* mrif_kValsTile = mrif_kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, mrif_numK - FHGridBase);
        cudaMemcpyToSymbol(c, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mrif <<< mrif_grid2, mrif_block2 >>> (
                mrif_numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] mrif took %f ms\\n\\n", kernel_time);
    // ---------------------------------------------------------------------------------------
"""

mrif_ptb_running_code = """
    // PTB running
    // ---------------------------------------------------------------------------------------
        int mrif_grid2_dim_x = mrif_grid2.x;
        int mrif_block2_dim_x = mrif_block2.x;
        mrif_grid2.x = mrif_blks == 0 ? mrif_grid2_dim_x : SM_NUM * mrif_blks;
        printf("[PTB] Running with mrif...\\n");
        printf("[PTB] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \\n", 
            mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_mrif <<< mrif_grid2, mrif_block2 >>> (mrif_numK, FHGridBase, ptb_x, ptb_y, ptb_z, ptb_outR, ptb_outI, 
                                    mrif_grid2_dim_x, mrif_block2_dim_x,
                                    mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] mrif took %f ms\\n\\n", kernel_time);
    // ---------------------------------------------------------------------------------------
"""

mrif_verify_code = """
    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_ori_outI, ori_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_gptb_outI, gptb_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        errors = 0;
        for (int i = 0; i < mrif_numX; i++) {
            float v1 = host_ori_outI[i];
            float v2 = host_gptb_outI[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 5) printf("%f %f\\n", v1, v2);
            }
            if (i < 3) printf("%d %f %f\\n", i, v1, v2);
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\\n");
        }
    // ---------------------------------------------------------------------------------------
"""

mrif_gptb_variables_code = """
"""

mrif_gptb_params_list = """mrif_numK, FHGridBase, gptb_x, gptb_y, gptb_z, gptb_outR, gptb_outI, 
            ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mrif_grid2.x * ori_mrif_grid2.y * ori_mrif_grid2.z"""

mrif_gptb_params_list_new = """mrif_numK, FHGridBase, gptb_x, gptb_y, gptb_z, gptb_outR, gptb_outI, 
    ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
    start_blk_no, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, end_blk_no, 0"""


def get_mrif_code_before_mix_kernel():
    return mrif_variables_code + mrif_solo_running_code + mrif_ptb_running_code + mrif_gptb_variables_code

def get_mrif_code_after_mix_kernel():
    return mrif_verify_code

def get_mrif_header_code():
    return mrif_header_code