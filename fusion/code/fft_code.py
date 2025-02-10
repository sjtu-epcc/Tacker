'''
Author: diagonal
Date: 2023-11-18 14:21:31
LastEditors: diagonal
LastEditTime: 2023-11-19 12:48:32
FilePath: /tacker/mix_kernels/code/fft_code.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
'''
Author: diagonal
Date: 2023-11-18 14:21:31
LastEditors: diagonal
LastEditTime: 2023-11-19 12:48:09
FilePath: /tacker/mix_kernels/code/fft_code.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
solo_ptb_fft_blks = 3

fft_header_code = """
#include "header/fft_header.h"
#include "kernel/fft_kernel.cu"
"""

fft_variables_code = """
    // fft variables
    // ---------------------------------------------------------------------------------------
		//8*1024*1024;
        int fft_blks = 3;
	    int fft_iter = 1;
		int n_bytes = FFT_N * FFT_B * sizeof(float2);
		int nthreads = FFT_T;
		srand(54321);

		float *host_shared_source =(float *)malloc(n_bytes);  
		float2 *source    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ori_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ptb_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_gptb_result	= (float2 *)malloc( n_bytes );

		for(int b=0; b<FFT_B;b++) {	
			for( int i = 0; i < FFT_N; i++ ) {
				source[b*FFT_N+i].x = (rand()/(float)RAND_MAX)*2-1;
				source[b*FFT_N+i].y = (rand()/(float)RAND_MAX)*2-1;
			}
		}

		// allocate device memory
		float2 *fft_ori_source;
		float *fft_ori_shared_source;
		cudaMalloc((void**) &fft_ori_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ori_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

		float2 *fft_ptb_source;
		float *fft_ptb_shared_source;
		cudaMalloc((void**) &fft_ptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_source, source, n_bytes, cudaMemcpyHostToDevice);

		// gptb
		float2 *fft_gptb_source;
		float *fft_gptb_shared_source;
		cudaMalloc((void**) &fft_gptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_gptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_source, source, n_bytes, cudaMemcpyHostToDevice);
    // ---------------------------------------------------------------------------------------
"""

fft_solo_running_code = """
	// SOLO running
    // ---------------------------------------------------------------------------------------
		dim3 fft_grid, ori_fft_grid;
		dim3 fft_block, ori_fft_block;
		fft_grid.x = FFT_B;
		fft_block.x = nthreads;
		ori_fft_grid = fft_grid;
		ori_fft_block = fft_block;

		printf("[ORI] Running with fft...\\n");
		printf("[ORI] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_fft<<<fft_grid, fft_block>>>(fft_ori_source, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] fft took %f ms\\n\\n", kernel_time);
    // ---------------------------------------------------------------------------------------
"""

fft_ptb_running_code = """ 
	// PTB running
    // ---------------------------------------------------------------------------------------
		int fft_grid_dim_x = fft_grid.x;
		int fft_block_dim_x = fft_block.x;
		fft_grid.x = fft_blks == 0 ? fft_grid_dim_x : SM_NUM * fft_blks;
		fft_block.x = fft_block_dim_x;
		printf("[PTB] Running with fft...\\n");
		printf("[PTB] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_fft<<<fft_grid, fft_block>>>(fft_ptb_source, fft_grid_dim_x, fft_block_dim_x, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] fft took %f ms\\n\\n", kernel_time);
    // ---------------------------------------------------------------------------------------
"""

fft_gptb_variables_code = """
"""

fft_gptb_params_list = """fft_gptb_source, 
		ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
		0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_fft_grid.x * ori_fft_grid.y * ori_fft_grid.z"""

fft_gptb_params_list_new = """fft_gptb_source, 
	ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
	start_blk_no, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, end_blk_no, 0"""

fft_verify_code = """
    // ---------------------------------------------------------------------------------------
	cudaMemcpy(host_fft_ori_result, fft_ori_source, n_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_fft_gptb_result, fft_gptb_source, n_bytes, cudaMemcpyDeviceToHost);
    errors = 0;
	for (int i = 0; i < FFT_N * FFT_B; i++) {
		float v1 = host_fft_ori_result[i].x;
		float v2 = host_fft_gptb_result[i].x;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\\n", v1, v2);
		}
		if (i < 3) printf("%d %f %f\\n", i, v1, v2);

		v1 = host_fft_ori_result[i].y;
		v2 = host_fft_gptb_result[i].y;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\\n", v1, v2);
		}
	}
	if (errors > 0) {
		printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\\n", errors);
	}
	else {
		printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\\n");
	}
    // ---------------------------------------------------------------------------------------
"""

def get_fft_header_code():
    return fft_header_code

def get_fft_code_before_mix_kernel():
    return fft_variables_code + fft_solo_running_code + fft_ptb_running_code + fft_gptb_variables_code

def get_fft_code_after_mix_kernel():
    return fft_verify_code