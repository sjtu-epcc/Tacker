solo_ptb_cp_blks = 6

cp_header_code = """
#include "header/cp_header.h"
#include "kernel/cp_kernel.cu"
"""
cp_variables_code = """
    // cp variables
    // ---------------------------------------------------------------------------------------
		int cp_blks = 8;
	    int cp_iter = 1;
        float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

        float *ori_output;	
		float *ptb_output;
		float *gptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
		cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
		float *host_gptb_energy = (float *) malloc(volmemsz);

        dim3 cp_grid, cp_block, ori_cp_grid, ori_cp_block;
        int atomstart = 1;
		int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------
"""

cp_solo_running_code = """
    // SOLO running
    // ---------------------------------------------------------------------------------------
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		ori_cp_grid = cp_grid;
		ori_cp_block = cp_block;
		printf("[ORI] Running with cp...\\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\\n\\n", kernel_time);

        cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------
"""

cp_ptb_running_code = f"""
	// PTB running
    // ---------------------------------------------------------------------------------------
        int solo_ptb_cp_blks = {solo_ptb_cp_blks};
	    cp_iter = 1;
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		cp_grid.x = solo_ptb_cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * solo_ptb_cp_blks;
		cp_grid.y = 1;
		printf("[PTB] Running with cp...\\n");
		printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb2_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] cp took %f ms\\n\\n", kernel_time);

        cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------
"""

cp_verify_code = """
	// Checking results
    // ---------------------------------------------------------------------------------------
    	cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	    cudaMemcpy(host_gptb_energy, gptb_output, volmemsz,  cudaMemcpyDeviceToHost);
            
        errors = 0;
        for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
            float v1 = host_ori_energy[i];
            float v2 = host_gptb_energy[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 10) printf("%f %f\\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORI VERSION does not agree with GPTB VERSION! %d errors!\\n", errors);
        }
        else {
            printf("Results verified: ORIG VERSION and GPTB VERSION agree.\\n");
        }
	// ---------------------------------------------------------------------------------------
"""

cp_gptb_variables_code = f"""
		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);
"""

cp_gptb_params_list = """runatoms, 0.1, gptb_output, 
    ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z"""

cp_gptb_params_list_new = """runatoms, 0.1, gptb_output,
    ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    start_blk_no, gptb_kernel_grid.x * gptb_kernel_grid.y * gptb_kernel_grid.z, end_blk_no, 0"""

cp_gptb_kernel_def_code = """
__global__ void general_ptb_cp(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int xindex  = __umul24(block_id_x, block_dimension_x) * UNROLLX
                            + thread_id_x;
        unsigned int yindex  = __umul24(block_id_y, block_dimension_y) + thread_id_y;
        unsigned int outaddr = (__umul24(grid_dimension_x, block_dimension_x) * UNROLLX) * yindex
                                + xindex;

        float coory = gridspacing * yindex;
        float coorx = gridspacing * xindex;

        float energyvalx1=0.0f;
        float energyvalx2=0.0f;
        float energyvalx3=0.0f;
        float energyvalx4=0.0f;
        float energyvalx5=0.0f;
        float energyvalx6=0.0f;
        float energyvalx7=0.0f;
        float energyvalx8=0.0f;

        float gridspacing_u = gridspacing * BLOCKSIZEX;

        int atomid;
        for (atomid=0; atomid<numatoms; atomid++) {
            float dy = coory - atominfo[atomid].y;
            float dyz2 = (dy * dy) + atominfo[atomid].z;

            float dx1 = coorx - atominfo[atomid].x;
            float dx2 = dx1 + gridspacing_u;
            float dx3 = dx2 + gridspacing_u;
            float dx4 = dx3 + gridspacing_u;
            float dx5 = dx4 + gridspacing_u;
            float dx6 = dx5 + gridspacing_u;
            float dx7 = dx6 + gridspacing_u;
            float dx8 = dx7 + gridspacing_u;

            energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dyz2));
            energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dyz2));
            energyvalx3 += atominfo[atomid].w * (1.0f / sqrtf(dx3*dx3 + dyz2));
            energyvalx4 += atominfo[atomid].w * (1.0f / sqrtf(dx4*dx4 + dyz2));
            energyvalx5 += atominfo[atomid].w * (1.0f / sqrtf(dx5*dx5 + dyz2));
            energyvalx6 += atominfo[atomid].w * (1.0f / sqrtf(dx6*dx6 + dyz2));
            energyvalx7 += atominfo[atomid].w * (1.0f / sqrtf(dx7*dx7 + dyz2));
            energyvalx8 += atominfo[atomid].w * (1.0f / sqrtf(dx8*dx8 + dyz2));
        }

        energygrid[outaddr]   += energyvalx1;
        energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
        energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
        energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
        energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
        energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
        energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
        energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
    }
}
"""

def get_cp_code_before_mix_kernel():
    return cp_variables_code + cp_solo_running_code + cp_ptb_running_code + cp_gptb_variables_code
def get_cp_header_code():
    return cp_header_code
def get_cp_code_after_mix_kernel():
    return cp_verify_code