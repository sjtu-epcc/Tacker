
// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.

__global__ void ori_cp(int numatoms, float gridspacing, float * energygrid) {
		unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
								+ threadIdx.x;
		unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
		unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
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


__global__ void ptb_cp(int numatoms, float gridspacing, float * energygrid, 
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,    
    int iteration) {
	unsigned int block_pos = blockIdx.x;
	
    int thread_id_x = threadIdx.x % block_dimension_x;
	int thread_id_y = threadIdx.x / block_dimension_x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
		}
		
		int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

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


__global__ void ptb2_cp(int numatoms, float gridspacing, float * energygrid, 
    int grid_dimension_x, int grid_dimension_y,    
    int iteration) {
	unsigned int block_pos = blockIdx.x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d\n", grid_dimension_x, grid_dimension_y);
    // }

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
		}
		
		int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

            unsigned int xindex  = __umul24(block_id_x, blockDim.x) * UNROLLX
								+ threadIdx.x;
			unsigned int yindex  = __umul24(block_id_y, blockDim.y) + threadIdx.y;
			unsigned int outaddr = (__umul24(grid_dimension_x, blockDim.x) * UNROLLX) * yindex
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


// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.
int copyatomstoconstbuf(float *atoms, int count, float zplane) {
	if (count > MAXATOMS) {
		printf("Atom count exceeds constant buffer storage capacity\n");
		return -1;
	}

	float atompre[4*MAXATOMS];
	int i;
	for (i=0; i<count*4; i+=4) {
		atompre[i    ] = atoms[i    ];
		atompre[i + 1] = atoms[i + 1];
		float dz = zplane - atoms[i + 2];
		atompre[i + 2]  = dz*dz;
		atompre[i + 3] = atoms[i + 3];
	}

	cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
	CUERR // check and clear any existing errors

	return 0;
}


/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
	dim3 size;
	int i;
	float *atoms;

	srand(54321);			// Ensure that atom placement is repeatable

	atoms = (float *) malloc(count * 4 * sizeof(float));
	*atombuf = atoms;

	// compute grid dimensions in angstroms
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;

	for (i=0; i<count; i++) {
		int addr = i * 4;
		atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
		atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
		atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
		atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
	}  

	return 0;
}


// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
// 
__device__ void mix_cp0(int numatoms, float gridspacing, float * energygrid, 
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int thread_step,
    int iteration) {
	unsigned int block_pos = blockIdx.x;
	
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
	int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    for (;; block_pos += CP_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
		}

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
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
}


__device__ void mix_cp1(int numatoms, float gridspacing, float * energygrid, 
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, 
    int thread_step, int iteration) {
	unsigned int block_pos = blockIdx.x + 68 * 2;
	
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
	int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    for (;; block_pos += CP_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
		}

        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
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
}


__device__ void general_ptb_cp0(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp1(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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


__device__ void general_ptb_cp2(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp3(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp4(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp5(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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


__device__ void general_ptb_cp6(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp7(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp8(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

__device__ void general_ptb_cp9(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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

extern "C" __global__ void g_general_ptb_cp(int numatoms, float gridspacing, float * energygrid, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base){
	// unsigned int block_pos = blockIdx.x + 68 * 2;   // TODO: why 68 * 2?
	
    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    
    // Calculate the 3D thread indices within the persistent thread block
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;
    // int thread_id_z = (threadIdx.x - thread_base) / (block_dimension_x * block_dimension_y);

    // ori
    // int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
	// int thread_id_y = (threadIdx.x - thread_base) / block_dimension_x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("function args: grid_dimension_x: %d, grid_dimension_y: %d, grid_dimension_z: %d, block_dimension_x: %d, block_dimension_y: %d, block_dimension_z: %d, ptb_start_block_pos: %d, ptb_iter_block_step: %d, ptb_end_block_pos: %d, thread_base: %d\n", 
    //         grid_dimension_x, grid_dimension_y, grid_dimension_z, block_dimension_x, block_dimension_y, block_dimension_z, ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos, thread_base);
    // }

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
		}

        // // ori
        // int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = block_pos / grid_dimension_x;


        int block_id_x = block_pos % grid_dimension_x;
		int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;
        // int block_id_z = block_pos / (grid_dimension_x * grid_dimension_y);

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