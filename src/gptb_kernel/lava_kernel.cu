#include "header/lava_header.h"

extern "C" __global__ void general_ptb_lava(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								float* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu,
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int bx = block_pos;
        int tx = thread_id_x;
        int wtx = tx;
        
        if(bx<d_dim_gpu.number_boxes){
            // parameters
            float a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

            // home box
            int first_i;
            FOUR_VECTOR* rA;
            FOUR_VECTOR* fA;
            __shared__ FOUR_VECTOR rA_shared[100];

            // nei box
            int pointer;
            int k = 0;
            int first_j;
            FOUR_VECTOR* rB;
            float* qB;
            int j = 0;
            __shared__ FOUR_VECTOR rB_shared[100];
            __shared__ double qB_shared[100];

            // common
            float r2;
            float u2;
            float vij;
            float fs;
            float fxij;
            float fyij;
            float fzij;
            THREE_VECTOR d;

            // home box - box parameters
            first_i = d_box_gpu[bx].offset;

            // home box - distance, force, charge and type parameters
            rA = &d_rv_gpu[first_i];
            fA = &d_fv_gpu[first_i];

            // home box - shared memory
            while(wtx<NUMBER_PAR_PER_BOX){
                rA_shared[wtx] = rA[wtx];
                wtx = wtx + NUMBER_THREADS;
            }
            wtx = tx;

            // synchronize threads  - not needed, but just to be safe
            asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

            // loop over neiing boxes of home box
            for (k=0; k<(1+d_box_gpu[bx].nn); k++){

                if(k==0){
                    pointer = bx; // set first box to be processed to home box
                }
                else{
                    pointer = d_box_gpu[bx].nei[k-1].number; 
                    // remaining boxes are nei boxes
                }

                // nei box - box parameters
                first_j = d_box_gpu[pointer].offset;

                // nei box - distance, (force), charge and (type) parameters
                rB = &d_rv_gpu[first_j];
                qB = &d_qv_gpu[first_j];

                // nei box - shared memory
                while(wtx<NUMBER_PAR_PER_BOX){
                    rB_shared[wtx] = rB[wtx];
                    qB_shared[wtx] = qB[wtx];
                    wtx = wtx + NUMBER_THREADS;
                }
                wtx = tx;

                // synchronize threads because in next section each thread accesses data brought in by different threads here
                asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

                // loop for the number of particles in the home box
                while(wtx<NUMBER_PAR_PER_BOX){

                    // loop for the number of particles in the current nei box
                    for (j=0; j<NUMBER_PAR_PER_BOX; j++){

                        r2 = (float)rA_shared[wtx].v + (float)rB_shared[j].v - DOT((float)rA_shared[wtx],(float)rB_shared[j]); 
                        u2 = a2*r2;
                        vij= exp(-u2);
                        fs = 2*vij;

                        d.x = (float)rA_shared[wtx].x  - (float)rB_shared[j].x;
                        fxij=fs*d.x;
                        d.y = (float)rA_shared[wtx].y  - (float)rB_shared[j].y;
                        fyij=fs*d.y;
                        d.z = (float)rA_shared[wtx].z  - (float)rB_shared[j].z;
                        fzij=fs*d.z;

                        fA[wtx].v +=  (double)((float)qB_shared[j]*vij);
                        fA[wtx].x +=  (double)((float)qB_shared[j]*fxij);
                        fA[wtx].y +=  (double)((float)qB_shared[j]*fyij);
                        fA[wtx].z +=  (double)((float)qB_shared[j]*fzij);

                    }

                    // increment work thread index
                    wtx = wtx + NUMBER_THREADS;

                }

                // reset work index
                wtx = tx;

                // synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
                asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

            }

        }
    }

}

extern "C" __device__ void internal_general_ptb_lava(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								float* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu,
    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)
{

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;
    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int bx = block_pos;
        int tx = thread_id_x;
        int wtx = tx;
        
        if(bx<d_dim_gpu.number_boxes){
            // parameters
            float a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

            // home box
            int first_i;
            FOUR_VECTOR* rA;
            FOUR_VECTOR* fA;
            __shared__ FOUR_VECTOR rA_shared[100];

            // nei box
            int pointer;
            int k = 0;
            int first_j;
            FOUR_VECTOR* rB;
            float* qB;
            int j = 0;
            __shared__ FOUR_VECTOR rB_shared[100];
            __shared__ double qB_shared[100];

            // common
            float r2;
            float u2;
            float vij;
            float fs;
            float fxij;
            float fyij;
            float fzij;
            THREE_VECTOR d;

            // home box - box parameters
            first_i = d_box_gpu[bx].offset;

            // home box - distance, force, charge and type parameters
            rA = &d_rv_gpu[first_i];
            fA = &d_fv_gpu[first_i];

            // home box - shared memory
            while(wtx<NUMBER_PAR_PER_BOX){
                rA_shared[wtx] = rA[wtx];
                wtx = wtx + NUMBER_THREADS;
            }
            wtx = tx;

            // synchronize threads  - not needed, but just to be safe
            asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

            // loop over neiing boxes of home box
            for (k=0; k<(1+d_box_gpu[bx].nn); k++){

                if(k==0){
                    pointer = bx; // set first box to be processed to home box
                }
                else{
                    pointer = d_box_gpu[bx].nei[k-1].number; 
                    // remaining boxes are nei boxes
                }

                // nei box - box parameters
                first_j = d_box_gpu[pointer].offset;

                // nei box - distance, (force), charge and (type) parameters
                rB = &d_rv_gpu[first_j];
                qB = &d_qv_gpu[first_j];

                // nei box - shared memory
                while(wtx<NUMBER_PAR_PER_BOX){
                    rB_shared[wtx] = rB[wtx];
                    qB_shared[wtx] = qB[wtx];
                    wtx = wtx + NUMBER_THREADS;
                }
                wtx = tx;

                // synchronize threads because in next section each thread accesses data brought in by different threads here
                asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

                // loop for the number of particles in the home box
                while(wtx<NUMBER_PAR_PER_BOX){

                    // loop for the number of particles in the current nei box
                    for (j=0; j<NUMBER_PAR_PER_BOX; j++){

                        r2 = (float)rA_shared[wtx].v + (float)rB_shared[j].v - DOT((float)rA_shared[wtx],(float)rB_shared[j]); 
                        u2 = a2*r2;
                        vij= exp(-u2);
                        fs = 2*vij;

                        d.x = (float)rA_shared[wtx].x  - (float)rB_shared[j].x;
                        fxij=fs*d.x;
                        d.y = (float)rA_shared[wtx].y  - (float)rB_shared[j].y;
                        fyij=fs*d.y;
                        d.z = (float)rA_shared[wtx].z  - (float)rB_shared[j].z;
                        fzij=fs*d.z;

                        fA[wtx].v +=  (double)((float)qB_shared[j]*vij);
                        fA[wtx].x +=  (double)((float)qB_shared[j]*fxij);
                        fA[wtx].y +=  (double)((float)qB_shared[j]*fyij);
                        fA[wtx].z +=  (double)((float)qB_shared[j]*fzij);

                    }

                    // increment work thread index
                    wtx = wtx + NUMBER_THREADS;

                }

                // reset work index
                wtx = tx;

                // synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
                asm volatile("bar.sync %0, %1;" : : "r"(10), "r"(128) : "memory");

            }

        }
    }

}