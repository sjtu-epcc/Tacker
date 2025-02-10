#include "header/cp_header.h"

extern "C" __global__ void ori_cp(int numatoms, float gridspacing, float * energygrid) {
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