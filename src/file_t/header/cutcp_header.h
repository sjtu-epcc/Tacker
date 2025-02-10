
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_SHIFT         5  /* # of bits to shift for mul/div by BIN_SIZE */
#define BIN_CACHE_MAXLEN 32  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */
#define REGION_SIZE     512  /* number of floats in lattice region */

#define NBRLIST_DIM  11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)

__constant__ int NbrListLen;
__constant__ int3 NbrList[NBRLIST_MAXLEN];

#include "pets_common.h"
#define CUTCP_GRID_DIM (SM_NUM * 4)


// ------------------------------------------------------------------------------
typedef struct LatticeDim_t {
    /* Number of lattice points in x, y, z dimensions */
    int nx, ny, nz;

    /* Lowest corner of lattice */
    Vec3 lo;

    /* Lattice spacing */
    float h;
} LatticeDim;

typedef struct Lattice_t {
    LatticeDim dim;
    float *lattice;
} Lattice;

#define ERRTOL 1e-4f
#define NOKERNELS             0
#define CUTOFF1               1
#define CUTOFF6              32
#define CUTOFF6OVERLAP       64
#define CUTOFFCPU         16384

int appenddata(const char *filename, int size, double time) {
	FILE *fp;
	fp=fopen(filename, "a");
	if (fp == NULL) {
		printf("error appending to file %s..\n", filename);
		return -1;
	}
	fprintf(fp, "%d  %.3f\n", size, time);
	fclose(fp);
	return 0;
}


LatticeDim lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h) {
	LatticeDim ret;

	ret.nx = (int) floorf((hi.x-lo.x)/h) + 1;
	ret.ny = (int) floorf((hi.y-lo.y)/h) + 1;
	ret.nz = (int) floorf((hi.z-lo.z)/h) + 1;
	ret.lo = lo;
	ret.h = h;

	return ret;
}


Lattice *create_lattice(LatticeDim dim) {
	int size;
	Lattice *lat = (Lattice *)malloc(sizeof(Lattice));

	if (lat == NULL) {
		fprintf(stderr, "Out of memory\n");
		exit(1);
	}

	lat->dim = dim;

	/* Round up the allocated size to a multiple of 8 */
	size = ((dim.nx * dim.ny * dim.nz) + 7) & ~7;
	lat->lattice = (float *)calloc(size, sizeof(float));

	if (lat->lattice == NULL) {
		fprintf(stderr, "Out of memory\n");
		exit(1);
	}

	return lat;
}


void destroy_lattice(Lattice *lat) {
	if (lat) {
		free(lat->lattice);
		free(lat);
	}
}


int prepare_input(Lattice *lattice,
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    float4 *binBaseAddr,
    int3* nbrlist) 
{
    int nx = lattice->dim.nx;
	int ny = lattice->dim.ny;
	int nz = lattice->dim.nz;
	float xlo = lattice->dim.lo.x;
	float ylo = lattice->dim.lo.y;
	float zlo = lattice->dim.lo.z;
	float h = lattice->dim.h;
	int natoms = atoms->size;
	Atom *atom = atoms->atoms;

	int nbrlistlen = 0;
	int binHistoFull[BIN_DEPTH+1] = { 0 };   /* clear every array element */
	int binHistoCover[BIN_DEPTH+1] = { 0 };  /* clear every array element */
	int num_excluded = 0;

	int xRegionDim, yRegionDim, zRegionDim;
	// int xRegionIndex, yRegionIndex, zRegionIndex;
	// int xOffset, yOffset, zOffset;
	int lnx, lny, lnz;
	// int lnall;
	// float *regionZeroAddr;
    // float *thisRegion;
	// float *regionZeroCuda;
	// int index, indexRegion;

    int c;
	int3 binDim;
	int nbins;
	// float4 *binBaseAddr, *binZeroAddr;
	float4 *binZeroAddr;
	int *bincntBaseAddr, *bincntZeroAddr;
	Atoms *extra = NULL;

    int i, j, k, n;
	int sum, total;

	float avgFillFull, avgFillCover;
	// const float cutoff2 = cutoff * cutoff;
	// const float inv_cutoff2 = 1.f / cutoff2;

    xRegionDim = (int) ceilf(nx/8.f);
	yRegionDim = (int) ceilf(ny/8.f);
	zRegionDim = (int) ceilf(nz/8.f);

    lnx = 8 * xRegionDim;
	lny = 8 * yRegionDim;
	lnz = 8 * zRegionDim;
	// printf("lnx %d, lny %d lnz %d \n", lnx, lny, lnz);
	// lnall = lnx * lny * lnz;
	// printf("lnall: %d \n", lnall);

    /* will receive energies from CUDA */
	// regionZeroAddr = (float *) malloc(lnall * sizeof(float));

	/* create bins */
	c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
	binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
	binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
	binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
	nbins = binDim.x * binDim.y * binDim.z;
	// binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
	binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

	bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
	bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

	/* create neighbor list */
	if (ceilf(BIN_LENGTH / (8*h)) == floorf(BIN_LENGTH / (8*h))) {
		float s = sqrtf(3);
		float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
		int cnt = 0;
		/* develop neighbor list around 1 cell */
		if (2*c + 1 > NBRLIST_DIM) {
			fprintf(stderr, "must have cutoff <= %f\n",
				(NBRLIST_DIM-1)/2 * BIN_LENGTH);
			return -1;
		}
		for (k = -c;  k <= c;  k++) {
			for (j = -c;  j <= c;  j++) {
				for (i = -c;  i <= c;  i++) {
					if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
					nbrlist[cnt].x = i;
					nbrlist[cnt].y = j;
					nbrlist[cnt].z = k;
					cnt++;
				}
			}
		}
		nbrlistlen = cnt;
	} else if (8*h <= 2*BIN_LENGTH) {
		float s = 2.f*sqrtf(3);
		float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
		int cnt = 0;
		/* develop neighbor list around 3-cube of cells */
		if (2*c + 3 > NBRLIST_DIM) {
			fprintf(stderr, "must have cutoff <= %f\n",
				(NBRLIST_DIM-3)/2 * BIN_LENGTH);
			return -1;
		}
		for (k = -c;  k <= c;  k++) {
			for (j = -c;  j <= c;  j++) {
				for (i = -c;  i <= c;  i++) {
					if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
					nbrlist[cnt].x = i;
					nbrlist[cnt].y = j;
					nbrlist[cnt].z = k;
					cnt++;
				}
			}
		}
		nbrlistlen = cnt;
	} else {
		fprintf(stderr, "must have h <= %f\n", 0.25 * BIN_LENGTH);
		return -1;
	}

    /* perform geometric hashing of atoms into bins */
	{
		/* array of extra atoms, permit average of one extra per bin */
		Atom *extra_atoms = (Atom *) calloc(nbins, sizeof(Atom));
		int extra_len = 0;
    
		for (n = 0;  n < natoms;  n++) {
			float4 p;
			p.x = atom[n].x - xlo;
			p.y = atom[n].y - ylo;
			p.z = atom[n].z - zlo;
			p.w = atom[n].q;
			i = (int) floorf(p.x * BIN_INVLEN);
			j = (int) floorf(p.y * BIN_INVLEN);
			k = (int) floorf(p.z * BIN_INVLEN);

			if (i >= -c && i < binDim.x - c &&
				j >= -c && j < binDim.y - c &&
				k >= -c && k < binDim.z - c &&
				atom[n].q != 0) {

				int index = (k * binDim.y + j) * binDim.x + i;
				float4 *bin = binZeroAddr + index * BIN_DEPTH;
				int bindex = bincntZeroAddr[index];

				if (bindex < BIN_DEPTH) {
					/* copy atom into bin and increase counter for this bin */
					bin[bindex] = p;
					bincntZeroAddr[index]++;
				} else {
					/* add index to array of extra atoms to be computed with CPU */
					if (extra_len >= nbins) {
						fprintf(stderr, "exceeded space for storing extra atoms\n");
						return -1;
					}
					extra_atoms[extra_len] = atom[n];
					extra_len++;
				}
			} else {
				/* excluded atoms are either outside bins or neutrally charged */
				num_excluded++;
			}
    	}

		/* Save result */
		extra = (Atoms *)malloc(sizeof(Atoms));
		extra->atoms = extra_atoms;
		extra->size = extra_len;
	}

    /* bin stats */
	sum = total = 0;
	for (n = 0;  n < nbins;  n++) {
		binHistoFull[ bincntBaseAddr[n] ]++;
		sum += bincntBaseAddr[n];
		total += BIN_DEPTH;
	}
	avgFillFull = sum / (float) total;
	sum = total = 0;
	for (k = 0;  k < binDim.z - 2*c;  k++) {
		for (j = 0;  j < binDim.y - 2*c;  j++) {
			for (i = 0;  i < binDim.x - 2*c;  i++) {
				int index = (k * binDim.y + j) * binDim.x + i;
				binHistoCover[ bincntZeroAddr[index] ]++;
				sum += bincntZeroAddr[index];
				total += BIN_DEPTH;
			}
		}
	}
	avgFillCover = sum / (float) total;

    // no meaning
    int return_value = (int) (avgFillCover + avgFillFull) - nbrlistlen;
    return return_value;
}