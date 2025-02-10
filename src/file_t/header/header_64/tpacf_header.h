typedef unsigned long hist_t;
#define D2R M_PI/180.0
#define R2D 180.0/M_PI
#define R2AM 60.0*180.0/M_PI
#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0
#define NUM_BINS 20

struct cartesian {
  float x, y, z;  // cartesian coodrinates
};

#define WARP_SIZE 32
// #define NUM_BANKS 16
// #define LOG_NUM_BANKS 4

#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)
#define HISTS_PER_WARP 16
#define NUM_HISTOGRAMS  (NUM_WARPS*HISTS_PER_WARP)
// #define THREADS_PER_HIST (WARP_SIZE/HISTS_PER_WARP)

__constant__ float dev_binb[NUM_BINS+1];

#define TPACF_GRID_DIM gridDim.x
