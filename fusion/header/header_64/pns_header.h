
static int N, s, t, N2, NSQUARE2;
uint32 host_mt[MERS_N];

float results[4];
float *h_vars, *hp_vars;
int *h_maxs, *hp_maxs;

#define MAX_DEVICE_MEM 2000000000
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BITS 8
