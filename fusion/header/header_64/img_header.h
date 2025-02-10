#define ByteSwap16(n) ( ((((unsigned int) n) << 8) & 0xFF00) | ((((unsigned int) n) >> 8) & 0x00FF) )
#define WARP_SIZE 32
#define BINSp 257

#define REP 11
#define NUM_BLOCKS (68 * 10 * 1)
#define THREADS 128

#include "pets_common.h"
#define IMG_GRID_DIM (SM_NUM * 1)