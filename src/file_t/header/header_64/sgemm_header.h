#define TILE_N 32
#define TILE_TB_HEIGHT 2
#define TILE_M (TILE_N * TILE_TB_HEIGHT)

#include "pets_common.h"
#define SGEMM_GRID_DIM (SM_NUM * 2)