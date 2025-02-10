#define TILE_N 16
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N * TILE_TB_HEIGHT)

#include "pets_common.h"
#define SGEMM_GRID_DIM (SM_NUM * 2)
