#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

const int tile_x = 32;
const int tile_y = 4;

#include "pets_common.h"
#define REGTIL_GRID_DIM (SM_NUM * 1)