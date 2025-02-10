
// const int tile_x = 32;
// const int tile_y = 4;

const int tile_x = 32;
const int tile_y = 2;

#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

#include "pets_common.h"
#define STENCIL_GRID_DIM (SM_NUM * 3)

