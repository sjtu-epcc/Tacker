
// #define WMMA_GRID_DIM 68
#define WMMA_GRID_DIM gridDim.x

// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// const int WMMA_N = 16;
// const int WMMA_K = 16;
const float alpha = 2.0f;
const float beta = 2.0f;
