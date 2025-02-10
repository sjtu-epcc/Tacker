//TEXTURE memory
texture<float,1> tex_x_float;

//constant memory
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];


void input_vec(char *fName,float *h_vec,int dim) {
  FILE* fid = fopen(fName, "rb");
  fread (h_vec, sizeof (float), dim, fid);
  fclose(fid);
  
}

#define SPMV_GRID_DIM gridDim.x
// #define SPMV_GRID_DIM 68