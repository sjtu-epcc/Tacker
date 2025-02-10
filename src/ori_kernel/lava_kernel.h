//lava_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


#ifndef LAVA_H
#define LAVA_H

typedef struct
{

	float x, y, z;
} THREE_VECTOR;

typedef struct
{
	float v, x, y, z;
} FOUR_VECTOR;

typedef struct nei_str
{
	// neighbor box
	int x, y, z;
	int number;
	long offset;
} nei_str;

typedef struct box_str
{
	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];
} box_str;

typedef struct par_str
{
	float alpha;
} par_str;

typedef struct dim_str
{
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
} dim_str;

#endif


struct OriLAVAParamsStruct {
    par_str d_par_gpu;
    dim_str d_dim_gpu;
    box_str* d_box_gpu;
    FOUR_VECTOR* d_rv_gpu;
    float* d_qv_gpu;
    FOUR_VECTOR* d_fv_gpu;
};

class OriLAVAKernel : public Kernel {
public:
    // 构造函数
    OriLAVAKernel(int id);

    // 析构函数
    ~OriLAVAKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel(){};  
    OriLAVAParamsStruct* LAVAKernelParams;
    
};