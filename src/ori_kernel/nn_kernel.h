//nw_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"

#ifndef NN_H
#define NN_H

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

#endif

struct OriNNParamsStruct {
    LatLong *d_locations; 
    float *d_distances; 
    int numRecords;
    float lat; 
    float lng;
};

class OriNNKernel : public Kernel {
public:
    // 构造函数
    OriNNKernel(int id);

    // 析构函数
    ~OriNNKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel(){};  
    OriNNParamsStruct* NNKernelParams;
    
};