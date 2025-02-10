#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define PATH_BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;