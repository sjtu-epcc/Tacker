// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
extern "C" int kernel_entry(float* Parameter_0_0, float** Result_99_0);
extern "C" void cuda_init();
extern "C" void cuda_free();
