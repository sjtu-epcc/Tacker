#pragma once
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h> 
#include <math.h> 
#include <sys/time.h>

#define HOT3D_BLOCK_SIZE 16
#define STR_SIZE 256

#define block_x_ 128 
#define block_y_ 2
#define block_z_ 1
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016; float chip_width = 0.016; /* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;