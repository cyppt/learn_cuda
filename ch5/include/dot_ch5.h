#ifndef INCLUDE_DOT_CH5_H_
#define INCLUDE_DOT_CH5_H_

#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define imin(a,b) (a<b?a:b)

const int N = 1024;
const int THREADS_PER_BLOCK = 256;
const int BLOCKS_PER_GRID = imin(32 , (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

extern "C" void DotKernel(float *a ,float *b, float *partial_c);

#endif /* INCLUDE_DOT_CH5_H_ */