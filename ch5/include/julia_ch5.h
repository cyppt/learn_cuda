#ifndef __JUILA_CH5__
#define __JULIA_CH5__

#include<iostream>
#include<cmath>
#include<cpu_bitmap.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define DIM 512
#define PI 3.1415926535897932f

extern "C" void Juila(CPUBitmap *bitmap);

#endif //__JUILA_CH5__