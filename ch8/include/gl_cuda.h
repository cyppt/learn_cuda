#ifndef INCLUDES_gl_cuda_H_
#define INCLUDES_gl_cuda_H_

#define GL_GLEXT_PROTOTYPES


#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cpu_bitmap.h>
// #include<cuda_runtime.h>
// #include<device_launch_parameters.h>
#include<cuda_gl_interop.h>
#include<book.h>

#define DIM 512

extern "C" void GlCudaKernel(int *argc, char **argv);

#endif //INCLUDES_gl_cuda_H_