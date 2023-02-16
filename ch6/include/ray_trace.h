#ifndef INCLUDE_RAY_TRACE_H_
#define INCLUDE_RAY_TRACE_H_

#include<iostream>
#include<cmath>
#include<cpu_bitmap.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define rnd(x)(x * rand() / RAND_MAX)
#define SPHERES 20
#define INF 2e10f
#define DIM 1024

struct Sphere
{
    float r,b,g;
    float radius;
    float x,y,z;
    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z; // 见readme
        }
        return -INF;
    }
};

extern "C" void ray_trace();

#endif //INCLUDE_RAY_TRACE_H_