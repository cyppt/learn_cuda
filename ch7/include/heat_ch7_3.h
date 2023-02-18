#ifndef INCLUDES_HEAT_CH7_3_H_
#define INCLUDES_HEAT_CH7_3_H_

#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cpu_bitmap.h>
#include<cpu_anim.h>
#include<cuda_runtime.h>


#define DIM 1024
#define UPDATA_RATE_PER_STEP 90
#define SPEED 0.25f
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f


extern "C" void HeatKernel();

#endif //INCLUDES_HEAT_CH7_3_H_
