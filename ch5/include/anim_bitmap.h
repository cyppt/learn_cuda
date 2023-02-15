#ifndef INCLUDES_NIM_BITMAP_H_
#define INCLUDES_ANIM_BITMAP_H_

#include<cpu_anim.h>

#define DIM 320

/*check if the compiler is of C++*/
#ifdef __cplusplus
//修饰符extern "C"是CUDA和C++混合编程时必须的。

struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

extern "C" bool bitmapKernel(DataBlock *data);

#endif


#endif /* INCLUDES_NIM_BITMAP_H_ */