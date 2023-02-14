#include<bitmap.h>
#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b):r(a), i(b){} 

    __device__ float magnitude2(void)  // 声明 是在设备运行的函数
    {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }

};

__device__ int julia(int x, int y)
{
    const float scale = 1.0;  //归一化 -1 ~ 1
    float jx = scale * (float)(DIM / 2 -x)/(DIM / 2);
    float jy = scale * (float)(DIM / 2 -y)/(DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i ++)  // julia 集合内点的判断
    {
        a = a * a + c;
        if(a.magnitude2() > 1000) return 0;
    }

    return 1;
}

__global__ void Kernel(unsigned char * ptr)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;  //gridDim 获得传入三维线程的第一维大小（api接口是三维，二维默认第三维为1）
  
  int juliaValue = julia(x, y);

  ptr[offset] = juliaValue;
}

extern "C" void JuliaKernel(unsigned char (*ptr)[DIM])
{
    unsigned char *dev_ptr;
    cudaMalloc((void**)&dev_ptr, DIM * DIM * sizeof(unsigned char));

    dim3 grid(DIM, DIM); //api接口是三维，二维默认第三维为1
    Kernel<<<grid, 1>>>(dev_ptr);

    cudaMemcpy(ptr, dev_ptr, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(dev_ptr);
}