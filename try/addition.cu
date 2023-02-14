#include <addition.h>
 __global__ void add(int *a, int *b, int *c)   //在设备用运行的
 {
     *c=*a+*b;
 }
 
 extern "C" bool addition(int a, int b, int *c)
 {
     int *d_a, *d_b, *d_c;
     int size=sizeof(int);
     
     cudaMalloc((void **)&d_a, size);
     cudaMalloc((void **)&d_b, size);
     cudaMalloc((void **)&d_c, size);
     
     cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);  //不能在主机代码中对cudaMalloc，分配的指针进行内存读写操作
     cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice); // 源指针位于主机上，传递给设备，可以反过来，也可DeviceToDevice
     
     add<<<1,1>>>(d_a, d_b, d_c);  //设备函数调用

    //  cudaDevicePror pror;
    //  int count;
    //  cudaGetDeviceCount(&count);
    //  for (int i = 0; i < count;i++)
    //  {
    //     cudaGetDeviceProperties(&prop, i);
    //     printf(" --- General Information for device %d --- \n", i);
    //     printf("Name: %s\n", prop.name);
    //     printf("Compute capability: %d.%      d\n", prop.major, prop.minor);
        
    //  }
     
     cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
     
     cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
     return true;
}