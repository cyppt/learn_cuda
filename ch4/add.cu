#include<add.h>
#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

extern "C" bool addKernel(int *a, int *b, int *c)
{
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<N,1>>>(dev_a, dev_b, dev_c); // 第一个参数 设备运行并行线程块数量

    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count;i++)
    {
    cudaGetDeviceProperties(&prop, i);
    printf(" --- General Information for device %d --- \n", i);
    printf("名字: %s\n", prop.name);
    std::cout << "设备计算能力:" << prop.major << "." << prop.minor << std::endl;
    std::cout << "显卡时钟频率:" << prop.clockRate * 1e-6f << " GHz" << std::endl;
    std::cout << "内存时钟频率:" << prop.memoryClockRate * 1e-3f << " MHz" << std::endl;
    std::cout << "内存总线带宽:" << prop.memoryBusWidth << " bit" << std::endl;
    std::cout << "总显存大小:" << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "总常量内存大小:" << prop.totalConstMem / 1024.0 << " KB" << std::endl;
    std::cout << "SM数量:" << prop.multiProcessorCount << std::endl;
    std::cout << "每个SM最大线程数:" << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个线程块(block)共享内存大小:" << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块(block)的最大线程数:" << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个线程块(block)的最大可用寄存器数:" << prop.regsPerBlock << std::endl;
    std::cout << "线程束(wrap)尺寸:" << prop.warpSize << std::endl;
    std::cout << "每个线程块(block)各个维度最大尺寸:" << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "每个线程格(grid)各个维度最大尺寸:" << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "最大存储间距:" << prop.memPitch / (1024.0 * 1024.0) << " MB" << std::endl;
    printf(" --- End General Information for device %d --- \n", i);
    }

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return true;
}