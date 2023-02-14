# 第5章 线程协作  
**线程通讯与同步** 
### 示例1知识点

* 核函数参数说明 
>核函数第一个参数表示线程块（block）的个数，第二个参数表示每个线程块有几个线程（thread。  
>线程块数量不能大于65535 个 线程最大数量与设备属性maxThreadsPerBlock的值有关  

* 线程块与线程并用
>当要计算大数据时，使用类似二维数组的方法防止越界。  
```int tid = threadIdx.x + blockIdx.x * blockDim.x;```
>对于所有线程来说，blockDim 是常数，threadIdx 与 blockIdx 一致 是cuda内置索引变量，blockDim 和 gridDim（线程格中每一维线程块的数量）类似   

* 关于任务分配
>不过这还有个问题，假设有N个并行任务，使用核函数```add<<<N/128, 128>>>```存在一个问题,那就是N/128 int/int，当N不是128的整数倍时，会启动少于预期数量的线程  
>处理是```add<<<(N+127)/128, 128>>>```来确保启动足够多的线程，同时在核函数中要判断是否越界
```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < N) // to do
else // do nothing
```  
* 关于任意长度矢量的求和
> 我们知道，线程块的数量要小于 65535个，当矢量长度超过65535 * 128时，调用上述核函数会失败，解决方法，在核函数内加入循环
```
__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x; // 递增的步长，GPU每一步共执行线程块*线程次
    }
}
```
> 完整程序见示例1