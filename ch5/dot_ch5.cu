#include<dot_ch5.h>

__global__ void dot(float *a ,float *b, float * c)
{
    __shared__ float cache[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cache_index = threadIdx.x;
    float temp = 0;

    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cache_index] = temp;
    // 对线程块中的线程进行同步
    __syncthreads();
    // 归约运算
    int i = blockDim.x / 2;
    while (i != 0)  
    {
      //每个线程块中都有这个函数 不需要 用for遍历cache
      if (cache_index < i) cache[cache_index] += cache[cache_index + i];
      __syncthreads(); // 等待所有线程执行完上述语句
      i /= 2;  
    }
    if (cache_index == 0) c[blockIdx.x] = cache[0];
}

extern "C" void DotKernel(float *a ,float *b, float *partial_c)
{
    float* dev_a, *dev_b, *dev_partial_c;

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, BLOCKS_PER_GRID * sizeof(float));

    cudaMemcpy(dev_a, a ,N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b ,N * sizeof(float), cudaMemcpyHostToDevice);

    dot <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, BLOCKS_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
}