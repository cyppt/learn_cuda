#include<julia_ch5.h>

__global__ void Kernel(unsigned char * ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * gridDim.x * blockDim.x;
    __shared__ float shared[16][16];
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
                                             (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
    __syncthreads(); //注释掉会产生不好看的图片

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

extern "C" void Juila(CPUBitmap *bitmap)
{
    unsigned char* dev_bitmap_ptr;
    cudaMalloc((void**)&dev_bitmap_ptr, bitmap->image_size());
    dim3 grids(DIM/16,DIM/16);
    dim3 threads(16,16);

    Kernel<<<grids,threads>>>(dev_bitmap_ptr);

    cudaMemcpy(bitmap->get_ptr(), dev_bitmap_ptr, bitmap->image_size(), cudaMemcpyDeviceToHost);

    bitmap->display_and_exit();

    cudaFree(dev_bitmap_ptr);

}

