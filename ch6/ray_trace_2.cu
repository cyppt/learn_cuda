#include<ray_trace_2.h>

__constant__ Sphere s[SPHERES]; // 常量内存

__global__ void kernel(unsigned char *ptr)
{   
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    // 图像坐标偏移，使z过原点
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);
    float r = 0, g = 0, b = 0;
    float maxz = -INF; // 具体原因见readme
    for (int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }
    ptr[offset * 4 + 0] = int(r * 255);
    ptr[offset * 4 + 1] = int(g * 255);
    ptr[offset * 4 + 2] = int(b * 255);
    ptr[offset * 4 + 3] = 255;
}

extern "C" void ray_trace()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    Sphere *temp_s;

    temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for(int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    cudaMemcpyToSymbol(s,temp_s, sizeof(Sphere) * SPHERES); // 常量内存 直接cudaMemcpy 不用malloc和free

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    cudaEventRecord(start , 0);
    kernel<<<grids, threads>>>(dev_bitmap);
    cudaEventRecord(stop , 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to generate:" << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
    free(temp_s);
}