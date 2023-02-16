#include<ray_trace.h>

__global__ void kernel(unsigned char *ptr, Sphere* s)
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
    cudaEventRecord(start , 0);
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    Sphere *dev_s;
    Sphere *s;

    s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for(int i = 0; i < SPHERES; i++)
    {
        s[i].r = rnd(1.0f);
        s[i].g = rnd(1.0f);
        s[i].b = rnd(1.0f);
        s[i].x = rnd(1000.0f) - 500;
        s[i].y = rnd(1000.0f) - 500;
        s[i].z = rnd(1000.0f) - 500;
        s[i].radius = rnd(100.0f) + 20;
    }

    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
    cudaMalloc((void**)&dev_s, sizeof(Sphere) * SPHERES);

    cudaMemcpy(dev_s, s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    cudaEventRecord(start , 0);
    kernel<<<grids, threads>>>(dev_bitmap, dev_s);
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
    cudaFree(dev_s);
    free(s);
}