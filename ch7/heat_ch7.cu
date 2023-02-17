#include<heat_ch7.h>

struct DataBlock
{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

// 书上没有给出的函数
template<typename T>
void swap(T a, T b)
{
    T temp;
    temp = b;
    b = a;
    a = temp;
}

__global__ void copy_const_kernel(float *input_ptr, const float *const_refer_ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(const_refer_ptr[offset] != 0) input_ptr[offset] = const_refer_ptr[offset];
}

__global__ void blend_kernel(float *current_srceen, const float *last_screen)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if(x == 0) left++;
    if(x == DIM - 1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if(y == 0) top += DIM;
    if(y == DIM - 1) top -= DIM;

    current_srceen[offset] = last_screen[offset] + SPEED * (last_screen[top] + 
    last_screen[bottom] + last_screen[left] + last_screen[right] -
    last_screen[offset] * 4);
}

//好像书上没有这个函数的定义 我根据网上资料写了一个
__global__ void float_to_color(unsigned char *ptr, const float *current_srceen)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float fscale = current_srceen[offset];

    fscale = (fscale - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
    int grayValue =  fscale * 255;
    int rgbValues[3] = {0};

    if (grayValue < 64)
	{
		rgbValues[2] = 0;
		rgbValues[1] = (int)(4 * grayValue);
		rgbValues[0] = 255;
	}
	else if (grayValue >= 64 && grayValue < 128)
	{
		rgbValues[2] = 0;
		rgbValues[1] = 255;
		rgbValues[0] = (int)(2 * 255 - 4 * grayValue);
	}
	else if (grayValue >= 128 && grayValue < 192)
	{
		rgbValues[2] = (int)(4 * grayValue - 2 * 255);
		rgbValues[1] = 255;
		rgbValues[0] = 0;
	}
	else
	{
		rgbValues[2] = 255;
		rgbValues[1] = (int)(4 * 255 - 4 * grayValue);
		rgbValues[0] = 0;
	}

    ptr[offset * 4 + 0] = rgbValues[0];
    ptr[offset * 4 + 1] = rgbValues[1];
    ptr[offset * 4 + 2] = rgbValues[2];
    ptr[offset * 4 + 3] = 255;

}


void anim_gpu(DataBlock *d)
{
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16,16);
    CPUAnimBitmap *bitmap = d->bitmap;
    float elapsedTime;

    cudaEventRecord(d->start, 0);
    for (int i = 0; i < UPDATA_RATE_PER_STEP; i++)
    {
        copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);

        swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
    cudaEventRecord(d->stop, 0);
    cudaEventSynchronize(d->stop);
    cudaMemcpy(bitmap->get_ptr(),d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Average Time per frame : %3.1f ms \n", d->totalTime/d->frames);
    std::cin.get();
}

void anim_exit(DataBlock *d)
{
    cudaFree(d->dev_constSrc);
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);

    cudaEventDestroy(d->start);
    cudaEventDestroy(d->stop);
}

extern "C" void HeatKernel()
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    cudaEventCreate(&data.start);
    cudaEventCreate(&data.stop);

    cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
    cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());

    float *temp = (float*)malloc(bitmap.image_size());
    for(int i = 0; i < DIM*DIM; i++)
    {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if((x > 300) && (x < 600) && (y > 310) && (y < 601)) temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for(int y = 800; y< 900; y++)
    {
        for(int x = 400; x < 500; x++)
        {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
    for(int i = 0; i < DIM*DIM; i++)
    {
        temp[i] = 0;
    }
    for(int y = 800; y< DIM; y++)
    {
        for(int x = 0; x < 200; x++)
        {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
    free(temp);
    bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void *))anim_exit);
}