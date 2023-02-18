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

texture<float> texConstSrc;
texture<float> texIn;
texture<float> texOut;

float * dev_temp; // 用于swap函数

// 书上没有给出的函数  注意是设备之间！！！得用cuda函数
void swap(float* dev_a, float * dev_b, long size)
{
    //float * dev_temp;  //写成全局变量 不用每次分配内存
    //cudaMalloc((void **)&dev_temp, size);
    
    cudaMemcpy(dev_temp, dev_a, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_a, dev_b, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_b, dev_temp, size, cudaMemcpyDeviceToDevice);
    //cudaFree(dev_temp);
}

__global__ void copy_const_kernel(float *input_ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex1Dfetch(texConstSrc,offset);
    if(c != 0) input_ptr[offset] = c;
}

__global__ void blend_kernel(float *dst, bool dstOut)
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

    float top_heat, left_heat, center_heat, right_heat, bottom_heat;
    if(dstOut)
    {
        top_heat = tex1Dfetch(texIn,top);
        left_heat = tex1Dfetch(texIn,left);
        center_heat = tex1Dfetch(texIn,offset);
        right_heat = tex1Dfetch(texIn,right);
        bottom_heat = tex1Dfetch(texIn,bottom);
    }
    else
    {
        top_heat = tex1Dfetch(texOut,top);
        left_heat = tex1Dfetch(texOut,left);
        center_heat = tex1Dfetch(texOut,offset);
        right_heat = tex1Dfetch(texOut,right);
        bottom_heat = tex1Dfetch(texOut,bottom);
    }
    dst[offset] = center_heat + SPEED * (top_heat + bottom_heat + 
    right_heat + left_heat - 4 * center_heat);
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
    volatile bool dstOut = true;

    cudaEventRecord(d->start, 0);
    for (int i = 0; i < UPDATA_RATE_PER_STEP; i++)
    {
        float *in, *out;
        if(dstOut)
        {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else
        {
            in = d->dev_outSrc;
            out = d->dev_inSrc;
        }
        copy_const_kernel<<<blocks, threads>>>(in);
        blend_kernel<<<blocks, threads>>>(out, dstOut);
        dstOut = !dstOut;

    }
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
    cudaEventRecord(d->stop, 0);
    cudaEventSynchronize(d->stop);
    cudaMemcpy(bitmap->get_ptr(),d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Average Time per frame : %3.1f ms \n", d->totalTime/d->frames);
    //std::cin.get(); // check
}

void anim_exit(DataBlock *d)
{
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);

    cudaFree(d->dev_constSrc);
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(dev_temp); // swap 交换变量

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
    cudaMalloc((void **)&dev_temp, bitmap.image_size()); // 用于swap函数交换

    cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, bitmap.image_size());
    cudaBindTexture(NULL, texIn, data.dev_inSrc, bitmap.image_size());
    cudaBindTexture(NULL, texOut, data.dev_outSrc, bitmap.image_size());

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