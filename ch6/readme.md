# 第6章 常量内存与事件
**常量内存与性能测试**
### 示例1、2知识点

* 光线追踪注释
> 因为相机固定z轴,面向原点,所以像素放出的光都是垂直xoy平面的,基于这点,可以计算是否与球碰撞
> 因为相机面向原点，且高度无限，我们将离相机的最近的点，转化为离原点最远的点，所以计算距离时候使dz + z
> 示例1没使用常量内存, 示例2使用了常量内存

### 示例3知识点 
* 测量GPU程序耗时 注意同步问题, 注意:只能用于核函数和设备内存复制代码,在用于和cpu混合的代码中,将得到不可靠的结果    
```
udaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start , 0);
    // to do in GPU
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
```    
