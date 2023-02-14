 #ifndef INCLUDES_BITMAP_H_
 #define INCLUDES_BITMAP_H_

 #define DIM 50
 
 /*check if the compiler is of C++*/
 #ifdef __cplusplus
 //修饰符extern "C"是CUDA和C++混合编程时必须的。
 extern "C" void JuliaKernel(unsigned char (*ptr)[DIM]);
 
 
 #endif
 
 
#endif /* INCLUDES_BITMAP_H_ */