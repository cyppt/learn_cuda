 #ifndef INCLUDES_ADD1_H_
 #define INCLUDES_ADD1_H_

 #define N 33*1024
 
 /*check if the compiler is of C++*/
 #ifdef __cplusplus
 //修饰符extern "C"是CUDA和C++混合编程时必须的。
 extern "C" bool addKernel(int *a, int *b, int *c);
 
 #endif
 
 
#endif /* INCLUDES_ADD1_H_ */