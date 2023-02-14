/*
  * addition.h
  *
  */
 
 #ifndef INCLUDES_ADDITION_H_
 #define INCLUDES_ADDITION_H_
 
 /*check if the compiler is of C++*/
 #ifdef __cplusplus
 //修饰符extern "C"是CUDA和C++混合编程时必须的。
 extern "C" bool addition(int a, int b, int *c);
 
 #endif
 
 
#endif /* INCLUDES_ADDITION_H_ */
