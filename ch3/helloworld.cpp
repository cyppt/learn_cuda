/*
  * helloworld.cpp
  *
  */
 
 #include <iostream>
 #include <addition.h>
 
 int main(int argc, char** argv)
 {
    int a=1,b=2,c;

    if(addition(a,b,&c)) //cuda 函数
        std::cout<<"c="<<c<<std::endl;
    else
        std::cout<<"Addition failed!"<<std::endl;
    return 0;
 }