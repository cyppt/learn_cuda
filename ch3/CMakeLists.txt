cmake_minimum_required(VERSION 3.0)
 
project(helloC++)
 
find_package(CUDA REQUIRED)
 
include_directories(include)
cuda_add_executable (helloworld helloworld.cpp addition.cu)