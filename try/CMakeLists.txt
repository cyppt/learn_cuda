cmake_minimum_required(VERSION 3.0)
 
project(helloC++)
 
find_package(CUDA REQUIRED)
 
include_directories(${PROJECT_SOURCE_DIR}/include) # 通过源码根目录来定位include2
cuda_add_executable (helloworld helloworld.cpp addition.cu)