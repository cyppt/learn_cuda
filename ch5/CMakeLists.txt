cmake_minimum_required(VERSION 3.0)
 
project(ch5_cuda)
 
find_package(CUDA REQUIRED)
 
include_directories(${PROJECT_SOURCE_DIR}/include) # 通过源码根目录来定位include2
cuda_add_executable (example1_ch5 example1_ch5.cpp add1.cu)