cmake_minimum_required(VERSION 3.0)
 
project(ch4_cuda)
 
find_package(CUDA REQUIRED)
 
include_directories(${PROJECT_SOURCE_DIR}/include) # 通过源码根目录来定位include2
cuda_add_executable (example1 example1.cpp add.cu)
cuda_add_executable (example2 example2.cpp bitmap.cu)