cmake_minimum_required(VERSION 3.0)
 
project(ch5_cuda)
 
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED) 

#该作用是指定cmake使用的OpenGL 版本。(当安装了nvidia驱动的opengl时候)
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()


include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include) # 通过源码根目录来定位include2

cuda_add_executable (example1_ch5 example1_ch5.cpp add1.cu)
cuda_add_executable (example2_ch5 example2_ch5.cpp anim_bitmap.cu)
target_link_libraries(example2_ch5 ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})