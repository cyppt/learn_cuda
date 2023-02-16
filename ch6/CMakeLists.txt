cmake_minimum_required(VERSION 3.0)

project(ch6_cuda)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

#该作用是指定cmake使用的OpenGL 版本。(当安装了nvidia驱动的opengl时候)
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE GLVND)
endif()

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable (example1_ch6 example1_ch6.cpp ray_trace.cu)
cuda_add_executable (example2_ch6 example2_ch6.cpp ray_trace_2.cu)


target_link_libraries(example1_ch6 ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(example2_ch6 ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
