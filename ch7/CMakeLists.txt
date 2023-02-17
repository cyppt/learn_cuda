cmake_minimum_required(VERSION 3.0)

project(ch7_cuda)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

#该作用是指定cmake使用的OpenGL 版本。(当安装了nvidia驱动的opengl时候)
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable (example1_ch7 example1_ch7.cpp heat_ch7.cu)

target_link_libraries(example1_ch7 ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
