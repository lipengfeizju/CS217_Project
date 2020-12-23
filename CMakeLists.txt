cmake_minimum_required(VERSION 3.15)
project(icp  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Debug")

include_directories(.)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -G")

add_library(data_io SHARED src/data_io.cpp)
# set_source_files_properties(src/icp.cpp PROPERTIES LANGUAGE CUDA)  # For Debug only
add_library(icp STATIC src/icp.cpp)
# set_property(TARGET icp PROPERTY CUDA_ARCHITECTURES 75)   # For Debug only

add_library(icp_cuda STATIC src/icp_cuda.cu)
set_property(TARGET icp_cuda PROPERTY CUDA_ARCHITECTURES 75)       

add_executable(registration_demo  src/main_demo.cpp)
target_link_libraries(registration_demo
                    icp data_io icp_cuda -lcublas -lcusolver)


add_executable(registration  src/main_registration.cpp)
target_link_libraries(registration
                    icp data_io icp_cuda -lcublas -lcusolver)

