cmake_minimum_required(VERSION 3.15)
project(icp  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_BUILD_TYPE "Debug")

include_directories(.)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -G")

add_library(data_io SHARED src/data_io.cpp)
# set_source_files_properties(src/icp.cpp PROPERTIES LANGUAGE CUDA)  # For Debug only
add_library(icp STATIC src/icp.cpp)
# set_property(TARGET icp PROPERTY CUDA_ARCHITECTURES 75)   # For Debug only

add_library(icp_cuda STATIC src/icp.cu)
set_property(TARGET icp_cuda PROPERTY CUDA_ARCHITECTURES 75)       


add_library(emicp_cuda STATIC src/emicp.cu)
set_property(TARGET emicp_cuda PROPERTY CUDA_ARCHITECTURES 75)      


# add_library(emicp_orig STATIC src/emicp_orig.cu) # TEST
# set_property(TARGET emicp_orig PROPERTY CUDA_ARCHITECTURES 75)  


# set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)  # For Debug only
add_executable(registration  src/main.cpp)
# set_property(TARGET registration PROPERTY CUDA_ARCHITECTURES 75)    # For Debug only

target_link_libraries(registration
                    icp data_io icp_cuda emicp_cuda -lcublas -lcusolver)
