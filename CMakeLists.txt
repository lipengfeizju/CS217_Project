cmake_minimum_required(VERSION 3.15)
project(icp)

set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_BUILD_TYPE "Debug")

include_directories(.)

add_library(data_io SHARED src/data_io.cpp)
add_library(icp STATIC src/icp.cpp)

add_executable(registration
                src/main.cpp)

target_link_libraries(registration
                    icp data_io)
