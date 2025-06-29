cmake_minimum_required(VERSION 3.16)
project(FastTD3 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

# Optional CLI11 for command line parsing
find_package(CLI11 QUIET)
if(CLI11_FOUND)
    message(STATUS "CLI11 found, will use CLI11 for command line parsing")
    add_definitions(-DCLI11_FOUND)
else()
    message(STATUS "CLI11 not found, will use fallback command line parsing")
endif()

# Optional PyTorch
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "PyTorch found: ${Torch_VERSION}")
    message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
    message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
else()
    message(STATUS "PyTorch not found, PyTorch version will not be built")
endif()

# Include directories
include_directories(include/core include/pytorch include/simple)

# Add PyTorch include directories if found
if(Torch_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests) 