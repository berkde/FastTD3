cmake_minimum_required(VERSION 3.16)
project(FastTD3_Simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    src/main_simple.cpp
    src/replay_buffer_simple.cpp
    src/normalizers_simple.cpp
    src/utils_simple.cpp
)

# Create executable
add_executable(fast_td3_simple ${SOURCES})

# Link libraries
target_link_libraries(fast_td3_simple
    Eigen3::Eigen
    spdlog::spdlog
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)

# Tests
enable_testing()
add_executable(tests_simple tests/test_simple.cpp src/replay_buffer_simple.cpp src/normalizers_simple.cpp src/utils_simple.cpp)
target_link_libraries(tests_simple GTest::gtest GTest::gtest_main spdlog::spdlog nlohmann_json::nlohmann_json)
add_test(NAME SimpleTests COMMAND tests_simple) 