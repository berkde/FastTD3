cmake_minimum_required(VERSION 3.18)
project(FastTD3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 REQUIRED)
find_package(nlohmann_json REQUIRED)

# Set compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add subdirectories for source files
add_subdirectory(src)

# Create main executable
add_executable(fast_td3 src/main.cpp)

# Link libraries
target_link_libraries(fast_td3 
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    fast_td3_lib
)

# Set properties for the executable
set_target_properties(fast_td3 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Copy torch libraries to output directory (for deployment)
if(WIN32)
    add_custom_command(TARGET fast_td3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:torch_cpu>
        $<TARGET_FILE:torch_cuda>
        $<TARGET_FILE_DIR:fast_td3>
    )
endif()

# Add tests if building in test mode
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif() 