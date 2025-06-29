cmake_minimum_required(VERSION 3.16)
project(FastTD3_Simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package, trying alternative names...")
    # Try alternative names for Eigen3
    find_package(Eigen3 QUIET NAMES Eigen3 eigen3)
endif()

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package, trying manual detection...")
    # Try to find Eigen3 in common locations
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense
        PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/homebrew/include/eigen3
        /usr/include
        /usr/local/include
    )
    
    if(EIGEN3_INCLUDE_DIR)
        set(Eigen3_FOUND TRUE)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
        message(STATUS "Eigen3 found manually: ${EIGEN3_INCLUDE_DIR}")
    else()
        message(STATUS "Eigen3 not found manually, trying to continue anyway...")
        # Try to continue with a default path
        set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
        message(STATUS "Using default Eigen3 path: ${EIGEN3_INCLUDE_DIR}")
        message(WARNING "Eigen3 not found. Please install Eigen3:")
        message(WARNING "  Ubuntu/Debian: sudo apt-get install libeigen3-dev")
        message(WARNING "  macOS: brew install eigen")
        message(WARNING "  Or download from: https://eigen.tuxfamily.org/")
    endif()
else()
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found via find_package, trying manual detection...")
    # Try to find spdlog in common locations
    find_path(SPDLOG_INCLUDE_DIR NAMES spdlog/spdlog.h
        PATHS
        /usr/include
        /usr/local/include
        /opt/homebrew/include
    )
    
    find_library(SPDLOG_LIBRARY NAMES spdlog
        PATHS
        /usr/lib
        /usr/local/lib
        /opt/homebrew/lib
    )
    
    if(SPDLOG_INCLUDE_DIR AND SPDLOG_LIBRARY)
        set(spdlog_FOUND TRUE)
        set(SPDLOG_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR})
        set(SPDLOG_LIBRARIES ${SPDLOG_LIBRARY})
        message(STATUS "spdlog found manually: ${SPDLOG_INCLUDE_DIR}")
    else()
        message(STATUS "spdlog not found manually, trying to continue anyway...")
        # Try to continue with a default path
        set(SPDLOG_INCLUDE_DIR "/usr/include")
        set(SPDLOG_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR})
        set(SPDLOG_LIBRARIES "spdlog")
        message(STATUS "Using default spdlog path: ${SPDLOG_INCLUDE_DIR}")
        message(WARNING "spdlog not found. Please install spdlog:")
        message(WARNING "  Ubuntu/Debian: sudo apt-get install libspdlog-dev")
        message(WARNING "  macOS: brew install spdlog")
    endif()
else()
    message(STATUS "spdlog found: ${spdlog_VERSION}")
endif()

find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

# Optional packages
find_package(CLI11 QUIET)
if(NOT CLI11_FOUND)
    message(STATUS "CLI11 not found. Command-line argument parsing will be limited.")
    add_definitions(-DNO_CLI11)
else()
    message(STATUS "CLI11 found: ${CLI11_VERSION}")
    add_definitions(-DUSE_CLI11)
endif()

# PyTorch (LibTorch) configuration
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    # Try to find PyTorch in system paths
    find_package(Torch QUIET)
endif()

if(NOT Torch_FOUND)
    # Try to find PyTorch via Python
    find_package(Python3 QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_CMAKE_PATH)
            list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
            find_package(Torch QUIET)
        endif()
    endif()
endif()

if(Torch_FOUND)
    message(STATUS "PyTorch found: ${Torch_VERSION}")
    message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
    message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
    add_definitions(-DUSE_PYTORCH)
else()
    message(WARNING "PyTorch not found. Building without PyTorch support.")
    message(STATUS "To enable PyTorch support:")
    message(STATUS "  1. Download LibTorch from https://pytorch.org/get-started/locally/")
    message(STATUS "  2. Extract to third_party/libtorch/")
    message(STATUS "  3. Or install PyTorch via pip and set CMAKE_PREFIX_PATH")
    add_definitions(-DNO_PYTORCH)
endif()

# Include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

if(Torch_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

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
    nlohmann_json::nlohmann_json
)

if(spdlog_FOUND)
    target_link_libraries(fast_td3_simple spdlog::spdlog)
else()
    target_link_libraries(fast_td3_simple ${SPDLOG_LIBRARIES})
    target_include_directories(fast_td3_simple PRIVATE ${SPDLOG_INCLUDE_DIRS})
endif()

if(CLI11_FOUND)
    target_link_libraries(fast_td3_simple CLI11::CLI11)
endif()

if(Torch_FOUND)
    target_link_libraries(fast_td3_simple ${TORCH_LIBRARIES})
    set_property(TARGET fast_td3_simple PROPERTY CXX_STANDARD 17)
    
    # Set RPATH for PyTorch libraries
    if(APPLE)
        set_target_properties(fast_td3_simple PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@executable_path;@executable_path/../lib"
        )
    elseif(UNIX AND NOT APPLE)
        set_target_properties(fast_td3_simple PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        )
    endif()
endif()

# Tests
enable_testing()
add_executable(tests_simple tests/test_simple.cpp src/replay_buffer_simple.cpp src/normalizers_simple.cpp src/utils_simple.cpp)
target_link_libraries(tests_simple GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

if(spdlog_FOUND)
    target_link_libraries(tests_simple spdlog::spdlog)
else()
    target_link_libraries(tests_simple ${SPDLOG_LIBRARIES})
    target_include_directories(tests_simple PRIVATE ${SPDLOG_INCLUDE_DIRS})
endif()

add_test(NAME SimpleTests COMMAND tests_simple)

# PyTorch version (if available)
if(Torch_FOUND)
    add_executable(fast_td3_pytorch src/main.cpp src/networks.cpp src/replay_buffer.cpp src/normalizers.cpp src/utils.cpp src/config.cpp)
    target_link_libraries(fast_td3_pytorch ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
    
    if(spdlog_FOUND)
        target_link_libraries(fast_td3_pytorch spdlog::spdlog)
    else()
        target_link_libraries(fast_td3_pytorch ${SPDLOG_LIBRARIES})
        target_include_directories(fast_td3_pytorch PRIVATE ${SPDLOG_INCLUDE_DIRS})
    endif()
    
    if(CLI11_FOUND)
        target_link_libraries(fast_td3_pytorch CLI11::CLI11)
    endif()
    
    set_property(TARGET fast_td3_pytorch PROPERTY CXX_STANDARD 17)
    
    # Set RPATH for PyTorch libraries
    if(APPLE)
        set_target_properties(fast_td3_pytorch PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@executable_path;@executable_path/../lib"
        )
    elseif(UNIX AND NOT APPLE)
        set_target_properties(fast_td3_pytorch PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        )
    endif()
endif() 