# Core components (shared)
set(CORE_SOURCES
    core/config.cpp
)

# Simple version (Eigen-based)
set(SIMPLE_SOURCES
    simple/main_simple.cpp
    simple/replay_buffer_simple.cpp
    simple/normalizers_simple.cpp
    simple/utils_simple.cpp
)

# PyTorch version (if available)
if(Torch_FOUND)
    set(PYTORCH_SOURCES
        pytorch/main.cpp
        pytorch/networks.cpp
        pytorch/replay_buffer.cpp
        pytorch/normalizers.cpp
        pytorch/utils.cpp
        pytorch/mujoco_env.cpp
    )
    
    set(MUJOCO_SOURCES
        pytorch/main_mujoco.cpp
        pytorch/networks.cpp
        pytorch/replay_buffer.cpp
        pytorch/normalizers.cpp
        pytorch/utils.cpp
        pytorch/mujoco_env.cpp
    )
endif()

# Build simple version
add_executable(fast_td3_simple ${CORE_SOURCES} ${SIMPLE_SOURCES})

target_link_libraries(fast_td3_simple
    Eigen3::Eigen
    spdlog::spdlog
)

if(CLI11_FOUND)
    target_link_libraries(fast_td3_simple CLI11::CLI11)
    target_compile_definitions(fast_td3_simple PRIVATE CLI11_FOUND)
endif()

# Build PyTorch version (if available)
if(Torch_FOUND)
    add_executable(fast_td3_pytorch ${CORE_SOURCES} ${PYTORCH_SOURCES})
    
    # Find pybind11
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        target_link_libraries(fast_td3_pytorch pybind11::embed)
        target_compile_definitions(fast_td3_pytorch PRIVATE PYBIND11_FOUND)
    else()
        # Try to find pybind11 via Python
        execute_process(
            COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 REQUIRED)
            target_link_libraries(fast_td3_pytorch pybind11::embed)
            target_compile_definitions(fast_td3_pytorch PRIVATE PYBIND11_FOUND)
        else()
            message(WARNING "pybind11 not found, MuJoCo environment will not be available")
        endif()
    endif()
    
    target_link_libraries(fast_td3_pytorch
        ${TORCH_LIBRARIES}
        spdlog::spdlog
    )
    
    if(CLI11_FOUND)
        target_link_libraries(fast_td3_pytorch CLI11::CLI11)
        target_compile_definitions(fast_td3_pytorch PRIVATE CLI11_FOUND)
    endif()
    
    # Build MuJoCo version
    add_executable(fast_td3_mujoco ${CORE_SOURCES} ${MUJOCO_SOURCES})
    
    # Find pybind11 for MuJoCo version
    if(pybind11_FOUND)
        target_link_libraries(fast_td3_mujoco pybind11::embed)
        target_compile_definitions(fast_td3_mujoco PRIVATE PYBIND11_FOUND)
    else()
        if(PYBIND11_CMAKE_DIR)
            target_link_libraries(fast_td3_mujoco pybind11::embed)
            target_compile_definitions(fast_td3_mujoco PRIVATE PYBIND11_FOUND)
        endif()
    endif()
    
    target_link_libraries(fast_td3_mujoco
        ${TORCH_LIBRARIES}
        spdlog::spdlog
    )
    
    if(CLI11_FOUND)
        target_link_libraries(fast_td3_mujoco CLI11::CLI11)
        target_compile_definitions(fast_td3_mujoco PRIVATE CLI11_FOUND)
    endif()
    
    # Set RPATH for PyTorch libraries
    if(APPLE)
        set_target_properties(fast_td3_pytorch PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@executable_path;@executable_path/../lib"
        )
        set_target_properties(fast_td3_mujoco PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@executable_path;@executable_path/../lib"
        )
    elseif(UNIX AND NOT APPLE)
        set_target_properties(fast_td3_pytorch PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        )
        set_target_properties(fast_td3_mujoco PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        )
    endif()
endif() 