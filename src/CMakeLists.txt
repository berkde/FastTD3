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
    
    target_link_libraries(fast_td3_pytorch
        ${TORCH_LIBRARIES}
        spdlog::spdlog
    )
    
    if(CLI11_FOUND)
        target_link_libraries(fast_td3_pytorch CLI11::CLI11)
        target_compile_definitions(fast_td3_pytorch PRIVATE CLI11_FOUND)
    endif()
    
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