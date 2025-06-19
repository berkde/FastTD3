# Create library for FastTD3 components
add_library(fast_td3_lib STATIC
    networks.cpp
    replay_buffer.cpp
    normalizers.cpp
    utils.cpp
    config.cpp
)

# Set properties for the library
set_target_properties(fast_td3_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link libraries for the static library
target_link_libraries(fast_td3_lib
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
) 