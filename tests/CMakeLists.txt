# Test configuration
find_package(GTest REQUIRED)

# Create test executable
add_executable(fast_td3_tests
    test_basic.cpp
)

# Link libraries
target_link_libraries(fast_td3_tests
    fast_td3_lib
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# Set properties
set_target_properties(fast_td3_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add test
add_test(NAME FastTD3Tests COMMAND fast_td3_tests)

# Enable CTest
enable_testing() 