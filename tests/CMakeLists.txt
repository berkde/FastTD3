# Enable testing
enable_testing()

# Find required packages for testing
find_package(GTest REQUIRED)
find_package(nlohmann_json QUIET)

# Simple version tests
set(SIMPLE_TEST_SOURCES
    simple/test_simple.cpp
    ../src/simple/replay_buffer_simple.cpp
    ../src/simple/normalizers_simple.cpp
    ../src/simple/utils_simple.cpp
)

add_executable(tests_simple ${SIMPLE_TEST_SOURCES})

target_link_libraries(tests_simple
    GTest::gtest
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
)

# Add nlohmann_json if found (optional for simple tests)
if(nlohmann_json_FOUND)
    target_link_libraries(tests_simple nlohmann_json::nlohmann_json)
endif()

add_test(NAME SimpleTests COMMAND tests_simple)

# PyTorch version tests (if available)
if(Torch_FOUND)
    set(PYTORCH_TEST_SOURCES
        pytorch/test_basic.cpp
    )
    
    add_executable(tests_pytorch ${PYTORCH_TEST_SOURCES})
    
    target_link_libraries(tests_pytorch
        GTest::gtest
        GTest::gtest_main
        ${TORCH_LIBRARIES}
        spdlog::spdlog
    )
    
    # Add nlohmann_json if found (optional for PyTorch tests)
    if(nlohmann_json_FOUND)
        target_link_libraries(tests_pytorch nlohmann_json::nlohmann_json)
    endif()
    
    add_test(NAME PyTorchTests COMMAND tests_pytorch)
endif()

# MuJoCo integration tests (if available)
if(Torch_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/src/pytorch/mujoco_env.cpp")
    # Add custom test for MuJoCo integration script
    add_test(
        NAME MuJoCoIntegrationTest
        COMMAND ${CMAKE_SOURCE_DIR}/tests/scripts/test_mujoco_integration.sh
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Set environment variables for headless mode in CI
    set_tests_properties(MuJoCoIntegrationTest PROPERTIES
        ENVIRONMENT "MUJOCO_GL=osmesa"
    )
endif() 