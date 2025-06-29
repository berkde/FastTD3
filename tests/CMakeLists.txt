# Enable testing
enable_testing()

# Find required packages for testing
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

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
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    spdlog::spdlog
)

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
        nlohmann_json::nlohmann_json
        ${TORCH_LIBRARIES}
        spdlog::spdlog
    )
    
    add_test(NAME PyTorchTests COMMAND tests_pytorch)
endif() 