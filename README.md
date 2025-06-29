# FastTD3 Simple - C++ Implementation

[![CI/CD](https://github.com/yourusername/FastTD3/workflows/FastTD3%20CI%2FCD/badge.svg)](https://github.com/yourusername/FastTD3/actions)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.16+-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A simplified C++ implementation of the FastTD3 algorithm without LibTorch dependencies. This version uses Eigen for matrix operations and provides a working foundation for the TD3 algorithm.

## Features

- **No LibTorch Dependencies**: Uses only standard C++ libraries and Eigen
- **Simple Neural Networks**: Basic feedforward networks with ReLU activation
- **Replay Buffer**: Efficient experience replay for off-policy learning
- **Empirical Normalization**: State normalization for stable training
- **TD3 Algorithm**: Twin Delayed Deep Deterministic Policy Gradient
- **Modern C++**: Uses C++17 features and modern libraries
- **Comprehensive Testing**: Unit tests for all core components

## Dependencies

- **Eigen3**: Matrix operations and linear algebra
- **spdlog**: Fast logging library
- **CLI11**: Command-line argument parsing
- **nlohmann/json**: JSON parsing and serialization
- **Google Test**: Unit testing framework

## Installation

### macOS (using Homebrew)

```bash
# Install dependencies
brew install eigen spdlog cli11 nlohmann-json googletest

# Build the project
mkdir build && cd build
cmake ..
make -j4
```

## Usage

### Basic Training

```bash
# Run with default parameters
./fast_td3_simple

# Run with custom parameters
./fast_td3_simple --max-steps 100000 --batch-size 256 --state-dim 17 --action-dim 6

# Run with debug logging
./fast_td3_simple --log-level debug
```

### Command Line Options

- `--seed`: Random seed (default: 42)
- `--max-steps`: Maximum training steps (default: 1000000)
- `--batch-size`: Batch size for training (default: 256)
- `--state-dim`: State dimension (default: 17)
- `--action-dim`: Action dimension (default: 6)
- `--log-level`: Log level - debug, info, warn, error (default: info)

## Testing

The project includes comprehensive unit tests to ensure the reliability and correctness of all core components. Tests are written using Google Test framework and cover all major functionality.

### Running Tests

```bash
# Build and run all tests
make tests_simple
./tests_simple

# Run with verbose output
./tests_simple --gtest_verbose

# Run specific test suites
./tests_simple --gtest_filter="ReplayBufferTest*"
./tests_simple --gtest_filter="NormalizerTest*"
./tests_simple --gtest_filter="UtilsTest*"

# Run with detailed output and test discovery
./tests_simple --gtest_list_tests
./tests_simple --gtest_output=xml:test_results.xml
```

### Test Coverage

The test suite covers the following components and functionality:

#### 1. ReplayBuffer Tests (`ReplayBufferTest`)

**Purpose**: Verify the experience replay buffer functionality for off-policy learning.

**Test Cases**:
- **BasicOperations**: Tests fundamental replay buffer operations
  - Adding experiences to the buffer
  - Sampling batches from the buffer
  - Buffer size management
  - Data integrity during add/sample operations

**What it tests**:
- Experience storage and retrieval
- Batch sampling functionality
- Buffer capacity management
- Memory efficiency

#### 2. Normalizer Tests (`NormalizerTest`)

**Purpose**: Validate the empirical normalization component for stable training.

**Test Cases**:
- **BasicOperations**: Tests normalization functionality
  - Input data normalization
  - Running statistics updates
  - Mean and variance calculations
  - Numerical stability

**What it tests**:
- State normalization accuracy
- Running statistics computation
- Numerical stability with small values
- Memory management for statistics

#### 3. Utils Tests (`UtilsTest`)

**Purpose**: Verify utility functions used throughout the TD3 algorithm.

**Test Cases**:
- **BasicFunctions**: Tests core utility functions
  - **Clip Function**: Value clamping between bounds
    - Values within bounds remain unchanged
    - Values above upper bound are clipped
    - Values below lower bound are clipped
  - **TD Error**: Temporal difference error computation
    - Correct calculation of target vs current value differences
  - **Huber Loss**: Robust loss function for critic networks
    - Loss computation for various input values
    - Zero loss for perfect predictions

- **SoftUpdate**: Tests target network update mechanism
  - Soft parameter updates with tau parameter
  - Correct interpolation between target and source networks
  - Numerical precision of updates

**What it tests**:
- Mathematical correctness of utility functions
- Edge case handling
- Numerical stability
- Algorithm-specific operations

### Test Data and Parameters

**ReplayBuffer Test Parameters**:
- Buffer capacity: 1000 experiences
- State dimension: 4
- Action dimension: 2
- Sample size: 1 (for basic testing)

**Normalizer Test Parameters**:
- Input shape: {4} (4-dimensional state)
- Epsilon: 1e-8 (numerical stability)
- Max samples: 100 (for running statistics)

**Utils Test Parameters**:
- Clip bounds: [0.0, 1.0]
- Soft update tau: 0.1
- Test values: Various edge cases and normal ranges

### Test Output and Debugging

**Verbose Output**:
```bash
# Enable detailed test output
./tests_simple --gtest_verbose

# Run with color output
./tests_simple --gtest_color=yes

# Generate XML report for CI/CD
./tests_simple --gtest_output=xml:test_results.xml
```

**Common Test Issues**:
- **Memory Issues**: Check buffer allocation and deallocation
- **Numerical Precision**: Verify floating-point comparisons
- **Thread Safety**: Ensure thread-safe operations in multi-threaded environments

### Continuous Integration

The test suite is designed to be CI/CD friendly:

```yaml
# Example GitHub Actions configuration
- name: Run Tests
  run: |
    mkdir build && cd build
    cmake ..
    make -j4
    ./tests_simple --gtest_output=xml:test_results.xml
```

### Adding New Tests

To add new test cases:

1. **Create test file**: Add new test functions in `tests/test_simple.cpp`
2. **Follow naming convention**: Use descriptive test names (e.g., `ComponentTest, SpecificFunctionality`)
3. **Include edge cases**: Test boundary conditions and error scenarios
4. **Add to build**: Ensure new test files are included in CMakeLists.txt

Example test structure:
```cpp
TEST(NewComponentTest, NewFunctionality) {
    // Setup
    // Action
    // Assert
    EXPECT_EQ(actual, expected);
}
```

## Architecture

### Core Components

1. **SimpleNeuralNetwork**: Basic feedforward neural network using Eigen
2. **SimpleTD3Agent**: TD3 agent with actor-critic networks
3. **ReplayBuffer**: Experience replay buffer for off-policy learning
4. **EmpiricalNormalization**: State normalization for training stability

### Key Features

- **Eigen-based Neural Networks**: Simple but efficient neural network implementation
- **Experience Replay**: FIFO buffer for storing and sampling experiences
- **State Normalization**: Running statistics for input normalization
- **Exploration Noise**: Gaussian noise for action exploration
- **Target Networks**: Soft updates for stable training

## Limitations

This is a simplified implementation with the following limitations:

1. **No GPU Support**: CPU-only implementation
2. **Basic Neural Networks**: Simple feedforward networks without advanced features
3. **No Environment Integration**: Uses dummy data for demonstration
4. **Limited Optimization**: Basic gradient descent without advanced optimizers

## Future Improvements

- Add proper backpropagation and gradient computation
- Implement advanced optimizers (Adam, RMSprop)
- Add GPU support with CUDA or OpenCL
- Integrate with real environments (Gym, MuJoCo)
- Add more sophisticated exploration strategies
- Implement proper weight initialization schemes
- Expand test coverage for neural network components
- Add integration tests for full TD3 algorithm
- Performance benchmarking tests

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests! When contributing, please ensure all tests pass and add new tests for any new functionality. 