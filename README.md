# FastTD3 - C++ Implementation

[![CI/CD](https://github.com/berkde/FastTD3/workflows/FastTD3%20CI%2FCD/badge.svg)](https://github.com/berkde/FastTD3/actions)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.16+-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A C++ implementation of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm with both simple Eigen-based and full PyTorch versions. The project provides a complete, production-ready implementation with comprehensive testing and CI/CD support. **Now with MuJoCo integration for humanoid and robotics training!**

## Features

- **Dual Implementation**: Both simple Eigen-based and full PyTorch versions
- **Simple Mode**: No LibTorch dependencies, uses Eigen for matrix operations
- **PyTorch Mode**: Full neural network capabilities with GPU support
- **MuJoCo Integration**: Real physics simulation with humanoid and robotics environments
- **Humanoid Training**: Pre-configured training scripts for humanoid robots
- **Replay Buffer**: Efficient experience replay for off-policy learning
- **Empirical Normalization**: State normalization for stable training
- **TD3 Algorithm**: Complete Twin Delayed Deep Deterministic Policy Gradient implementation
- **Modern C++**: Uses C++17 features and modern libraries
- **Comprehensive Testing**: Unit tests for all core components
- **CI/CD Pipeline**: Automated building and testing on multiple platforms

## Dependencies

- **Eigen3**: Matrix operations and linear algebra
- **spdlog**: Fast logging library
- **CLI11**: Command-line argument parsing
- **nlohmann/json**: JSON parsing and serialization
- **Google Test**: Unit testing framework
- **PyTorch**: Neural network operations (for PyTorch mode)
- **MuJoCo**: Physics simulation (for MuJoCo mode)
- **pybind11**: Python-C++ bindings (for MuJoCo integration)

## Installation

### Prerequisites

The project supports three modes:
1. **Simple Mode**: Uses only Eigen for matrix operations (no PyTorch required)
2. **PyTorch Mode**: Full PyTorch integration with neural networks
3. **MuJoCo Mode**: PyTorch + MuJoCo integration for real physics simulation

### Dependencies

#### Required (All Modes)
- **Eigen3**: Matrix operations and linear algebra
- **spdlog**: Fast logging library
- **CLI11**: Command-line argument parsing
- **nlohmann/json**: JSON parsing and serialization
- **Google Test**: Unit testing framework

#### Optional (PyTorch Mode)
- **PyTorch (LibTorch)**: Neural network operations and GPU support

#### Optional (MuJoCo Mode)
- **Python 3.11**: Required for MuJoCo compatibility
- **PyTorch**: Neural network operations
- **MuJoCo**: Physics simulation engine
- **Gymnasium**: RL environment interface
- **pybind11**: Python-C++ bindings

### PyTorch Setup

#### Automatic Setup (Recommended)
```bash
# Run the setup script
./setup_pytorch.sh

# Follow the interactive prompts to install PyTorch
```

#### Manual Setup

**Option 1: Download LibTorch**
```bash
# Download from https://pytorch.org/get-started/locally/
# Extract to third_party/libtorch/
mkdir -p third_party
cd third_party
# Download and extract libtorch-*.zip
cd ..
```

**Option 2: Install via pip**
```bash
# Install PyTorch via pip
pip install torch torchvision torchaudio

# Build with Python PyTorch path
cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
```

### MuJoCo Setup

For humanoid and robotics training, install MuJoCo dependencies:

```bash
# Install Python 3.11 (recommended for compatibility)
brew install python@3.11

# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install MuJoCo dependencies
pip install torch gymnasium[mujoco] pybind11
```

### macOS (using Homebrew)

```bash
# Install dependencies
brew install eigen spdlog cli11 nlohmann-json googletest

# Build the project
mkdir build && cd build
cmake ..
make -j4
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  libeigen3-dev \
  libspdlog-dev \
  nlohmann-json3-dev \
  libgtest-dev \
  libgmock-dev

# Build the project
mkdir build && cd build
cmake ..
make -j4
```

### Windows

```bash
# Install dependencies via vcpkg or manually
# Build with Visual Studio or MinGW
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage

### Building

The project will automatically detect if PyTorch and MuJoCo are available and build accordingly:

```bash
mkdir build && cd build
cmake ..
make -j4
```

**Available Executables:**
- `fast_td3_simple`: Eigen-only version (always available)
- `fast_td3_pytorch`: Full PyTorch version (if PyTorch found)
- `fast_td3_mujoco`: MuJoCo integration version (if PyTorch + MuJoCo found)

### Running

#### Simple Mode (Eigen Only)
```bash
# Run with default parameters
./fast_td3_simple

# Run with custom parameters
./fast_td3_simple --max-steps 100000 --batch-size 256 --state-dim 17 --action-dim 6
```

#### PyTorch Mode (Full Features)
```bash
# Run with default parameters
./fast_td3_pytorch

# Run with GPU support (if available)
./fast_td3_pytorch --cuda --device-rank 0
```

#### MuJoCo Mode (Humanoid Training)
```bash
# Basic humanoid training
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 1000000

# Humanoid standup training
./fast_td3_mujoco --env_name HumanoidStandup-v5 --total_timesteps 2000000

# Walker2d training
./fast_td3_mujoco --env_name Walker2d-v5 --total_timesteps 500000

# High-performance training with all optimizations
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 2000000 --num_envs 256 --cuda
```

### Humanoid Training Scripts

Pre-configured training scripts are available for easy humanoid training:

```bash
# Run the humanoid training script
./tests/scripts/humanoid_training.sh

# Or run individual examples
./tests/scripts/humanoid_training.sh  # Runs all examples
```

The script includes:
- Basic Humanoid training (Humanoid-v5)
- Humanoid Standup training (HumanoidStandup-v5)
- Walker2d training (Walker2d-v5)
- High-performance training with all optimizations

### Available MuJoCo Environments

The MuJoCo integration supports the following environments:

**Humanoid Environments:**
- `Humanoid-v5`: Basic humanoid locomotion
- `HumanoidStandup-v5`: Humanoid standup task
- `Walker2d-v5`: 2D walker locomotion
- `HalfCheetah-v5`: Half-cheetah locomotion
- `Ant-v5`: Ant locomotion
- `Hopper-v5`: Hopper locomotion

**Legacy v2 Environments (fallback):**
- `Humanoid-v2`, `HumanoidStandup-v2`, `Walker2d-v2`, etc.

### Command Line Options

#### Simple Mode Options
- `--seed`: Random seed (default: 42)
- `--max-steps`: Maximum training steps (default: 1000000)
- `--batch-size`: Batch size for training (default: 256)
- `--state-dim`: State dimension (default: 17)
- `--action-dim`: Action dimension (default: 6)
- `--log-level`: Log level - debug, info, warn, error (default: info)

#### PyTorch Mode Options
- All simple mode options plus:
- `--cuda`: Enable CUDA support
- `--device-rank`: GPU device rank (default: 0)
- `--num-envs`: Number of parallel environments
- `--total-timesteps`: Total training timesteps
- `--learning-starts`: Steps before learning begins
- `--policy-frequency`: Policy update frequency
- `--num-updates`: Number of updates per step

#### MuJoCo Mode Options
- All PyTorch mode options plus:
- `--env_name`: MuJoCo environment name (e.g., "Humanoid-v5")
- `--obs_normalization`: Enable observation normalization
- `--reward_normalization`: Enable reward normalization
- `--use_cdq`: Enable CDQ (Continuous Distributional Q-learning)

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

### MuJoCo Integration Tests

```bash
# Test MuJoCo environment creation
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 10 --num_envs 1 --learning_starts 0

# Test headless mode (for CI/CD)
MUJOCO_GL=osmesa ./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 10
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
5. **MuJoCoEnvironment**: Python-C++ wrapper for MuJoCo environments
6. **Actor/Critic Networks**: PyTorch-based neural networks for TD3

### Key Features

- **Eigen-based Neural Networks**: Simple but efficient neural network implementation
- **Experience Replay**: FIFO buffer for storing and sampling experiences
- **State Normalization**: Running statistics for input normalization
- **Exploration Noise**: Gaussian noise for action exploration
- **Target Networks**: Soft updates for stable training
- **MuJoCo Integration**: Real physics simulation for robotics training
- **Humanoid Training**: Pre-configured scripts for humanoid robot training

## Limitations

This implementation has the following limitations:

1. **Simple Mode**: CPU-only implementation with basic neural networks
2. **MuJoCo Mode**: Requires Python 3.11+ for compatibility
3. **Environment Dependencies**: MuJoCo environments require specific Python packages
4. **GPU Support**: Limited to PyTorch and MuJoCo modes

## Future Improvements

- Add proper backpropagation and gradient computation for simple mode
- Implement advanced optimizers (Adam, RMSprop) for simple mode
- Add GPU support with CUDA or OpenCL for simple mode
- Expand MuJoCo environment support
- Add more sophisticated exploration strategies
- Implement proper weight initialization schemes
- Expand test coverage for neural network components
- Add integration tests for full TD3 algorithm
- Performance benchmarking tests
- Support for more robotics environments
- Real-time visualization and monitoring

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests! When contributing, please ensure all tests pass and add new tests for any new functionality.

## Documentation

- [MuJoCo Integration Guide](docs/MUJOCO_INTEGRATION.md) - Detailed guide for MuJoCo setup and usage 