# FastTD3 Simple - C++ Implementation

A simplified C++ implementation of the FastTD3 algorithm without LibTorch dependencies. This version uses Eigen for matrix operations and provides a working foundation for the TD3 algorithm.

## Features

- **No LibTorch Dependencies**: Uses only standard C++ libraries and Eigen
- **Simple Neural Networks**: Basic feedforward networks with ReLU activation
- **Replay Buffer**: Efficient experience replay for off-policy learning
- **Empirical Normalization**: State normalization for stable training
- **TD3 Algorithm**: Twin Delayed Deep Deterministic Policy Gradient
- **Modern C++**: Uses C++17 features and modern libraries

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

### Running Tests

```bash
# Run all tests
./tests_simple

# Run with verbose output
./tests_simple --gtest_verbose
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

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests! 