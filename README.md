# FastTD3 - C++ Implementation

This is a C++ implementation of the FastTD3 algorithm, a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm optimized for complex humanoid control tasks.

## Features

- **High Performance**: Leverages LibTorch (PyTorch C++ API) for efficient tensor operations
- **Distributional Q-Learning**: Implements distributional Q-functions for better value estimation
- **Modern C++**: Uses C++17 features and modern libraries
- **Flexible Configuration**: Command-line argument parsing with environment-specific defaults
- **Comprehensive Logging**: Structured logging with spdlog
- **Memory Efficient**: Optimized replay buffer with support for asymmetric observations
- **GPU Support**: Full CUDA and MPS support for accelerated training

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Compiler**: GCC 7+, Clang 6+, or MSVC 2019+
- **CMake**: 3.18 or higher
- **CUDA**: 11.0+ (optional, for GPU acceleration)

### Dependencies

The following libraries are required and will be automatically downloaded by CMake:

- **LibTorch**: PyTorch C++ API
- **Eigen3**: Linear algebra library
- **spdlog**: Fast logging library
- **CLI11**: Command-line argument parsing
- **nlohmann/json**: JSON parsing and serialization

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/berkde/FastTD3
cd FastTD3
```

### 2. Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev nlohmann-json3-dev
```

#### macOS
```bash
brew install cmake eigen nlohmann-json
```

#### Windows
Install Visual Studio 2019 or later with C++ development tools.

### 3. Build the Project

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 4. Install LibTorch

Download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/) and extract it to a directory. Then set the `CMAKE_PREFIX_PATH`:

```bash
# Linux/macOS
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH

# Windows
set CMAKE_PREFIX_PATH=C:\path\to\libtorch;%CMAKE_PREFIX_PATH%
```

## Usage

### Basic Training

```bash
./fast_td3 --env_name h1hand-stand-v0 --seed 1 --total_timesteps 100000
```

### Advanced Configuration

```bash
./fast_td3 \
    --env_name h1hand-reach-v0 \
    --agent fasttd3 \
    --seed 42 \
    --num_envs 128 \
    --total_timesteps 150000 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --batch_size 32768 \
    --buffer_size 51200 \
    --gamma 0.99 \
    --tau 0.1 \
    --num_updates 2 \
    --use_cdq \
    --obs_normalization \
    --cuda \
    --device_rank 0
```

### Environment-Specific Configurations

The C++ implementation automatically applies environment-specific hyperparameters:

#### HumanoidBench Environments
```bash
./fast_td3 --env_name h1hand-balance-hard-v0 --seed 1
```

#### MuJoCo Playground Environments
```bash
./fast_td3 --env_name T1JoystickFlatTerrain --seed 1
```

#### IsaacLab Environments
```bash
./fast_td3 --env_name Isaac-Velocity-Flat-G1-v0 --seed 1
```

## Configuration Options

### Environment Settings
- `--env_name`: Environment name (default: h1hand-stand-v0)
- `--agent`: Agent type: fasttd3, fasttd3_simbav2 (default: fasttd3)
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA acceleration (default: true)
- `--device_rank`: GPU device rank (default: 0)

### Training Parameters
- `--num_envs`: Number of parallel environments (default: 128)
- `--total_timesteps`: Total training timesteps (default: 150000)
- `--critic_learning_rate`: Critic learning rate (default: 3e-4)
- `--actor_learning_rate`: Actor learning rate (default: 3e-4)
- `--batch_size`: Training batch size (default: 32768)
- `--buffer_size`: Replay buffer size (default: 51200)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.1)
- `--num_updates`: Updates per environment step (default: 2)

### Network Architecture
- `--critic_hidden_dim`: Critic hidden dimension (default: 1024)
- `--actor_hidden_dim`: Actor hidden dimension (default: 512)
- `--num_atoms`: Number of distributional atoms (default: 101)
- `--v_min`: Minimum value support (default: -250.0)
- `--v_max`: Maximum value support (default: 250.0)

### Training Features
- `--use_cdq`: Enable clipped double Q-learning (default: true)
- `--obs_normalization`: Enable observation normalization (default: true)
- `--reward_normalization`: Enable reward normalization (default: false)
- `--amp`: Enable automatic mixed precision (default: true)
- `--compile`: Use torch.compile optimization (default: true)

## Project Structure

```
FastTD3/
├── CMakeLists.txt              # Main CMake configuration
├── src/
│   ├── CMakeLists.txt          # Source CMake configuration
│   ├── main.cpp                # Main training loop
│   ├── networks.cpp            # Neural network implementations
│   ├── replay_buffer.cpp       # Experience replay buffer
│   ├── normalizers.cpp         # Observation and reward normalizers
│   ├── config.cpp              # Configuration management
│   └── utils.cpp               # Utility functions
├── include/
│   ├── networks.hpp            # Network class declarations
│   ├── replay_buffer.hpp       # Replay buffer interface
│   ├── normalizers.hpp         # Normalizer interfaces
│   ├── config.hpp              # Configuration structures
│   └── utils.hpp               # Utility function declarations
├── examples/                   # Example scripts and configurations
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Key Components

### Neural Networks
- **DistributionalQNetwork**: Implements distributional Q-functions with categorical distributions
- **Critic**: Twin Q-networks with distributional outputs
- **Actor**: Policy network with exploration noise

### Replay Buffer
- **SimpleReplayBuffer**: Efficient experience storage with support for:
  - N-step returns
  - Asymmetric observations
  - Memory-efficient privileged observation storage

### Normalizers
- **EmpiricalNormalization**: Online observation normalization
- **RewardNormalizer**: Reward scaling and normalization

### Configuration
- **ConfigManager**: Handles command-line parsing and environment-specific defaults
- **JSON Support**: Save/load configurations in JSON format

## Performance Optimizations

1. **GPU Memory Management**: Efficient tensor operations with minimal CPU-GPU transfers
2. **Batch Processing**: Vectorized operations for parallel environments
3. **Memory Pooling**: Reuse tensor allocations to reduce memory fragmentation
4. **Mixed Precision**: Automatic mixed precision training for faster computation
5. **Compilation**: TorchScript compilation for optimized inference

## Logging and Monitoring

The implementation uses structured logging with different levels:

```cpp
spdlog::info("Training started with {} environments", num_envs);
spdlog::warn("High memory usage detected");
spdlog::error("Failed to load checkpoint: {}", error_msg);
spdlog::debug("Detailed training statistics");
```

## Troubleshooting

### Common Issues

1. **CMake cannot find LibTorch**
   - Ensure `CMAKE_PREFIX_PATH` is set correctly
   - Verify LibTorch installation path

2. **CUDA out of memory**
   - Reduce `batch_size` or `num_envs`
   - Use gradient accumulation for larger effective batch sizes

3. **Slow training**
   - Enable `--amp` for mixed precision
   - Use `--compile` for torch optimization
   - Ensure GPU is being utilized

4. **Compilation errors**
   - Update to C++17 compatible compiler
   - Check all dependencies are installed

### Performance Tuning

- **Batch Size**: Larger batch sizes generally improve training stability
- **Learning Rate**: Start with default values and adjust based on convergence
- **Network Size**: Larger networks may improve performance but increase training time
- **Update Frequency**: Higher update frequency can improve sample efficiency


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This C++ implementation is based on the original Python FastTD3 implementation. Thanks to the original authors for their research and the PyTorch team for the excellent C++ API. 