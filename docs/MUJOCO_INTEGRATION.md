# MuJoCo Integration Guide

This document provides detailed information about the MuJoCo integration in FastTD3, including setup, usage, and troubleshooting.

## Overview

The MuJoCo integration allows FastTD3 to train on real physics simulation environments using the MuJoCo physics engine. This enables training on complex robotics tasks like humanoid locomotion, manipulation, and other physics-based environments.

## Features

- **Real Physics Simulation**: Accurate physics using MuJoCo engine
- **Multiple Environments**: Support for various MuJoCo environments
- **Python-C++ Bridge**: Seamless integration between C++ TD3 and Python environments
- **GPU Support**: Full GPU acceleration for training
- **Headless Mode**: CI/CD compatible with no display requirements

## Supported Environments

### Humanoid Environments
- `Humanoid-v5`: Basic humanoid locomotion task
- `HumanoidStandup-v5`: Humanoid standup from lying position
- `Walker2d-v5`: 2D walker locomotion
- `HalfCheetah-v5`: Half-cheetah locomotion
- `Ant-v5`: Ant locomotion
- `Hopper-v5`: Hopper locomotion

### Legacy Support
- All v2 environments are supported as fallback
- Automatic version detection and fallback

## Installation

### Prerequisites

1. **Python 3.11+**: Required for MuJoCo compatibility
2. **PyTorch**: Neural network operations
3. **MuJoCo**: Physics simulation engine
4. **Gymnasium**: RL environment interface

### Setup Steps

#### 1. Install Python 3.11

```bash
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv python3.11-pip
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch
pip install torch torchvision torchaudio

# Install MuJoCo and related packages
pip install "gymnasium[mujoco]" pybind11
```

#### 4. Build FastTD3 with MuJoCo Support

```bash
# Build with MuJoCo support
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
make -j4
```

## Usage

### Basic Training

```bash
# Basic humanoid training
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 1000000

# Humanoid standup training
./fast_td3_mujoco --env_name HumanoidStandup-v5 --total_timesteps 2000000

# Walker2d training
./fast_td3_mujoco --env_name Walker2d-v5 --total_timesteps 500000
```

### Advanced Training

```bash
# High-performance training with all optimizations
./fast_td3_mujoco \
    --env_name Humanoid-v5 \
    --total_timesteps 2000000 \
    --num_envs 256 \
    --cuda \
    --obs_normalization \
    --reward_normalization

# Custom hyperparameters
./fast_td3_mujoco \
    --env_name Humanoid-v5 \
    --total_timesteps 1000000 \
    --learning_rate 3e-4 \
    --batch_size 256 \
    --buffer_size 1000000 \
    --learning_starts 1000 \
    --policy_frequency 2
```

### Training Scripts

Use the provided training scripts for easy setup:

```bash
# Run all humanoid training examples
./tests/scripts/humanoid_training.sh

# Run specific example
./tests/scripts/humanoid_training.sh humanoid_basic
```

## Command Line Options

### Environment Options
- `--env_name`: MuJoCo environment name (e.g., "Humanoid-v5")
- `--num_envs`: Number of parallel environments (default: 1)
- `--obs_normalization`: Enable observation normalization
- `--reward_normalization`: Enable reward normalization

### Training Options
- `--total_timesteps`: Total training timesteps
- `--learning_starts`: Steps before learning begins
- `--batch_size`: Batch size for training
- `--buffer_size`: Replay buffer size
- `--learning_rate`: Learning rate for networks
- `--policy_frequency`: Policy update frequency
- `--tau`: Target network update rate

### Hardware Options
- `--cuda`: Enable CUDA support
- `--device_rank`: GPU device rank (default: 0)

### Advanced Options
- `--use_cdq`: Enable CDQ (Continuous Distributional Q-learning)
- `--seed`: Random seed for reproducibility

## Environment Details

### Humanoid-v5
- **State Space**: 376-dimensional continuous
- **Action Space**: 17-dimensional continuous
- **Reward**: Based on forward progress and energy efficiency
- **Termination**: When humanoid falls or reaches time limit

### HumanoidStandup-v5
- **State Space**: 376-dimensional continuous
- **Action Space**: 17-dimensional continuous
- **Reward**: Based on height and energy efficiency
- **Termination**: When humanoid stands up or time limit reached

### Walker2d-v5
- **State Space**: 17-dimensional continuous
- **Action Space**: 6-dimensional continuous
- **Reward**: Based on forward progress and energy efficiency
- **Termination**: When walker falls or reaches time limit

## Performance Tips

### Training Optimization
1. **Use Multiple Environments**: Increase `--num_envs` for faster training
2. **Enable Normalization**: Use `--obs_normalization` and `--reward_normalization`
3. **GPU Acceleration**: Use `--cuda` for faster training
4. **Batch Size**: Use larger batch sizes (256-512) for stability

### Memory Management
1. **Buffer Size**: Adjust `--buffer_size` based on available memory
2. **Number of Environments**: Balance between speed and memory usage
3. **Batch Size**: Larger batches use more memory but train faster

### Hyperparameter Tuning
1. **Learning Rate**: Start with 3e-4, adjust based on training stability
2. **Policy Frequency**: Use 2 for most environments
3. **Tau**: Use 0.005 for stable target network updates

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility
**Problem**: MuJoCo fails to load with Python 3.13+
**Solution**: Use Python 3.11 or 3.10

```bash
# Check Python version
python --version

# If using wrong version, create new environment
python3.11 -m venv venv311
source venv311/bin/activate
```

#### 2. GLFW/OpenGL Issues
**Problem**: Display-related errors in headless environments
**Solution**: Set headless rendering

```bash
# For CI/CD or headless servers
export MUJOCO_GL=osmesa

# For local training without display
export MUJOCO_GL=egl
```

#### 3. Environment Not Found
**Problem**: Environment name not recognized
**Solution**: Check environment name and version

```bash
# List available environments
python3 -c "import gymnasium as gym; print([env for env in gym.envs.registry.keys() if 'Humanoid' in env])"

# Use correct environment name
./fast_td3_mujoco --env_name Humanoid-v5
```

#### 4. Memory Issues
**Problem**: Out of memory errors
**Solution**: Reduce batch size and number of environments

```bash
# Reduce memory usage
./fast_td3_mujoco --env_name Humanoid-v5 --batch_size 128 --num_envs 1
```

#### 5. Training Instability
**Problem**: Training diverges or becomes unstable
**Solution**: Adjust hyperparameters

```bash
# More stable training
./fast_td3_mujoco \
    --env_name Humanoid-v5 \
    --learning_rate 1e-4 \
    --batch_size 256 \
    --obs_normalization \
    --reward_normalization
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug log level
export SPDLOG_LEVEL=debug

# Run with debug output
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 1000
```

### Testing Integration

Test the MuJoCo integration:

```bash
# Run integration test
./tests/scripts/test_mujoco_integration.sh

# Test specific environment
./fast_td3_mujoco --env_name Humanoid-v5 --total_timesteps 10 --num_envs 1 --learning_starts 0
```

## CI/CD Integration

The MuJoCo integration is designed to work in CI/CD environments:

### GitHub Actions
```yaml
- name: Test MuJoCo Integration
  env:
    MUJOCO_GL: osmesa
  run: |
    ./tests/scripts/test_mujoco_integration.sh
```

### Docker
```dockerfile
# Set headless rendering
ENV MUJOCO_GL=osmesa

# Install dependencies
RUN pip install "gymnasium[mujoco]" pybind11
```

## Performance Benchmarks

### Training Times (approximate)
- **Humanoid-v5**: ~2-4 hours on GPU for 1M steps
- **Walker2d-v5**: ~30-60 minutes on GPU for 500K steps
- **HalfCheetah-v5**: ~15-30 minutes on GPU for 500K steps

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, GPU training
- **Optimal**: 32GB+ RAM, RTX 3080+ GPU

## Future Enhancements

- Support for more MuJoCo environments
- Real-time visualization and monitoring
- Advanced exploration strategies
- Multi-agent training support
- Custom environment creation
- Performance profiling tools

## Contributing

When contributing to the MuJoCo integration:

1. Test with multiple environments
2. Verify CI/CD compatibility
3. Update documentation
4. Add appropriate tests
5. Follow the existing code style

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [FastTD3 Repository](https://github.com/berkde/FastTD3) 