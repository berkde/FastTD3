#!/bin/bash

# Humanoid Training with MuJoCo
# This script demonstrates training humanoid robots using FastTD3 with MuJoCo

echo "FastTD3 Humanoid Training with MuJoCo"
echo "====================================="

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Check if MuJoCo executable exists
if [ ! -f "./fast_td3_mujoco" ]; then
    echo "Error: fast_td3_mujoco executable not found. Please build the project first."
    echo "Run: mkdir -p build && cd build && cmake .. && make -j4"
    exit 1
fi

# Example 1: Basic Humanoid training
echo "Example 1: Basic Humanoid training"
./fast_td3_mujoco \
    --env_name Humanoid-v2 \
    --seed 42 \
    --total_timesteps 1000000 \
    --num_envs 64 \
    --batch_size 16384 \
    --buffer_size 1024000 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_frequency 2 \
    --noise_clip 0.5 \
    --policy_noise 0.2 \
    --learning_starts 1000 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --obs_normalization \
    --use_cdq

# Example 2: Humanoid Standup training
echo "Example 2: Humanoid Standup training"
./fast_td3_mujoco \
    --env_name HumanoidStandup-v2 \
    --seed 123 \
    --total_timesteps 2000000 \
    --num_envs 128 \
    --batch_size 32768 \
    --buffer_size 2048000 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_frequency 2 \
    --noise_clip 0.5 \
    --policy_noise 0.2 \
    --learning_starts 2000 \
    --eval_interval 10000 \
    --save_interval 20000 \
    --obs_normalization \
    --use_cdq

# Example 3: Walker2d training
echo "Example 3: Walker2d training"
./fast_td3_mujoco \
    --env_name Walker2d-v2 \
    --seed 456 \
    --total_timesteps 500000 \
    --num_envs 64 \
    --batch_size 16384 \
    --buffer_size 512000 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_frequency 2 \
    --noise_clip 0.5 \
    --policy_noise 0.2 \
    --learning_starts 1000 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --obs_normalization \
    --use_cdq

# Example 4: High-performance training with all optimizations
echo "Example 4: High-performance Humanoid training"
./fast_td3_mujoco \
    --env_name Humanoid-v2 \
    --seed 789 \
    --total_timesteps 2000000 \
    --num_envs 256 \
    --batch_size 65536 \
    --buffer_size 4096000 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_frequency 2 \
    --noise_clip 0.5 \
    --policy_noise 0.2 \
    --learning_starts 5000 \
    --eval_interval 10000 \
    --save_interval 50000 \
    --obs_normalization \
    --use_cdq \
    --cuda \
    --device_rank 0

echo "Humanoid training examples completed!"
echo "Check the checkpoints/ directory for saved models"
echo "Check the logs/ directory for training logs"
echo ""
echo "To monitor training progress, you can use:"
echo "  tail -f logs/training.log"
echo ""
echo "To visualize a trained model, you can create a separate evaluation script." 