#!/bin/bash

# Simple FastTD3 Training Examples
# This script demonstrates various training configurations

echo "FastTD3 C++ Training Examples"
echo "============================="

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Example 1: Basic HumanoidBench training
echo "Example 1: Basic HumanoidBench training"
./fast_td3 \
    --env_name h1hand-stand-v0 \
    --seed 42 \
    --total_timesteps 50000 \
    --num_envs 64 \
    --batch_size 16384 \
    --eval_interval 1000 \
    --save_interval 5000

# Example 2: MuJoCo Playground with custom settings
echo "Example 2: MuJoCo Playground training"
./fast_td3 \
    --env_name T1JoystickFlatTerrain \
    --seed 123 \
    --total_timesteps 100000 \
    --num_envs 512 \
    --batch_size 16384 \
    --gamma 0.97 \
    --v_min -10.0 \
    --v_max 10.0 \
    --use_cdq \
    --obs_normalization

# Example 3: IsaacLab environment
echo "Example 3: IsaacLab training"
./fast_td3 \
    --env_name Isaac-Velocity-Flat-G1-v0 \
    --seed 456 \
    --total_timesteps 75000 \
    --num_envs 2048 \
    --num_steps 8 \
    --num_updates 4 \
    --action_bounds 1.0 \
    --render_interval 0

# Example 4: High-performance training with all optimizations
echo "Example 4: High-performance training"
./fast_td3 \
    --env_name h1hand-reach-v0 \
    --seed 789 \
    --total_timesteps 100000 \
    --num_envs 128 \
    --batch_size 32768 \
    --buffer_size 51200 \
    --critic_learning_rate 3e-4 \
    --actor_learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 0.1 \
    --num_updates 2 \
    --use_cdq \
    --obs_normalization \
    --amp \
    --compile \
    --cuda \
    --device_rank 0 \
    --eval_interval 2000 \
    --save_interval 10000

echo "Training examples completed!"
echo "Check the checkpoints/ directory for saved models"
echo "Check the logs/ directory for training logs" 