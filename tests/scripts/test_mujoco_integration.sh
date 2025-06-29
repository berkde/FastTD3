#!/bin/bash

# Test script for MuJoCo integration
# This script tests the MuJoCo environment creation and basic functionality

set -e

echo "🧪 Testing MuJoCo Integration..."

# Check if we're in a CI environment
if [ -n "$CI" ]; then
    echo "Running in CI environment"
    export MUJOCO_GL=osmesa
fi

# Check if fast_td3_mujoco executable exists
if [ ! -f "./fast_td3_mujoco" ]; then
    echo "❌ fast_td3_mujoco executable not found!"
    echo "Available executables:"
    ls -la | grep fast_td3 || echo "No fast_td3 executables found"
    exit 1
fi

echo "✅ fast_td3_mujoco executable found"

# Test 1: Help command
echo "📋 Testing help command..."
./fast_td3_mujoco --help || {
    echo "❌ Help command failed"
    exit 1
}
echo "✅ Help command works"

# Test 2: Environment creation test
echo "🌍 Testing environment creation..."
timeout 30s ./fast_td3_mujoco \
    --env_name Humanoid-v5 \
    --total_timesteps 10 \
    --num_envs 1 \
    --learning_starts 0 \
    --batch_size 32 || {
    echo "❌ Environment creation test failed"
    exit 1
}
echo "✅ Environment creation works"

# Test 3: Basic training test (very short)
echo "🎯 Testing basic training..."
timeout 60s ./fast_td3_mujoco \
    --env_name Humanoid-v5 \
    --total_timesteps 50 \
    --num_envs 2 \
    --learning_starts 0 \
    --batch_size 16 || {
    echo "❌ Basic training test failed"
    exit 1
}
echo "✅ Basic training works"

# Test 4: Different environment test
echo "🤖 Testing Walker2d environment..."
timeout 30s ./fast_td3_mujoco \
    --env_name Walker2d-v5 \
    --total_timesteps 10 \
    --num_envs 1 \
    --learning_starts 0 \
    --batch_size 32 || {
    echo "❌ Walker2d environment test failed"
    exit 1
}
echo "✅ Walker2d environment works"

echo "🎉 All MuJoCo integration tests passed!"
echo "✅ Environment creation: OK"
echo "✅ Basic training: OK"
echo "✅ Multiple environments: OK"
echo "✅ CI/CD compatibility: OK" 