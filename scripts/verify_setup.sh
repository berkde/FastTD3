#!/bin/bash

# FastTD3 Setup Verification Script
# This script verifies that all components are properly installed and configured

set -e

echo "🔍 FastTD3 Setup Verification"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${RED}❌ $message${NC}"
    fi
}

# Check Python version
echo -e "\n${BLUE}Python Environment:${NC}"
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "not found")
if [[ "$python_version" == "3.11"* ]] || [[ "$python_version" == "3.10"* ]]; then
    print_status "OK" "Python $python_version (compatible with MuJoCo)"
elif [[ "$python_version" == "3.13"* ]]; then
    print_status "WARN" "Python $python_version (may have MuJoCo compatibility issues)"
else
    print_status "FAIL" "Python $python_version (incompatible with MuJoCo)"
fi

# Check PyTorch
echo -e "\n${BLUE}PyTorch Installation:${NC}"
if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    print_status "OK" "PyTorch is installed"
else
    print_status "FAIL" "PyTorch is not installed"
fi

# Check MuJoCo
echo -e "\n${BLUE}MuJoCo Dependencies:${NC}"
if python3 -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)" 2>/dev/null; then
    print_status "OK" "Gymnasium is installed"
else
    print_status "FAIL" "Gymnasium is not installed"
fi

if python3 -c "import mujoco; print('MuJoCo version:', mujoco.__version__)" 2>/dev/null; then
    print_status "OK" "MuJoCo is installed"
else
    print_status "FAIL" "MuJoCo is not installed"
fi

# Check pybind11
echo -e "\n${BLUE}pybind11:${NC}"
if python3 -c "import pybind11; print('pybind11 version:', pybind11.__version__)" 2>/dev/null; then
    print_status "OK" "pybind11 is installed"
else
    print_status "FAIL" "pybind11 is not installed"
fi

# Check C++ dependencies
echo -e "\n${BLUE}C++ Dependencies:${NC}"

# Check Eigen3
if pkg-config --exists eigen3 2>/dev/null; then
    eigen_version=$(pkg-config --modversion eigen3 2>/dev/null || echo "unknown")
    print_status "OK" "Eigen3 $eigen_version is installed"
else
    print_status "FAIL" "Eigen3 is not installed"
fi

# Check spdlog
if pkg-config --exists spdlog 2>/dev/null; then
    spdlog_version=$(pkg-config --modversion spdlog 2>/dev/null || echo "unknown")
    print_status "OK" "spdlog $spdlog_version is installed"
else
    print_status "FAIL" "spdlog is not installed"
fi

# Check nlohmann-json
if pkg-config --exists nlohmann_json 2>/dev/null; then
    print_status "OK" "nlohmann-json is installed"
else
    print_status "FAIL" "nlohmann-json is not installed"
fi

# Check GTest
if pkg-config --exists gtest 2>/dev/null; then
    print_status "OK" "Google Test is installed"
else
    print_status "FAIL" "Google Test is not installed"
fi

# Check if build directory exists and has executables
echo -e "\n${BLUE}Build Status:${NC}"
if [ -d "build" ]; then
    print_status "OK" "Build directory exists"
    
    # Check for executables
    if [ -f "build/fast_td3_simple" ]; then
        print_status "OK" "fast_td3_simple executable found"
    else
        print_status "FAIL" "fast_td3_simple executable not found"
    fi
    
    if [ -f "build/fast_td3_pytorch" ]; then
        print_status "OK" "fast_td3_pytorch executable found"
    else
        print_status "WARN" "fast_td3_pytorch executable not found (PyTorch mode)"
    fi
    
    if [ -f "build/fast_td3_mujoco" ]; then
        print_status "OK" "fast_td3_mujoco executable found"
    else
        print_status "WARN" "fast_td3_mujoco executable not found (MuJoCo mode)"
    fi
else
    print_status "WARN" "Build directory does not exist - run 'mkdir build && cd build && cmake .. && make'"
fi

# Check environment variables
echo -e "\n${BLUE}Environment Variables:${NC}"
if [ -n "$MUJOCO_GL" ]; then
    print_status "OK" "MUJOCO_GL is set to: $MUJOCO_GL"
else
    print_status "WARN" "MUJOCO_GL is not set (may cause display issues)"
fi

# Summary
echo -e "\n${BLUE}Summary:${NC}"
echo "=============================="

if [ -f "build/fast_td3_simple" ]; then
    echo -e "${GREEN}✅ Simple Mode: Ready${NC}"
else
    echo -e "${RED}❌ Simple Mode: Not ready${NC}"
fi

if [ -f "build/fast_td3_pytorch" ]; then
    echo -e "${GREEN}✅ PyTorch Mode: Ready${NC}"
else
    echo -e "${YELLOW}⚠️  PyTorch Mode: Not ready${NC}"
fi

if [ -f "build/fast_td3_mujoco" ]; then
    echo -e "${GREEN}✅ MuJoCo Mode: Ready${NC}"
else
    echo -e "${YELLOW}⚠️  MuJoCo Mode: Not ready${NC}"
fi

echo -e "\n${BLUE}Next Steps:${NC}"
if [ ! -d "build" ] || [ ! -f "build/fast_td3_simple" ]; then
    echo "1. Build the project: mkdir build && cd build && cmake .. && make"
fi

if [ ! -f "build/fast_td3_pytorch" ]; then
    echo "2. Install PyTorch: pip install torch torchvision torchaudio"
    echo "3. Rebuild with PyTorch: cd build && cmake -DCMAKE_PREFIX_PATH=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') .. && make"
fi

if [ ! -f "build/fast_td3_mujoco" ]; then
    echo "4. Install MuJoCo: pip install 'gymnasium[mujoco]' pybind11"
    echo "5. Rebuild with MuJoCo support"
fi

echo -e "\n${GREEN}🎉 Setup verification complete!${NC}" 