#!/bin/bash

# FastTD3 PyTorch Setup Script
# This script helps set up PyTorch (LibTorch) for the FastTD3 project

set -e

echo "🚀 FastTD3 PyTorch Setup Script"
echo "================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Detected OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected OS: macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    echo "Detected OS: Windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="x64"
elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    ARCH="arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "📋 Architecture: $ARCH"

# Create third_party directory
mkdir -p third_party

# Function to download and extract LibTorch
download_libtorch() {
    local version=$1
    local url=$2
    local filename=$(basename $url)
    
    echo "Downloading PyTorch $version..."
    echo "URL: $url"
    
    if command -v wget &> /dev/null; then
        wget -O "third_party/$filename" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "third_party/$filename" "$url"
    else
        echo "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    echo "Extracting PyTorch..."
    cd third_party
    tar -xzf "$filename"
    rm "$filename"
    cd ..
    
    echo "PyTorch $version installed successfully!"
}

# Function to install via pip
install_via_pip() {
    echo "🐍 Installing PyTorch via pip..."
    
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Please install Python 3.7+ first."
        exit 1
    fi
    
    # Install PyTorch
    if [[ "$OS" == "macos" ]]; then
        python3 -m pip install torch torchvision torchaudio
    else
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    echo "PyTorch installed via pip!"
    echo "You can now build with: cmake -DCMAKE_PREFIX_PATH=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') .."
}

# Main setup logic
echo ""
echo "Choose installation method:"
echo "1. Download LibTorch (recommended for C++ projects)"
echo "2. Install via pip (requires Python)"
echo "3. Manual setup instructions"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Available PyTorch versions:"
        echo "1. PyTorch 2.1.0 (stable)"
        echo "2. PyTorch 2.0.1 (stable)"
        echo "3. PyTorch 1.13.1 (stable)"
        read -p "Enter version choice (1-3): " version_choice
        
        case $version_choice in
            1)
                version="2.1.0"
                ;;
            2)
                version="2.0.1"
                ;;
            3)
                version="1.13.1"
                ;;
            *)
                echo "Invalid choice. Using PyTorch 2.1.0"
                version="2.1.0"
                ;;
        esac
        
        # Construct download URL
        if [[ "$OS" == "linux" ]]; then
            if [[ "$ARCH" == "x64" ]]; then
                url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip"
            else
                url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip"
            fi
        elif [[ "$OS" == "macos" ]]; then
            if [[ "$ARCH" == "x64" ]]; then
                url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-$version.zip"
            else
                url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-$version.zip"
            fi
        else
            url="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-$version%2Bcpu.zip"
        fi
        
        download_libtorch $version $url
        ;;
    2)
        install_via_pip
        ;;
    3)
        echo ""
        echo "Manual Setup Instructions:"
        echo "============================="
        echo ""
        echo "1. Download LibTorch from: https://pytorch.org/get-started/locally/"
        echo "2. Extract to: third_party/libtorch/"
        echo "3. Or install PyTorch via pip: pip install torch"
        echo "4. Build with: cmake .. && make"
        echo ""
        echo "For more details, see: https://pytorch.org/cppdocs/installing.html"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Build the project: mkdir build && cd build && cmake .. && make"
echo "2. Run tests: ./tests_simple"
echo "3. Run PyTorch version: ./fast_td3_pytorch (if PyTorch was found)"
echo "4. Run simple version: ./fast_td3_simple"
echo ""
echo "💡 If you encounter issues:"
echo "- Check that PyTorch was found: cmake .. | grep -i torch"
echo "- Verify library paths: ldd ./fast_td3_pytorch (Linux) or otool -L ./fast_td3_pytorch (macOS)"
echo "- Check CMake logs for detailed error messages" 