#!/bin/bash

# FastTD3 C++ Build Script
# This script automates the build process with different configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_deps+=("make")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All dependencies found!"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up build environment..."
    
    local os=$(detect_os)
    
    # Check for LibTorch
    if [[ -z "$CMAKE_PREFIX_PATH" ]]; then
        print_warning "CMAKE_PREFIX_PATH not set. Please set it to your LibTorch installation path."
        print_status "Example: export CMAKE_PREFIX_PATH=/path/to/libtorch:\$CMAKE_PREFIX_PATH"
        
        # Try to find LibTorch in common locations
        local libtorch_paths=(
            "/usr/local/libtorch"
            "/opt/libtorch"
            "$HOME/libtorch"
            "$HOME/.local/libtorch"
        )
        
        for path in "${libtorch_paths[@]}"; do
            if [[ -d "$path" ]]; then
                print_status "Found LibTorch at: $path"
                export CMAKE_PREFIX_PATH="$path:$CMAKE_PREFIX_PATH"
                break
            fi
        done
    fi
    
    # Create build directory
    mkdir -p build
    cd build
}

# Function to build project
build_project() {
    local build_type=${1:-Release}
    local enable_tests=${2:-false}
    local enable_cuda=${3:-false}
    
    print_status "Building FastTD3 (${build_type})..."
    
    # Configure with CMake
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DCMAKE_CXX_STANDARD=17"
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    )
    
    if [[ "$enable_tests" == "true" ]]; then
        cmake_args+=("-DBUILD_TESTS=ON")
    fi
    
    if [[ "$enable_cuda" == "true" ]]; then
        cmake_args+=("-DUSE_CUDA=ON")
    fi
    
    cmake "${cmake_args[@]}" ..
    
    # Build
    local jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    make -j"$jobs"
    
    print_success "Build completed successfully!"
}

# Function to run tests
run_tests() {
    if [[ -f "fast_td3_tests" ]]; then
        print_status "Running tests..."
        ./fast_td3_tests
        print_success "Tests completed!"
    else
        print_warning "Test executable not found. Build with tests enabled first."
    fi
}

# Function to clean build
clean_build() {
    print_status "Cleaning build directory..."
    rm -rf build/*
    print_success "Build directory cleaned!"
}

# Function to show help
show_help() {
    echo "FastTD3 C++ Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --debug         Build in debug mode"
    echo "  -t, --tests         Build with tests enabled"
    echo "  -c, --cuda          Build with CUDA support"
    echo "  -r, --run-tests     Run tests after building"
    echo "  --clean             Clean build directory"
    echo "  --check-deps        Check dependencies only"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build in release mode"
    echo "  $0 -d               # Build in debug mode"
    echo "  $0 -t -r            # Build with tests and run them"
    echo "  $0 -c               # Build with CUDA support"
    echo "  $0 --clean          # Clean build directory"
}

# Main script
main() {
    local build_type="Release"
    local enable_tests=false
    local enable_cuda=false
    local run_tests_after=false
    local clean_only=false
    local check_deps_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--debug)
                build_type="Debug"
                shift
                ;;
            -t|--tests)
                enable_tests=true
                shift
                ;;
            -c|--cuda)
                enable_cuda=true
                shift
                ;;
            -r|--run-tests)
                run_tests_after=true
                shift
                ;;
            --clean)
                clean_only=true
                shift
                ;;
            --check-deps)
                check_deps_only=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check dependencies
    check_dependencies
    
    if [[ "$check_deps_only" == "true" ]]; then
        exit 0
    fi
    
    # Clean if requested
    if [[ "$clean_only" == "true" ]]; then
        clean_build
        exit 0
    fi
    
    # Setup environment
    setup_environment
    
    # Build project
    build_project "$build_type" "$enable_tests" "$enable_cuda"
    
    # Run tests if requested
    if [[ "$run_tests_after" == "true" ]]; then
        run_tests
    fi
    
    print_success "FastTD3 C++ build completed successfully!"
    print_status "You can now run: ./fast_td3 --help"
}

# Run main function with all arguments
main "$@" 