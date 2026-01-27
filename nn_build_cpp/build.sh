#!/bin/bash
#
# build.sh - Build the C++ batch loader library
#
# Prerequisites:
#   - CMake 3.14+
#   - C++17 compiler (g++ 7+ or clang++ 5+)
#   - libzstd-dev
#
# Ubuntu/Debian:
#   sudo apt-get install cmake build-essential libzstd-dev pkg-config
#
# Fedora/RHEL:
#   sudo dnf install cmake gcc-c++ libzstd-devel pkgconfig
#
# macOS:
#   brew install cmake zstd pkg-config
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Building C++ Batch Loader"
echo "========================================"

# Check dependencies
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Please install cmake."
    exit 1
fi

if ! pkg-config --exists libzstd; then
    echo "ERROR: libzstd not found. Please install libzstd-dev."
    echo "  Ubuntu/Debian: sudo apt-get install libzstd-dev"
    echo "  Fedora/RHEL:   sudo dnf install libzstd-devel"
    echo "  macOS:         brew install zstd"
    exit 1
fi

echo "  ✓ cmake found"
echo "  ✓ libzstd found"

# Create build directory
mkdir -p build
cd build

# Configure
echo ""
echo "Configuring..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check result
if [ -f "libbatch_loader.so" ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "========================================"
    echo ""
    echo "Library: $(pwd)/libbatch_loader.so"
    echo ""
    echo "To use in Python:"
    echo "  1. Copy libbatch_loader.so to your project directory, or"
    echo "  2. Add $(pwd) to LD_LIBRARY_PATH"
    echo ""
    echo "Example:"
    echo "  cp $(pwd)/libbatch_loader.so /path/to/your/project/"
    echo "  # or"
    echo "  export LD_LIBRARY_PATH=$(pwd):\$LD_LIBRARY_PATH"
    echo ""
else
    echo "ERROR: Build failed - libbatch_loader.so not found"
    exit 1
fi
