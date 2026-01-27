# C++ Batch Loader for NNUE Training

A high-performance multi-threaded data loading pipeline that can saturate high-end GPUs even with limited CPU cores.

## Performance

| Loader Type | CPU Cores Needed | GPU Utilization |
|-------------|------------------|-----------------|
| Python DataLoader | 10-16 | ~80-95% |
| **C++ Batch Loader** | **4-6** | **~80-95%** |

The C++ loader achieves the same GPU utilization with **2-3x fewer CPU cores** because:
- Native C++ is 10-50x faster than Python for data parsing
- Multi-threaded decompression and batch preparation
- Zero-copy transfer to PyTorch via numpy arrays
- Pre-sorted indices for coalesced sparse tensors

## Prerequisites

- CMake 3.14+
- C++17 compiler (g++ 7+ or clang++ 5+)
- libzstd development files
- pthreads

### Ubuntu/Debian
```bash
sudo apt-get install cmake build-essential libzstd-dev pkg-config
```

### Fedora/RHEL
```bash
sudo dnf install cmake gcc-c++ libzstd-devel pkgconfig
```

### macOS
```bash
brew install cmake zstd pkg-config
```

## Building

```bash
cd cpp_batch_loader
chmod +x build.sh
./build.sh
```

Or manually:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

This creates `libbatch_loader.so` in the `build/` directory.

## Installation

Copy the library to your project directory:
```bash
cp cpp_batch_loader/build/libbatch_loader.so .
```

Or add to library path:
```bash
export LD_LIBRARY_PATH=/path/to/cpp_batch_loader/build:$LD_LIBRARY_PATH
```

Also copy `batch_loader.py` to your project:
```bash
cp cpp_batch_loader/batch_loader.py .
```

## Usage

### With nn_train.py (Automatic)

The training script automatically uses the C++ loader if available:

```python
# In nn_train.py, these are the defaults:
CPP_LOADER = True   # Enable C++ loader
SPARSE_COO = True   # Fallback if C++ loader not available
```

Then run training as usual:
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue --num-workers 4
```

If the C++ library is found, you'll see:
```
Training mode: C++ batch loader (high performance)
```

If not found, it falls back to Python with a note:
```
Training mode: Sparse COO tensors (dense gradients)
Note: C++ loader not available, falling back to Python loader
      Build with: cd cpp_batch_loader && ./build.sh
```

### Standalone Usage

```python
from batch_loader import CppBatchLoader
import torch

loader = CppBatchLoader(
    shard_paths=["data/shard_001.bin.zst", "data/shard_002.bin.zst"],
    batch_size=16384,
    num_workers=4,          # C++ threads (not Python processes)
    num_features=40960,     # HalfKP feature count
    shuffle=True,
    device=torch.device("cuda")
)

for white_sparse, black_sparse, stm, targets in loader:
    outputs = model(white_sparse, black_sparse, stm)
    loss = criterion(outputs, targets)
    # ... training step
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    C++ Shared Library                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │     │
│  │ Read Shard  │  │ Read Shard  │  │ Read Shard  │     │
│  │ Decompress  │  │ Decompress  │  │ Decompress  │     │
│  │ Parse Batch │  │ Parse Batch │  │ Parse Batch │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌───────────────────────┐                  │
│              │  Thread-safe Queue    │                  │
│              │  (Ready Batches)      │                  │
│              └───────────┬───────────┘                  │
└──────────────────────────┼──────────────────────────────┘
                           ▼
                    Python (ctypes)
                    torch.sparse_coo_tensor()
                    GPU Training
```

## Configuration Recommendations

### Half A100 (40GB) + 6 vCPUs
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 4 --batch-size 16384
```

### Full A100 (80GB) + 12 vCPUs  
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 8 --batch-size 32768
```

### RTX 4090 + 8 vCPUs
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 6 --batch-size 16384
```

## Files

| File | Description |
|------|-------------|
| `batch_loader.h` | C header file with API definitions |
| `batch_loader.cpp` | C++ implementation |
| `batch_loader.py` | Python wrapper using ctypes |
| `CMakeLists.txt` | CMake build configuration |
| `build.sh` | Convenience build script |

## Troubleshooting

### Library not found
```
RuntimeError: Could not find libbatch_loader.so
```
**Solution:** Copy `libbatch_loader.so` to your project directory, or set:
```bash
export LD_LIBRARY_PATH=/path/to/cpp_batch_loader/build:$LD_LIBRARY_PATH
```

### Decompression errors
```
Decompression failed: ...
```
**Solution:** Ensure shard files are valid zstd-compressed files created by `prepare_data.py`.

### Still low GPU utilization
If GPU utilization is still low with the C++ loader:
1. Try increasing `--num-workers` (up to number of CPU cores - 2)
2. Check if disk I/O is the bottleneck (use SSD)
3. Monitor with `htop` to see if CPU cores are saturated

### Compilation errors
If CMake fails to find zstd:
```bash
# Ubuntu/Debian
sudo apt-get install libzstd-dev pkg-config

# Then rebuild
cd build && cmake .. && make -j$(nproc)
```
