# HashEmb Project — Claude Instructions

## Project Overview
HashEmb is a host-memory (CPU DDR) based Hash Embedding Table library for large-scale search/recommendation/advertising systems. The dense model sits on GPU while the sparse embedding table lives in CPU memory, with data flowing between CPU and GPU via PCIe with pinned memory.

## Build Commands

### ⚠️ 平台编译说明
C++ 扩展是平台相关的，需要在目标机器上重新编译：
- `.so` / `.pyd` 不能跨平台/跨 Python 版本使用
- `.gitignore` 已排除所有编译产物，不会提交到仓库

### 在 GPU 机器上首次构建
```bash
# 方式一（推荐）：一行安装，自动编译
pip install -e .

# 方式二：仅编译扩展（不安装 Python 包）
python setup.py build_ext --inplace
```

### 开发流程（修改 C++ 后重新编译）
```bash
python setup.py build_ext --inplace
```

### 运行测试
```bash
# 全部测试
python -m pytest tests/ -v

# 仅 CPU 测试（跳过 GPU 测试）
python -m pytest tests/ -v -k "not cuda"

# 单个测试
python -m pytest tests/test_basic.py -v -k "test_name"
```

## Project Structure
```
hashemb/
├── setup.py                  # Build config (pybind11 + CUDA)
├── csrc/                     # C++ source
│   ├── hash_table.h/.cpp          # Robin Hood hash map core
│   ├── embedding_table.h/.cpp     # Embedding vector storage + batch ops
│   └── pybind_binding.cpp         # pybind11 → Python bridge
├── hashemb/                  # Python package
│   ├── __init__.py
│   ├── core.py               # PyTorch wrapper (autograd + nn.Module)
│   └── utils.py              # Pinned memory buffer, helpers
├── examples/                 # Usage demos
│   └── demo.py
├── tests/                    # Test suite
│   └── test_basic.py
├── CLAUDE.md                 # This file
└── README.md                 # Project design doc
```

## Design Conventions

### Hash Table Architecture
- **Single table** for all features (unified hash embedding table)
- **16-way internal sharding** (buckets) for concurrency
- **Open addressing** with Robin Hood hashing
- **Continuous memory buffer** for embedding vectors (pre-allocated)

### API Conventions
- `embedding_dim`: dimension of each embedding vector
- `capacity`: max number of unique keys (pre-allocated slots)
- Lookup only **lifts rank**, never reduces:
  - `(B,) → (B, D)`
  - `(B, S) → (B, S, D)`
  - `(B, F, S) → (B, F, S, D)`
- Backward: gradient sum for duplicate feat_ids (same as `nn.Embedding`)
- No forward pooling — keep sequence structure intact

### Parameter Name Convention
- `embedding_dim` → `dim` in code (short, consistent with PyTorch convention)
- `capacity` stays as `capacity`
- Use `feat_id` or `id` for feature ID tensor

### Coding Style
- C++: Google style (snake_case for functions, PascalCase for classes)
- Python: PEP 8 (snake_case)
- Headers: fully self-contained, inline hot paths marked `__attribute__((always_inline))`
