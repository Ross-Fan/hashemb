# HashEmb Project — Claude Instructions

## Project Overview
HashEmb is a host-memory (CPU DDR) based Hash Embedding Table library for large-scale search/recommendation/advertising systems. The dense model sits on GPU while the sparse embedding table lives in CPU memory, with data flowing between CPU and GPU via PCIe with pinned memory.

### Key Design Decisions
- **Single-table architecture** with 16-way sharding
- **Block-based on-demand allocation** (no hard capacity limit, memory grows as keys are inserted)
- **Auto-grow hash buckets** (Robin Hood open addressing, doubles and rehashes when full)
- **Native C++ optimizers** (SGD / Adam with in-place gradient zeroing)
- **Correct gradient accumulation** (`.contiguous().numpy()` in backward for non-contiguous tensors)

## Build Commands

### ⚠️ 平台编译说明
C++ 扩展是平台相关的，需要在目标机器上重新编译：
- `.so` / `.pyd` 不能跨平台/跨 Python 版本使用
- `.gitignore` 已排除所有编译产物，不会提交到仓库

### 构建与编译
```bash
# 一行安装，自动编译
pip install -e .

# 仅编译扩展（修改 C++ 后）
python setup.py build_ext --inplace
```

### 运行测试
```bash
# 全部测试（37 个用例）
python -m pytest tests/ -v

# 单个文件
python -m pytest tests/test_basic.py -v
python -m pytest tests/test_dynamic_expansion.py -v -s

# 仅 CPU 测试
python -m pytest tests/ -v -k "not cuda"

# ML-1M 对比（需先下载数据集）
python examples/compare_ml1m.py
```

## Project Structure
```
hashemb/
├── setup.py                  # Build config (pybind11)
├── csrc/                     # C++ source
│   ├── hash_table.h/.cpp          # Robin Hood hash map + auto-grow buckets
│   ├── embedding_table.h/.cpp     # Block-based embedding storage + native optimizers
│   └── pybind_binding.cpp         # pybind11 → Python bridge
├── hashemb/                  # Python package
│   ├── __init__.py
│   ├── core.py               # PyTorch autograd.Function + HashEmbedding(nn.Module)
│   └── utils.py              # Pinned memory buffer, helpers
├── examples/
│   ├── demo.py               # Basic usage
│   └── compare_ml1m.py       # ML-1M nn.Embedding vs HashEmb comparison
├── tests/
│   ├── test_basic.py               # C++ core + PyTorch wrapper tests (21 cases)
│   ├── test_adam_verify.py         # C++ Adam vs PyTorch manual Adam step-by-step
│   ├── test_dynamic_expansion.py   # Stress test: block expansion + hash auto-grow
│   ├── test_pure_torch_vs_hash.py  # nn.Embedding vs HashEmb step-by-step SGD+Adam
│   └── test_submodel.py            # BigModel architecture + sequence + cross-device
├── CLAUDE.md                 # This file
└── README.md                 # Project design doc
```

## Design Conventions

### Hash Table Architecture
- **Single table** for all features (unified hash embedding table)
- **16-way internal sharding** (buckets) for concurrency
- **Open addressing** with Robin Hood hashing
- **Auto-grow buckets** — no hard capacity limit, no `hash_int64` (feat_id uses `key & 0xF` for bucket, `key & mask` for home position)
- **Block-based memory** — `[block_size, D]` blocks allocated on demand (default 10M per block)

### Native C++ Optimizers (embedding_table.cpp)
- SGD: `w[d] -= lr * g[d]`
- Adam: `m[d] = b1*m + (1-b1)*g; v[d] = b2*v + (1-b2)*g²; w -= lr*m̂/(√v̂+ɛ)`
- **In-place gradient zeroing**: inside `step()` loop, `g[d] = 0.0f` after use — avoids `zero_grad()` memset on entire block
- **Fast path (single block)**: direct pointer arithmetic `base_ptr + slot * D`, no division via `slot_ptr()`
- **No `zero_grad()` call** after the loop — gradients zeroed in-place in the inner loop

### Gradient Contiguity (core.py)
```python
@staticmethod
def backward(ctx, grad_output):
    grad_np = grad_output.contiguous().numpy()  # ← critical: grad_output may be non-contiguous!
    ...
    ctx.table.scatter_add_grad(slot_np, grad_np)
```

### API Conventions
- `embedding_dim`: dimension of each embedding vector
- `capacity`: initial hash table capacity hint (NOT a hard limit — table auto-grows)
- `block_size`: number of slots per memory block (default 10M)
- Lookup only **lifts rank**, never reduces:
  - `(B,) → (B, D)`
  - `(B, S) → (B, S, D)`
  - `(B, F, S) → (B, F, S, D)`
- Backward: gradient sum for duplicate feat_ids (same as `nn.Embedding`)
- No forward pooling — keep sequence structure intact

### Training Loop Pattern
```python
emb = HashEmbedding(64, 10_000_000, optimizer='adam', lr=0.001)

for batch in loader:
    out = emb(keys)         # forward: CPU lookup → grad graph
    loss = F.cross_entropy(dense(out), y)
    loss.backward()          # grads → C++ grad_buffer (contiguous!)
    dense_opt.step()         # update dense params
    emb.step()               # C++ Adam/SGD update + zero grads in-place

# Save / load
torch.save({'hash_emb': emb.state_dict()}, 'ckpt.pt')
emb.load_state_dict(torch.load('ckpt.pt')['hash_emb'])
```

### BigModel (Per-Field Tables)
When feat_id spaces overlap (e.g., user_id=1 and movie_id=1), use per-field tables:
```python
class FeatureEmbedder(torch.nn.Module):
    def __init__(self, feature_names, dim, capacity, optimizer='adam', lr=0.01):
        for name in feature_names:
            self.__setattr__(f'emb_{name}', HashEmbedding(dim, capacity, optimizer=optimizer, lr=lr))

    def forward(self, feat_dict):
        return {name: self.emb_{name}(feat_dict[name]) for name in self.feature_names}

    def step(self):
        for name in self.feature_names:
            self.emb_{name}.step()
```

### Parameter Name Convention
- `embedding_dim` → keep as `embedding_dim` (used in both Python and C++)
- `capacity` — hash table initial hint, NOT hard limit
- Use `feat_id` or `id` for feature ID tensor

### Coding Style
- C++: Google style (snake_case for functions, PascalCase for classes)
- Python: PEP 8 (snake_case)
- Headers: fully self-contained, inline hot paths marked `__attribute__((always_inline))`

## Performance
ML-1M benchmark (batch=1024, dim=16, Adam, Apple M2):
- `nn.Embedding`: 1.11 s/epoch
- `HashEmb`    : 1.22 s/epoch (≈1.1x overhead)
- `step()` optimization brought 72ms → 0.09ms (800x) by eliminating 640MB memset and slot_ptr division
