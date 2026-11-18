**English** | [中文](#中文说明)

# HashEmb — CPU Host-Memory Hash Embedding Table for PyTorch

## Overview

HashEmb is a high-performance hash embedding table library backed by CPU host memory (DDR), designed for large-scale sparse embeddings in search / recommendation / advertising systems.

### Core Idea

Store the large sparse embedding table in CPU host memory (DDR, scaling to terabytes), keep the dense model on the GPU (HBM), implement the underlying high-performance hash table in C++ and expose it to PyTorch via a pybind11 wrapper, and move data between CPU and GPU over PCIe.

```
┌───────────────────────────────────────────────┐
│                   GPU (HBM)                     │
│  ┌─────────────────────────────────────────┐   │
│  │  Dense Network (MLP / Transformer etc.) │   │
│  │  Embedding Tensor (GPU pinned buffer)   │   │
│  └─────────────────────────────────────────┘   │
│            ▲ PCIe (DMA, pinned memory)          │
│            │                                     │
│  ┌────────┴────────────────────────────────┐   │
│  │           CPU (DDR)                       │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │  C++ Hash Embedding Table        │    │   │
│  │  │  • Robin Hood open addressing    │    │   │
│  │  │  • 16-way internal sharding      │    │   │
│  │  │  • Block-based on-demand alloc   │    │   │
│  │  │  • Native SGD / Adam optimizers  │    │   │
│  │  └──────────────────────────────────┘    │   │
│  │            ↕ pybind11                     │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │  PyTorch Wrapper                 │    │   │
│  │  │  • autograd.Function             │    │   │
│  │  │  • nn.Module (HashEmbedding)     │    │   │
│  │  │  • Checkpoint serialisation      │    │   │
│  │  └──────────────────────────────────┘    │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Installation & Build

### Requirements

| Dependency | Notes |
|---|---|
| Python | ≥ 3.8 |
| C++ compiler | GCC ≥ 7 / Clang (macOS) |
| PyTorch | ≥ 1.13 |
| pybind11 | fetched automatically by setup.py |

### Build

```bash
# Install from source (compiles the C++ extension automatically)
pip install -e .

# Compile the C++ extension only
python setup.py build_ext --inplace

# Verify the installation
python -c "from hashemb import HashEmbedding; print('OK')"

# Run the full test suite
python -m pytest tests/ -v
```

> **Note**: the C++ extension (`.so` / `.pyd`) is platform-specific and **must be recompiled on the target machine**.

### Rebuild after changing C++

```bash
python setup.py build_ext --inplace
```

### Running tests

```bash
# Full suite (C++ core + PyTorch wrapper)
python -m pytest tests/ -v

# A single test file
python -m pytest tests/test_dynamic_expansion.py -v -s

# CPU-only tests
python -m pytest tests/ -v -k "not cuda"

# ML-1M real-data benchmark (download the ml-1m dataset first)
python examples/compare_ml1m.py
```

## Quick Start

```python
import numpy as np
import torch
from hashemb import HashEmbedding

# ── Create the hash embedding table (CPU memory, allocated on demand) ──
emb = HashEmbedding(
    embedding_dim=64,
    capacity=10_000_000,     # initial capacity hint (NOT a hard limit)
    optimizer="adam", lr=0.001,
    block_size=10_000_000,   # allocate one block per 10M slots
)

# ── Training loop ──
for batch in loader:
    keys = batch["feat_ids"]              # (B, S) int64, on GPU/MPS
    out = emb(keys)                       # (B, S, D) auto-transferred to the target device
    loss = F.cross_entropy(dense(out), y)
    loss.backward()                       # gradients accumulate into the C++ grad_buffer
    dense_optim.step()                    # update dense parameters
    emb.step()                            # C++ Adam/SGD update + zero grad_buffer

# ── Save a checkpoint ──
# Save the dense model with torch.save
torch.save({"dense": dense.state_dict()}, "dense.pt")

# Save the hash embedding table to a binary format (streamed per bucket, zero extra memory)
# Optional eviction filters low-frequency / stale features
emb.save("emb.hashemb", min_count=5, max_idle_steps=5000)

# ── Load and resume training ──
dense.load_state_dict(torch.load("dense.pt"))

# Re-create a table with the same spec, then load the binary file
emb = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)
emb.load("emb.hashemb")

# ── Export for serving / inspection ──
# Extract keys and embeddings from a saved checkpoint into NPZ files
# Writes embeddings_bucket_00.npz ... embeddings_bucket_15.npz
from hashemb.utils import export_hashemb_to_npz

export_hashemb_to_npz("emb.hashemb", "embeddings.npz", embedding_dim=64, capacity=10_000_000)

embedding_by_key = {}
for bucket_id in range(16):
    z = np.load(f"embeddings_bucket_{bucket_id:02d}.npz")
    for key, vector in zip(z["keys"], z["embeddings"]):
        embedding_by_key[int(key)] = vector
```

`save()` is for training checkpoints / resume; `export_hashemb_to_npz()` exports from a saved checkpoint and writes only `(hash_id, embedding)`. It writes 16 files, `embeddings_bucket_00.npz` to `embeddings_bucket_15.npz`, one per hash bucket, so the whole table is never materialized at once (avoids OOM). Each bucket file contains only key/vector data and basic metadata; export order is not guaranteed, so consumers should align by `keys`.

## Core Design

### Block-Based On-Demand Storage

Instead of pre-allocating one large `[capacity, D]` contiguous buffer, embeddings use **append-on-demand blocks**:

- Each block holds 10M slots by default (`block_size` configurable)
- New blocks are appended automatically when full — no hard capacity limit
- grad / m / v are allocated on demand the same way (m/v only for Adam)
- Suited to hundreds-of-billions of features / terabyte-scale embeddings

```
hash table:   feat_id → slot_index (auto-grows)
block chain:  [block_0 | block_1 | ... | block_N]  each block = [block_size, D]

lookup:   key[0] → find_or_create → slot_5 → block_0[5]
step():   direct ptr = base_data + slot * D  (single-block fast path, no division)
```

### Auto-Grow Buckets

- Each bucket starts with 25% slack based on the `capacity` hint (load factor ≤ 0.8)
- When full it doubles and rehashes — no hard limit to configure
- No `hash_int64` second hash (feat_id is already uniform; use `key & 0xF` to pick the bucket)

### Unified Single Table

All features share one hash embedding table. `embedding_dim` is global.

```
Table: HashTable
  ├── Bucket 0  │ key_0 │ key_1 │ ... │ key_n │
  ├── Bucket 1  │ key_0 │ key_1 │ ... │ key_n │
  ├── ...
  └── Bucket 15 │ key_0 │ key_1 │ ... │ key_n │
```

### Lookup Semantics (rank-lifting only, never reducing)

| Input shape | Output shape |
|---|---|
| `(B,)` | `(B, D)` |
| `(B, S)` | `(B, S, D)` |
| `(B, F, S)` | `(B, F, S, D)` |

- Sequence structure is preserved — no forward pooling
- Automatic cross-device transfer (CPU lookup → GPU forward → backward)

### Backward Semantics

- Equivalent to PyTorch `nn.Embedding`'s `index_add_` behavior
- Duplicate feat_ids in the same batch have their gradients **summed**
- Accumulated in `autograd.Function.backward()` via the C++ `scatter_add_grad`

> **Note**: PyTorch's `grad_output` may be non-C-contiguous (e.g. from strided ops).
> The Python side calls `.contiguous().numpy()` so C++ reads the correct memory layout.

### Native Optimizers (implemented in C++)

| Optimizer | step update | State |
|---|---|---|
| SGD | `w -= lr * g` | weight + grad |
| Adam | `m = b1*m + (1-b1)g; v = b2*v + (1-b2)g^2; w -= lr * m_hat/(√v̂ + ɛ)` | weight + grad + m + v + t |

- The gradient is **zeroed in place** in the `step()` inner loop, avoiding the full `zero_grad()` memset
- The single-block case uses **direct pointer arithmetic** `base_ptr + slot * D`, avoiding the division in `slot_ptr()`

### Training Pattern (BigModel / per-field tables)

```python
class FeatureEmbedder(torch.nn.Module):
    """One independent HashEmbedding per field (ID spaces may overlap)."""
    def __init__(self, feature_names, dim, capacity):
        for name in feature_names:
            self.__setattr__(f"emb_{name}", HashEmbedding(dim, capacity, ...))

    def forward(self, feat_dict):
        return {name: getattr(self, f"emb_{name}")(feat_dict[name])
                for name in self.feature_names}

    def step(self):
        for name in self.feature_names:
            getattr(self, f"emb_{name}").step()
```

### Key Eviction

As training proceeds the table keeps growing (new feat_ids keep being inserted). **Eviction** periodically prunes low-frequency or stale features to bound memory usage.

#### How it works

The C++ side maintains per-slot statistics in every `step()`:

| Field | Type | Meaning |
|---|---|---|
| `update_count` | uint32 | cumulative number of step() updates for this key |
| `last_step` | uint32 | global_step of the most recent step() update |
| `global_step_` | int64 | global monotonically increasing step counter |

Eviction happens **at save time**: only keys passing the filter are written to the binary file. Loading it back yields a compacted table.

```
train N steps → save(with eviction) → load → continue training
  table grows        keep important keys only    table compacted
```

#### Eviction parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_count` | 0 (disabled) | evict keys with `update_count < min_count` |
| `max_idle_steps` | 0 (disabled) | evict keys with `(global_step - last_step) > max_idle_steps` |
| `combine` | `"and"` | when both conditions are enabled: `"and"` = evict only if both hold (conservative); `"or"` = evict if either holds |

- Only `min_count` (`max_idle_steps=0`): frequency only
- Only `max_idle_steps` (`min_count=0`): idle time only
- Both set to 0: no eviction, equivalent to a plain save

#### Example

```python
import torch
from hashemb import HashEmbedding

# ── Phase 1: train several epochs; the table keeps growing ──
emb = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)

for epoch in range(10):
    for batch in loader:
        out = emb(batch["feat_ids"])
        loss = model(out, batch["labels"])
        loss.backward()
        emb.step()

print(f"keys before eviction: {emb.num_entries:,}")  # e.g. 12,345,678

# ── Phase 2: evict low-frequency features (keep keys updated > 10 times) ──
emb.save("emb_filtered.hashemb", min_count=10)

# ── Phase 3: rebuild a table and load the compacted data ──
emb2 = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)
emb2.load("emb_filtered.hashemb")
print(f"keys after eviction: {emb2.num_entries:,}")  # e.g. 2,345,678

# ── Phase 4: continue training ──
# Stats are preserved, so eviction params still apply on the next save
for epoch in range(10):
    for batch in loader:
        out = emb2(batch["feat_ids"])
        loss.backward()
        emb2.step()

# Evict again: keys updated < 20 times OR idle > 10000 steps
emb2.save("emb_filtered2.hashemb", min_count=20, max_idle_steps=10000, combine="or")
```

#### Typical policies

| Scenario | Settings | Effect |
|---|---|---|
| Daily cleanup | `min_count=5, max_idle_steps=20000` | prune low-frequency and stale noise features |
| Idle-only cleanup | `min_count=0, max_idle_steps=50000` | keep all active features, drop zombie keys |
| Frequency-only cleanup | `min_count=100, max_idle_steps=0` | keep only truly hot features, aggressively prune the long tail |
| Conservative (default) | `min_count=0, max_idle_steps=0` | no eviction, keep every key |

> **Note**: `global_step_` is persisted to the binary file, so `max_idle_steps` spans the save/load boundary. E.g. save after 10k steps, load, train 5k more → `global_step_` = 15k; the kept keys have `last_step` ≤ 10k and won't be wrongly evicted.

#### Without eviction: `state_dict` / `torch.save`

The binary save/load is designed for eviction and large tables. If you don't need eviction you can keep using `state_dict()` + `torch.save` (beware OOM on large tables):

```python
torch.save({
    "hash_emb": emb.state_dict(),
    "dense": dense.state_dict(),
}, "checkpoint.pt")

# Restore
ckpt = torch.load("checkpoint.pt")
emb.load_state_dict(ckpt["hash_emb"])
```

## Project Structure

```
hashemb/
├── setup.py                        # Python package build config
├── CLAUDE.md                       # Claude Code working instructions
├── README.md                       # Project design doc (this file)
├── csrc/                           # C++ core source
│   ├── hash_table.h                # Robin Hood hash map + 16-way sharding + auto-grow
│   ├── hash_table.cpp
│   ├── embedding_table.h           # Block-based storage + batch ops + native optimizers
│   ├── embedding_table.cpp
│   └── pybind_binding.cpp          # pybind11 → Python bridge
├── hashemb/                        # Python package
│   ├── __init__.py
│   ├── core.py                     # PyTorch autograd + nn.Module wrapper
│   └── utils.py
├── examples/
│   ├── demo.py                     # basic usage example
│   └── compare_ml1m.py             # ML-1M nn.Embedding vs HashEmb comparison
└── tests/
    ├── test_basic.py               # C++ core + PyTorch wrapper tests (21 cases)
    ├── test_adam_verify.py         # C++ Adam vs PyTorch manual Adam, step by step
    ├── test_dynamic_expansion.py   # dynamic expansion stress test (capacity=100, 1000 inserts)
    ├── test_pure_torch_vs_hash.py  # nn.Embedding vs HashEmb, step by step (SGD + Adam)
    ├── test_submodel.py            # BigModel architecture test (multi-field + sequence + cross-device)
    └── test_eviction.py            # eviction tests (24 cases)
```

## Performance (ML-1M, 1M ratings, 802K training samples)

Architecture: per-field embedding concat → Linear(2×16, 1) → sigmoid

| Model | AUC | Time/epoch | Relative to nn.Embedding |
|---|---|---|---|
| nn.Embedding (GPU) | 0.7911 | 1.11s | 1.0x |
| HashEmb (CPU) | 0.7910 | **1.22s** | **1.1x** |

> Environment: Apple M2 CPU-only, batch_size=1024, dim=16, Adam LR=0.01, 20 epochs.

## Roadmap

### V1 ✅ (current)
- [x] C++ hash table core (Robin Hood + auto-grow buckets + 16-way sharding)
- [x] Block-based on-demand allocation (no hard capacity limit)
- [x] Native SGD + Adam optimizers (in C++, with m/v/t state)
- [x] pybind11 bindings (numpy / torch tensor interface)
- [x] PyTorch autograd wrapper (correct non-contiguous gradient handling)
- [x] Checkpoint persistence (keys + slots + weight + grad + optimizer state)
- [x] Performance: direct pointer arithmetic + in-place grad zeroing (step() 72ms → 0.09ms)
- [x] ML-1M validation: AUC 0.7910 vs nn.Embedding 0.7911
- [x] Binary save/load (streamed per bucket, zero extra memory, avoids OOM on large tables)
- [x] Key eviction (update_count / last_step stats + hard-rule filtering)

### V2 — Performance (planned)
- [ ] Pinned memory buffer pooling
- [ ] Multi-threaded concurrent lookup
- [ ] Asynchronous prefetch pipeline

### V3 — Advanced (planned)
- [ ] GPU HBM hot-key cache (LRU / K-LRU)
- [ ] Swap-in / swap-out overlapped with compute
- [ ] Multi-GPU All-to-All sharded routing

## References

- [TorchRec](https://github.com/pytorch/torchrec) — PyTorch recommendation library
- [FBGEMM](https://github.com/pytorch/FBGEMM) — Facebook quantization & embedding library

---

# 中文说明

## 概述

HashEmb 是一个基于 CPU 主机内存（DDR）的高性能 Hash Embedding Table 库，专为搜推广系统中大规模稀疏 Embedding 场景设计。

### 核心思路

将大规模稀疏 Embedding Table 存储在 CPU 主机内存（DDR，可达 TB 级），稠密模型（Dense 端）放置在 GPU 显存（HBM），通过 C++ 编写底层高性能 Hash Table，Python（pybind11）封装对接 PyTorch，数据通过 PCIe 在 CPU 与 GPU 之间流转。

```
┌───────────────────────────────────────────────┐
│                   GPU (HBM)                     │
│  ┌─────────────────────────────────────────┐   │
│  │  Dense Network (MLP / Transformer etc.) │   │
│  │  Embedding Tensor (GPU pinned buffer)   │   │
│  └─────────────────────────────────────────┘   │
│            ▲ PCIe (DMA, pinned memory)          │
│            │                                     │
│  ┌────────┴────────────────────────────────┐   │
│  │           CPU (DDR)                       │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │  C++ Hash Embedding Table        │    │   │
│  │  │  • Robin Hood open addressing    │    │   │
│  │  │  • 16-way internal sharding      │    │   │
│  │  │  • Block-based on-demand alloc   │    │   │
│  │  │  • Native SGD / Adam optimizers  │    │   │
│  │  └──────────────────────────────────┘    │   │
│  │            ↕ pybind11                     │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │  PyTorch Wrapper                 │    │   │
│  │  │  • autograd.Function             │    │   │
│  │  │  • nn.Module (HashEmbedding)     │    │   │
│  │  │  • Checkpoint serialisation      │    │   │
│  │  └──────────────────────────────────┘    │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 安装与编译

### 系统要求

| 依赖 | 说明 |
|---|---|
| Python | ≥ 3.8 |
| C++ 编译器 | GCC ≥ 7 / Clang (macOS) |
| PyTorch | ≥ 1.13 |
| pybind11 | 由 setup.py 自动拉取 |

### 构建

```bash
# 从源码安装（自动编译 C++ 扩展）
pip install -e .

# 仅编译 C++ 扩展
python setup.py build_ext --inplace

# 验证安装
python -c "from hashemb import HashEmbedding; print('OK')"

# 跑全部测试
python -m pytest tests/ -v
```

> **注意**：C++ 扩展是平台相关的（`.so` / `.pyd`），**必须在目标机器上重新编译**。

### 修改 C++ 后更新

```bash
python setup.py build_ext --inplace
```

### 运行测试

```bash
# 全部测试（C++ 核心 + PyTorch wrapper）
python -m pytest tests/ -v

# 单个测试文件
python -m pytest tests/test_dynamic_expansion.py -v -s

# 仅 CPU 测试
python -m pytest tests/ -v -k "not cuda"

# ML-1M 真实数据对比（需先下载 ml-1m 数据集）
python examples/compare_ml1m.py
```

## 快速使用

```python
import numpy as np
import torch
from hashemb import HashEmbedding

# ── 创建 hash embedding table（CPU 内存，按需分配） ──────────
emb = HashEmbedding(
    embedding_dim=64,
    capacity=10_000_000,     # hash table 初始容量 hint（非硬上限）
    optimizer="adam", lr=0.001,
    block_size=10_000_000,   # 每 10M 个 slot 分配一块内存
)

# ── 训练循环 ──────────────────────────────────────────────────
for batch in loader:
    keys = batch["feat_ids"]              # (B, S) int64, on GPU/MPS
    out = emb(keys)                       # (B, S, D) 自动传输到目标设备
    loss = F.cross_entropy(dense(out), y)
    loss.backward()                       # 梯度累积到 C++ grad_buffer
    dense_optim.step()                    # 更新 dense 参数
    emb.step()                            # C++ 内 Adam/SGD 更新，清 grad_buffer

# ── 保存 checkpoint ─────────────────────────────────────────
# dense 模型用 torch.save 保存
torch.save({"dense": dense.state_dict()}, "dense.pt")

# hash embedding table 用二进制格式保存（bucket 流式写入，零额外内存）
# 可附带淘汰参数过滤低频/长期未更新特征
emb.save("emb.hashemb", min_count=5, max_idle_steps=5000)

# ── 加载继续训练 ─────────────────────────────────────────────
dense.load_state_dict(torch.load("dense.pt"))

# 重新创建同规格 table，load 二进制文件
emb = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)
emb.load("emb.hashemb")

# ── 导出用于 serving / inspection ───────────────────────────
# 从已保存的 checkpoint 中提取 keys 和 embeddings 到 NPZ 文件
# 写出 embeddings_bucket_00.npz ... embeddings_bucket_15.npz
from hashemb.utils import export_hashemb_to_npz

export_hashemb_to_npz("emb.hashemb", "embeddings.npz", embedding_dim=64, capacity=10_000_000)

embedding_by_key = {}
for bucket_id in range(16):
    z = np.load(f"embeddings_bucket_{bucket_id:02d}.npz")
    for key, vector in zip(z["keys"], z["embeddings"]):
        embedding_by_key[int(key)] = vector
```

`save()` 用于训练 checkpoint / resume；`export_hashemb_to_npz()` 用于从已保存的 checkpoint 中导出 NPZ，只写出 `(hash_id, embedding)`。调用后会按 16 个 hash bucket 分别写出 `embeddings_bucket_00.npz` 到 `embeddings_bucket_15.npz`，避免一次性物化全表导致 OOM。每个 bucket 文件只包含 key/vector 数据和基础 metadata，导出顺序不保证稳定，消费侧应按 `keys` 对齐。

## 核心设计

### 动态按需分配（Block-Based Storage）

Embedding 存储不再一次性预分配 `[capacity, D]` 的大连续 buffer，而是采用**分块按需追加**策略：

- 每块默认 10M 个 slot（`block_size` 可配置）
- slot 用完自动追加新 block，无硬容量上限
- 同样按需分配 grad / m / v（仅 Adam 需要 m/v）
- 适合百亿级特征、TB 级 Embedding 场景

```
hash table:   feat_id → slot_index（自动增长）
block chain:  [block_0 | block_1 | ... | block_N]  每个 block = [block_size, D]

lookup:   key[0] → find_or_create → slot_5 → block_0[5]
step():   direct ptr = base_data + slot * D  （单 block 快路径无除法）
```

### Hash 表自动扩容（Auto-Grow Buckets）

- 每 bucket 初始根据 `capacity` hint 分配 25% slack（load factor ≤ 0.8）
- 满时自动翻倍并 rehash，无需用户设置硬上限
- 不再有 `hash_int64` 二次哈希（feat_id 已均匀分布，直接用 `key & 0xF` 分桶）

### 单表统一架构（Unified Single Table）

所有特征共享一个 Hash Embedding Table。`embedding_dim` 全局统一。

```
Table: HashTable
  ├── Bucket 0  │ key_0 │ key_1 │ ... │ key_n │
  ├── Bucket 1  │ key_0 │ key_1 │ ... │ key_n │
  ├── ...
  └── Bucket 15 │ key_0 │ key_1 │ ... │ key_n │
```

### Lookup 语义（只升维、不降维）

| 输入 shape | 输出 shape |
|---|---|
| `(B,)` | `(B, D)` |
| `(B, S)` | `(B, S, D)` |
| `(B, F, S)` | `(B, F, S, D)` |

- 保留序列结构，不做前向 pooling
- 自动跨设备传输（CPU lookup → GPU forward → backward）

### 反向传播语义

- 等价于 PyTorch `nn.Embedding` 的 `index_add_` 行为
- 同一 Batch 内相同 feat_id 的所有位置**梯度求和**
- `autograd.Function` 的 `backward()` 中通过 C++ `scatter_add_grad` 累积

> **注意**：PyTorch 的 `grad_output` 可能非 C-contiguous（来自 strided 操作），
> Python 端使用 `.contiguous().numpy()` 确保 C++ 读到正确的内存布局。

### 原生优化器（C++ 内实现）

| 优化器 | step 更新公式 | 状态 |
|---|---|---|
| SGD | `w -= lr * g` | 仅需 weight + grad |
| Adam | `m = b1*m + (1-b1)g; v = b2*v + (1-b2)g^2; w -= lr * m_hat/(√v̂ + ɛ)` | weight + grad + m + v + t |

- `step()` 内循环中**原地清零梯度**，省掉 `zero_grad()` 的全量 memset
- 单 block 场景使用**直接指针运算** `base_ptr + slot * D`，避免 `slot_ptr()` 的除法开销

### 训练循环模式（BigModel 架构）

```python
class FeatureEmbedder(torch.nn.Module):
    """每个 field 一个独立的 HashEmbedding（ID 空间可能重叠）。"""
    def __init__(self, feature_names, dim, capacity):
        for name in feature_names:
            self.__setattr__(f"emb_{name}", HashEmbedding(dim, capacity, ...))

    def forward(self, feat_dict):
        return {name: self.emb_{name}(feat_dict[name]) for name in self.feature_names}

    def step(self):
        for name in self.feature_names:
            self.emb_{name}.step()
```


### 特征淘汰（Key Eviction）

随着训练进行，Embedding Table 会持续膨胀（新 feat_id 不断插入）。
可以通过**淘汰（Eviction）**机制周期性地清理低频或长时间未更新的特征，控制内存占用。

#### 工作原理

C++ 端在每个 `step()` 中自动维护每个 slot 的统计信息：

| 字段 | 类型 | 含义 |
|---|---|---|
| `update_count` | uint32 | 该 key 被 step() 更新的累计次数 |
| `last_step` | uint32 | 该 key 最近一次被 step() 更新时的 global_step |
| `global_step_` | int64 | 全局单调递增的训练步数计数器 |

淘汰在 **save 时发生**：只有通过过滤条件的 key 才会写入二进制文件。
然后 `load` 回来即可得到一个精简后的 table。

```
训练 N 步 → save(with eviction) → load → 继续训练
  table 膨胀        仅保留重要特征       table 已精简
```

#### 淘汰参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `min_count` | 0（禁用） | 淘汰 `update_count < min_count` 的 key |
| `max_idle_steps` | 0（禁用） | 淘汰 `(global_step - last_step) > max_idle_steps` 的 key |
| `combine` | `"and"` | 当两个条件都启用时：`"and"` = 同时满足才淘汰（保守），`"or"` = 任一满足即淘汰 |

- 若仅设置 `min_count`（`max_idle_steps=0`）：只看更新频次，不看空闲时间
- 若仅设置 `max_idle_steps`（`min_count=0`）：只看空闲时间，不看更新频次
- 若两者都设为 0：不做淘汰，等价于普通 save

#### 使用示例

```python
import torch
from hashemb import HashEmbedding

# ── 阶段一：训练若干 epoch，table 不断膨胀 ─────────────
emb = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)

for epoch in range(10):
    for batch in loader:
        out = emb(batch["feat_ids"])
        loss = model(out, batch["labels"])
        loss.backward()
        emb.step()

print(f"淘汰前 keys: {emb.num_entries:,}")  # 例: 12,345,678

# ── 阶段二：淘汰低频特征（仅保留更新超过 10 次的 key）───
emb.save("emb_filtered.hashemb", min_count=10)

# ── 阶段三：重建 table，load 精简后的数据 ──────────────
emb2 = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)
emb2.load("emb_filtered.hashemb")
print(f"淘汰后 keys: {emb2.num_entries:,}")  # 例: 2,345,678

# ── 阶段四：继续训练 ──────────────────────────────────
# 淘汰后 stats 仍然保留，
# 后续再次 save 时仍可继续使用淘汰参数
for epoch in range(10):
    for batch in loader:
        out = emb2(batch["feat_ids"])
        loss.backward()
        emb2.step()

# 再次淘汰：更新不足 20 次 OR 空闲超过 10000 步的 key
emb2.save("emb_filtered2.hashemb", min_count=20, max_idle_steps=10000, combine="or")
```

#### 典型淘汰策略

| 场景 | 参数设置 | 效果 |
|---|---|---|
| 每日定时清理 | `min_count=5, max_idle_steps=20000` | 淘汰订阅不足且过期的噪声特征 |
| 仅在长时间未更新时清理 | `min_count=0, max_idle_steps=50000` | 保留所有活跃特征，只清除僵尸 key |
| 仅按频次阈值清理 | `min_count=100, max_idle_steps=0` | 保留真正的热门特征，激进清理长尾 |
| 保守策略（默认） | `min_count=0, max_idle_steps=0` | 不做淘汰，保留全部 key |

> **注意**：淘汰后的 `global_step_` 会保留（写入文件），因此 `max_idle_steps` 的计算
> 会跨越 save/load 边界。例如训练 10k 步后淘汰保存，load 后继续训练 5k 步，
> 此时 `global_step_` = 15k，之前保存时未被淘汰 key 的 `last_step` ≤ 10k，
> 不会被误淘汰。

#### 无需淘汰时仍可用 `state_dict` / `torch.save`

二进制 save/load 是为淘汰和大表场景设计的。若不需要淘汰，也可以继续使用
`state_dict()` + `torch.save` 的方式保存 checkpoint（需注意大表可能 OOM）：

```python
# 小表场景：torch.save 更方便
torch.save({
    "hash_emb": emb.state_dict(),
    "dense": dense.state_dict(),
}, "checkpoint.pt")

# 恢复
ckpt = torch.load("checkpoint.pt")
emb.load_state_dict(ckpt["hash_emb"])
```


### 特征淘汰（Key Eviction）

随着训练进行，Embedding Table 会持续膨胀（新 feat_id 不断插入）。
可以通过**淘汰（Eviction）**机制周期性地清理低频或长时间未更新的特征，控制内存占用。

#### 工作原理

C++ 端在每个 `step()` 中自动维护每个 slot 的统计信息：

| 字段 | 类型 | 含义 |
|---|---|---|
| `update_count` | uint32 | 该 key 被 step() 更新的累计次数 |
| `last_step` | uint32 | 该 key 最近一次被 step() 更新时的 global_step |
| `global_step_` | int64 | 全局单调递增的训练步数计数器 |

淘汰在 **save 时发生**：只有通过过滤条件的 key 才会写入二进制文件。
然后 `load` 回来即可得到一个精简后的 table。

```
训练 N 步 → save(with eviction) → load → 继续训练
  table 膨胀        仅保留重要特征       table 已精简
```

#### 淘汰参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `min_count` | 0（禁用） | 淘汰 `update_count < min_count` 的 key |
| `max_idle_steps` | 0（禁用） | 淘汰 `(global_step - last_step) > max_idle_steps` 的 key |
| `combine` | `"and"` | 当两个条件都启用时：`"and"` = 同时满足才淘汰（保守），`"or"` = 任一满足即淘汰 |

- 若仅设置 `min_count`（`max_idle_steps=0`）：只看更新频次，不看空闲时间
- 若仅设置 `max_idle_steps`（`min_count=0`）：只看空闲时间，不看更新频次
- 若两者都设为 0：不做淘汰，等价于普通 save

#### 使用示例

```python
import torch
from hashemb import HashEmbedding

# 阶段一：训练若干 epoch，table 不断膨胀
emb = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)

for epoch in range(10):
    for batch in loader:
        out = emb(batch["feat_ids"])
        loss = model(out, batch["labels"])
        loss.backward()
        emb.step()

print(f"淘汰前 keys: {emb.num_entries:,}")

# 阶段二：淘汰低频特征（仅保留更新超过 10 次的 key）
emb.save("emb_filtered.hashemb", min_count=10)

# 阶段三：重建 table，load 精简后的数据
emb2 = HashEmbedding(embedding_dim=64, capacity=10_000_000, optimizer="adam", lr=0.001)
emb2.load("emb_filtered.hashemb")
print(f"淘汰后 keys: {emb2.num_entries:,}")

# 阶段四：继续训练
# stats 仍然保留，后续 save 仍可使用淘汰参数
for epoch in range(10):
    for batch in loader:
        out = emb2(batch["feat_ids"])
        loss.backward()
        emb2.step()

# 再次淘汰：更新不足 20 次 OR 空闲超过 10000 步
emb2.save("emb_filtered2.hashemb", min_count=20, max_idle_steps=10000, combine="or")
```

#### 典型淘汰策略

| 场景 | 参数设置 | 效果 |
|---|---|---|
| 每日定时清理 | `min_count=5, max_idle_steps=20000` | 淘汰低频且过期的噪声特征 |
| 仅清理长时间未更新 | `min_count=0, max_idle_steps=50000` | 保留所有活跃特征，只清除僵尸 key |
| 仅按频次清理 | `min_count=100, max_idle_steps=0` | 保留热门特征，激进清理长尾 |
| 保守（默认） | `min_count=0, max_idle_steps=0` | 不做淘汰，保留全部 key |

> **注意**：`global_step_` 会持久化到二进制文件，因此 `max_idle_steps` 的计算会跨越
> save/load 边界。训练 10k 步后淘汰保存，load 再训练 5k 步后 `global_step_` = 15k，
> 之前未被淘汰 key 的 `last_step` ≤ 10k，不会被误淘汰。

#### 无需淘汰时仍可用 `state_dict` / `torch.save`

二进制 save/load 是为淘汰和大表场景设计的。若不需要淘汰，也可以继续使用
`state_dict()` + `torch.save` 的方式保存 checkpoint：

```python
torch.save({
    "hash_emb": emb.state_dict(),
    "dense": dense.state_dict(),
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt")
emb.load_state_dict(ckpt["hash_emb"])
```

## 项目结构

```
hashemb/
├── setup.py                        # Python package build config
├── CLAUDE.md                       # Claude Code 工作指示
├── README.md                       # 项目设计文档（本文）
├── csrc/                           # C++ 核心源码
│   ├── hash_table.h                # Robin Hood Hash Map + 16-way sharding + auto-grow
│   ├── hash_table.cpp
│   ├── embedding_table.h           # Block-based 存储 + 批量操作 + 原生优化器
│   ├── embedding_table.cpp
│   └── pybind_binding.cpp          # pybind11 → Python 桥接
├── hashemb/                        # Python 包
│   ├── __init__.py
│   ├── core.py                     # PyTorch autograd + nn.Module 封装
│   └── utils.py
├── examples/
│   ├── demo.py                     # 基础使用示例
│   └── compare_ml1m.py             # ML-1M nn.Embedding vs HashEmb 对比
└── tests/
    ├── test_basic.py               # C++ 核心 + PyTorch wrapper 基础测试（21 个用例）
    ├── test_adam_verify.py         # C++ Adam 与 PyTorch manual Adam 逐步校验
    ├── test_dynamic_expansion.py   # 动态扩容压力测试（容量=100, 插入1000）
    ├── test_pure_torch_vs_hash.py  # nn.Embedding 与 HashEmb 逐步对比（SGD + Adam）
    ├── test_submodel.py            # BigModel 架构测试（多 field + 序列特征 + 跨设备）
    ├── test_eviction.py            # 淘汰机制测试（24 个用例）
    ├── test_eviction.py            # 淘汰机制测试（24 个用例）
```

## 性能基准（ML-1M, 100 万评分, 802K 训练样本）

Architecture: per-field embedding concat → Linear(2×16, 1) → sigmoid

| 模型 | AUC | 时间/epoch | 相对 nn.Embedding |
|---|---|---|---|
| nn.Embedding (GPU) | 0.7911 | 1.11s | 1.0x |
| HashEmb (CPU) | 0.7910 | **1.22s** | **1.1x** |

> 测试环境：Apple M2 CPU-only，batch_size=1024，dim=16，Adam LR=0.01，20 epochs。

## 开发路线

### V1 ✅（当前阶段）
- [x] C++ Hash Table 核心（Robin Hood + auto-grow bucket + 16-way sharding）
- [x] Block-based 动态按需分配（无硬容量上限）
- [x] 原生 SGD + Adam 优化器（C++ 内实现，含 m/v/t 状态）
- [x] pybind11 绑定（numpy / torch tensor 接口）
- [x] PyTorch autograd 封装（正确非连续梯度处理）
- [x] Checkpoint 持久化（keys + slots + weight + grad + optimizer state）
- [x] 性能优化：直接指针运算 + 原地清零 grad（step() 从 72ms → 0.09ms）
- [x] ML-1M 真实数据验证：AUC 0.7910 vs nn.Embedding 0.7911

- [x] 二进制 save/load（bucket 流式写入，零额外内存，避免大表 OOM）
- [x] Key Eviction 淘汰机制（update_count / last_step 统计 + 硬规则过滤）
- [x] 二进制 save/load（bucket 流式写入，零额外内存，避免大表 OOM）
- [x] Key Eviction 淘汰机制（update_count / last_step 统计 + 硬规则过滤）
### V2 —— 性能优化（待开始）
- [ ] Pinned memory buffer 池化
- [ ] 多线程并发 lookup
- [ ] 异步预取流水线

### V3 —— 进阶（待开始）
- [ ] GPU HBM 热点 Cache（LRU/K-LRU）
- [ ] Swap-in / Swap-out 与计算 overlap
- [ ] 多卡 All-to-All 分片路由

## 参考实现

- [TorchRec](https://github.com/pytorch/torchrec) - PyTorch 推荐系统库
- [FBGEMM](https://github.com/pytorch/FBGGEMM) - Facebook 量化与嵌入库

