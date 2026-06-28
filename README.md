# HashEmb — CPU Host-Memory Hash Embedding Table for PyTorch

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
torch.save({
    "dense":     dense.state_dict(),
    "hash_emb":  emb.state_dict(),        # keys + slots + weight + grad + m/v/t
    "optimizer": dense_optim.state_dict(),
}, "checkpoint.pt")

# ── 加载继续训练 ─────────────────────────────────────────────
ckpt = torch.load("checkpoint.pt")
emb.load_state_dict(ckpt["hash_emb"])
```

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
    └── test_submodel.py            # BigModel 架构测试（多 field + 序列特征 + 跨设备）
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
