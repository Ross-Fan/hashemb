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
│  │  │  • Pre-allocated memory pool     │    │   │
│  │  │  • Batch lookup / batch update   │    │   │
│  │  └──────────────────────────────────┘    │   │
│  │            ↕ pybind11                     │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │  PyTorch Wrapper                 │    │   │
│  │  │  • autograd.Function             │    │   │
│  │  │  • nn.Module (HashEmbedding)     │    │   │
│  │  │  • Pinned memory transfer        │    │   │
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
| PyTorch | ≥ 1.13（CUDA / MPS 后端均可，自动检测） |
| pybind11 | 由 setup.py 自动拉取 |

### 在 GPU (Linux) 或 M1 Mac 上构建

```bash
# 从源码安装（自动编译 C++ 扩展）
pip install -e .

# 验证安装
python -c "from hashemb import HashEmbedding; print('OK')"

# 跑测试
python -m pytest tests/ -v
```

> **注意**：C++ 扩展是平台相关的（`.so` / `.pyd`）。**必须在目标机器上重新编译**。
> `.gitignore` 已排除所有编译产物，不会提交到仓库。

### 仅编译 C++ 扩展（不安装 Python 包）

```bash
python setup.py build_ext --inplace
```

### 修改 C++ 代码后的更新

每次修改 `csrc/` 下的 C++ 文件后，重新编译即可生效：

```bash
python setup.py build_ext --inplace
```

### 测试

```bash
# 全部测试（C++ 核心 + PyTorch wrapper）
python -m pytest tests/ -v

# 仅 C++ 核心测试（无需 PyTorch）
python -m pytest tests/ -v -k "TestCpp"

# 跳过加速器测试（纯 CPU）
python -m pytest tests/ -v -k "not backward_gradient_sum and not training_step"

# 运行 demo
python examples/demo.py
```

### 快速使用

```python
import torch
from hashemb import HashEmbedding

# ── 创建 hash embedding table（CPU 内存） ──────────────────────────
emb = HashEmbedding(
    embedding_dim=64, capacity=10_000_000,
    optimizer="adam", lr=0.001,
)

# ── 训练循环 ──────────────────────────────────────────────────────
for batch in loader:
    keys = batch["feat_ids"]              # (B, S) int64, on GPU/MPS
    out = emb(keys)                       # (B, S, D) on GPU/MPS
    loss = F.cross_entropy(dense(out), y)
    loss.backward()                       # 梯度累计到 C++ grad_buffer
    dense_optim.step()                    # 更新 dense 参数
    emb.step()                            # C++ 内 Adam 更新，清 grad_buffer

# ── 保存 checkpoint ─────────────────────────────────────────────
torch.save({
    "dense":     dense.state_dict(),
    "hash_emb":  emb.state_dict(),        # keys, slots, weight, grad, m, v, t
    "optimizer": dense_optim.state_dict(),
}, "checkpoint.pt")

# ── 加载继续训练 ─────────────────────────────────────────────────
ckpt = torch.load("checkpoint.pt")
emb.load_state_dict(ckpt["hash_emb"])     # 恢复 hash 映射 + 权重 + 优化器状态
```

## 核心设计

### 单表统一架构（Unified Single Table）

所有特征共享 **一个** Hash Embedding Table。特征 ID 不区分 field，统一通过一个哈希表存取。

```
Table: UnifiedHashEmbeddingTable
  ├── Bucket 0  │ key_0 │ key_1 │ ... │ key_n │
  ├── Bucket 1  │ key_0 │ key_1 │ ... │ key_n │
  ├── ...
  └── Bucket 15 │ key_0 │ key_1 │ ... │ key_n │

所有 bucket 共享一个连续的 Embedding Buffer:
[f0_emb│f1_emb│f2_emb│...│fN_emb]  每个 embedding = embedding_dim * float
```

- `embedding_dim`: 全局统一，所有特征共享同一维度
- `capacity`: 预分配的 embedding 槽位数，用于计算内存占用

### Lookup 语义（只升维、不降维）

| 输入 shape | 输出 shape | 说明 |
|---|---|---|
| `(B,)` | `(B, D)` | 批量样本 |
| `(B, S)` | `(B, S, D)` | 批量 + 序列 |
| `(B, F, S)` | `(B, F, S, D)` | 批量 + 多域 + 序列 |
| `(B, F, S, ...)` | `(B, F, S, ..., D)` | 通用升维 |

- **不丢失序列结构**: 保留每个 position 的独立 embedding，供上层 Transformer / Attention 使用
- **不做前向 Pooling**: 不在 lookup 层做 mean/sum 等序列聚合操作

### 反向传播语义

- 等价于 PyTorch `nn.Embedding` 的 `index_add_` 行为
- 同一 Batch 内，同一个 feat_id 出现在多个位置（同一特征序列内重复，或不同特征的序列内同时出现），backward 时对该 feat_id 所有位置的梯度进行 **求和（sum）** 后更新
- 示例：
  ```
  前向: 特征A（序列特征）= [1, 3258091, 5, 3258091, 9]
        特征B（上下文特征）= [3258091, 42, 7]

  对 id=3258091:
    grad = grad_A_pos1 + grad_A_pos3 + grad_B_pos0  # 求和
    embedding[3258091] -= lr * grad                  # SGD 更新
  ```

### 内部 16-Way Sharding

Hash Table 内部固定分为 16 个 Bucket（分桶），每个 Bucket 是一个独立的开放寻址哈希表：

- 分桶依据: `hash(feat_id) & 0xF` 取低 4 bit 确定 bucket
- 每个 Bucket 内部使用 Robin Hood 哈希（开放寻址，线性探测）
- 降低锁粒度：每个 Bucket 一把锁，支持 16 路并发

## 项目结构

```
hashemb/
├── setup.py                        # Python package build config
├── CLAUDE.md                       # Claude Code 工作指示
├── README.md                       # 项目设计文档（本文）
├── csrc/                           # C++ 核心源码
│   ├── hash_table.h                # Robin Hood Hash Map 实现
│   ├── hash_table.cpp
│   ├── embedding_table.h           # Embedding 向量存储与批量操作
│   ├── embedding_table.cpp
│   └── pybind_binding.cpp          # pybind11 → Python 桥接
├── hashemb/                        # Python 包
│   ├── __init__.py
│   ├── core.py                     # PyTorch autograd + nn.Module
│   └── utils.py                    # 工具函数（pinned buffer 等）
├── examples/
│   └── demo.py                     # 使用示例
└── tests/
    └── test_basic.py               # 基础测试
```

## 开发路线

### V1 —— 基础链路打通 ✅（当前阶段）
- [x] C++ Hash Table 核心（Robin Hood 开放寻址 + 16-bucket sharding）
- [x] C++ Embedding Table 引擎（连续 buffer + batch lookup / batch update）
- [x] pybind11 绑定（支持 numpy / `torch::Tensor` 接口）
- [x] PyTorch `autograd.Function` 封装（CPU lookup → GPU 传输 → backward gradient sum）
- [x] `nn.Module` 层 `HashEmbedding(dim, capacity)` 接口
- [x] 测试：正确性验证（梯度累积、rank 升维、序列特征）

### V2 —— 性能优化（待开始）
- [ ] Pinned memory buffer 池化，零拷贝 torch 转换
- [ ] 多线程并发 lookup（每个 bucket 独立线程）
- [ ] 异步预取 (prefetch next batch while GPU computes)
- [ ] Benchmark: PCIe 传输基准性能

### V3 —— 进阶（待开始）
- [ ] GPU HBM 作为热点 Cache（LRU/K-LRU）
- [ ] Swap-in / Swap-out 流水线与计算 overlap
- [ ] 多卡训练 + All-to-All 分片路由

## 参考实现

- [TorchRec](https://github.com/pytorch/torchrec) - PyTorch 推荐系统库
- [FBGEMM](https://github.com/pytorch/FBGGEMM) - Facebook 量化与嵌入库
- [Ascend RecSDK](https://www.hiascend.com/) - 昇腾推荐系统 SDK
