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
# 只包含 hash ID 和 float32 embedding vector，不包含 grad / Adam m,v / slots / stats
emb.export("embeddings.npz")

z = np.load("embeddings.npz")
keys = z["keys"]
vectors = z["embeddings"]
```

`save()` 用于训练 checkpoint / resume；`export()` 用于外部消费，只写出 `(hash_id, embedding)`，导出顺序不保证稳定，消费侧应按 `keys` 对齐。

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

