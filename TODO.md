# HashEmb Scaling TODO

## 10 亿 feat_id 空间扩展问题分析

分析基准：capacity=10亿（ID 哈希区间），实际插入 1 亿~10 亿条，dim=16，Adam。

---

## P0 — 必须修复

### [ ] int32_t slot 索引溢出

- 位置：`hash_table.h:16`（`Bucket::slot_indices` 类型 `int32_t*`）、`hash_table.cpp:173`（`static_cast<int32_t>(num_entries_)`）
- 问题：slot 索引上限 2,147,483,647，超过后溢出到负数。`lookup()` 中 `if (slot < 0)` 会当作 padding 丢弃数据。
- 修复：全链路改为 `int64_t`（`slot_indices`、`dists`、`num_entries_`、`slot_ptr` 参数、pybind 接口）

### [x] step() O(n) 全表遍历 ✅ 已修复

- 位置：`embedding_table.cpp:161`
- 问题：每次 step 遍历全部已插入条目。1 亿条 × dim=16 = 16 亿 float，CPU 上 ~0.5~3s/step。10 亿条 → 10~30s/step。
- **修复方案**：dirty slot tracking
  - `scatter_add_grad()` 中标记被 touch 的 slot 为 dirty
  - `step()` 只遍历 `dirty_slots_`（O(batch_size)），不清零不再 O(num_entries)
  - `zero_grad()` 也只清零 dirty slot 的梯度
  - **语义**：sparse Adam——只更新有梯度的 slot，无梯度 slot 的 m/v 不衰减（同 PyTorch sparse Adam）
  - 37 个测试全部通过，ML-1M AUC = 0.7911（与 nn.Embedding 完全一致）
  - 速度：ML-1M 上 1.15s/epoch（优化前 1.22s，nn.Embedding 1.07s）

---

## P1 — 高优先级

### [ ] Hash bucket 预分配浪费

- 位置：`hash_table.cpp:124-133`
- 问题：capacity=10 亿时，每个 bucket 预分配 128M slot（16 字节/slot），16 buckets 共 32GB，其中 ~16GB 空闲 slack。
- 修复思路：
  - 缩小 slack（从 25% 降到 10%）
  - 使用更激进的自适应增长（每次 +25% 而不是翻倍）
  - lazy bucket 分配（用到了再分配 bucket）

### [/] state_dict() 双倍内存尖峰 ✅ 已用二进制 save/load 替代

- 位置：`pybind_binding.cpp:102-138`
- 问题：C++ vector 存一份 + numpy array copy 一份。1 亿条峰值 = 53GB（驻留 26GB 的两倍）。10 亿条 = 512GB 峰值。
- **决策**：不修 `state_dict()` 了。改为 C++ 端二进制文件直接读写，零额外内存
  - `EmbeddingTable::save()` / `load()`：按 bucket 顺序遍历，数据直接从 block buffer 读到文件
  - 文件格式：header + 16 个 bucket section，每条记录存 key + slot + weight + grad + m + v
  - `load()` 用 `find_or_create` 重建 key→slot 映射，数据直接 fread 到 block buffer
  - Python 端：`emb.save(path)` / `emb.load(path)`，不需要走 `torch.save`
  - 两阶段训练中：embedding 表独立保存，dense 模型照常 `torch.save`

### [x] Bucket::grow() 无异常安全 ✅ 已修复

- 位置：`hash_table.cpp:57-101`
- 问题：三连 `new[]` 中任何一个失败，之前成功的分配永久泄露。
- **修复方案**：Bucket 内 `int64_t*`/`int32_t*` 裸数组改为 `std::vector<T>`
  - `allocate()`：`assign()` 替代 `new[]` + `memset`
  - `grow()`：局部 vector → rehash → `std::move` 到成员（析构自动清理）
  - 同时修了 `next_pow2` 的 int32_t 溢出（加 `std::overflow_error` 检查）
  - 37 个测试全部通过

---

## P2 — 可优化

### [ ] 多 block 时 slot_ptr() 除法开销

- 位置：`embedding_table.h:49-53`
- 问题：多 block 后每次访问 slot_ptr 做除法和取模。step() 里 4 次调用 × 1 亿条 = ~40ms 额外除法开销。
- 修复：预计算 slot 到 block_id + offset 的偏移数组，但会牺牲内存

### [ ] next_pow2 int32_t 溢出

- 位置：`hash_table.h:67-76`
- 问题：bucket 多次 grow 后 capacity*2 超过 int32_t 上限，next_pow2 返回 0 或负数。
- 修复：改为 `uint32_t` 或 `int64_t`，加溢出检查

---

## 扩展阅读

### 核心矛盾

不是内存，是**时间**：
- 1 亿条时 step() 已 ~0.5s，10 亿条时 ~10s+
- 训练一个 epoch = batch 数 × step() 时间 → 不可接受
- 解决方案：GPU 端做 cache layer（LRU-K），CPU 端 step 异步后台进行

### 内存下限（物理上省不掉）

满 10 亿条 × dim=16，Adam：

| 数据 | 占用 |
|---|---|
| Weight | 64 GB |
| Grad | 64 GB |
| m | 64 GB |
| v | 64 GB |
| **小计** | **256 GB** |

### 建议规模限值

| 规模 | 可行性 | 瓶颈 |
|---|---|---|
| ≤ 1 千万条 | 完全 OK | 无 |
| ≤ 1 亿条 | 可用但仍需优化 | step() ~0.5s，state_dict 尖峰 |
| ≤ 10 亿条 | 需要大型改造 | int64 slot, 稀疏 step, 零拷贝导出 |
| > 10 亿条 | 需要 GPU cache | 纯 CPU step 不可行 |


---
  一、性能问题

  1. step() O(n) 全表遍历 — 最严重

  // embedding_table.cpp:161
  void EmbeddingTable::step() {
    int64_t n = hash_table_.num_entries();
    for (int64_t slot = 0; slot < n; ++slot) {
      // 每个 slot 的 D 个维度都要计算
    }
  }

  每次 step 必须遍历全部已插入条目。假设实际插了 1 亿条、dim=16：
  - 每 step 处理 1 亿 × 16 = 16 亿个 float
  - Adam 下还有 m/v 的读写
  - 粗略估算：CPU 上 500ms ~ 几秒 / step
  - 如果插到 10 亿条，每 step 处理 160 亿 float → 10~30 秒 / step

  2. 多 block 时 slot_ptr() 除法开销

  // embedding_table.h:49-53
  inline float* slot_ptr(const std::vector<Block>& blocks, int64_t slot_id, int32_t D, int64_t block_size) {
    int64_t block_id = slot_id / block_size;   // 除法
    int64_t offset = slot_id % block_size;     // 取模
    return blocks[block_id].data + offset * D;
  }

  单 block 时走了快路径（base_ptr + slot * D），但多 block 后每次访问都要做除法和取模。step() 里对 weight/grad/m/v 各调用一次，每次 step 就是 4 次除法/取模 × 1 亿条 = 约 40ms
  的除法开销。

  ---
  二、内存问题

  3. Hash bucket 预分配浪费

  按 capacity=10亿 构造时：

  // hash_table.cpp:124-133
  per_bucket = next_pow2((10亿/16) × 1.25)
             = next_pow2(78,125,000)
             = 134,217,728  // 128M slot

  每个 bucket: 128M × (8+4+4) = 2GB
  16 buckets: 32GB   ← 建表就占，还没插入任何数据

  其中 ~16GB 是空闲 slack（load factor 0.8 的代价）。如果实际上只插了 1 亿条，这 32GB 就白白占着。

  4. state_dict() 双倍内存尖峰

  // pybind_binding.cpp:102-138
  void state_dict() {
    // 第一步：C++ vector 存一份
    std::vector<float> weight, grad, m, v;  // 共 26.8GB (1亿条×dim=16)

    // 第二步：numpy array 再 copy 一份
    auto np_weight = make_float_array(n, D);  // 又 26.8GB
    std::memcpy(np_weight.request().ptr, weight.data(), ...);
  }

  ┌─────────┬─────────────────────────┬──────────────────────────────┐
  │  场景   │        实际驻留         │       state_dict 峰值        │
  ├─────────┼─────────────────────────┼──────────────────────────────┤
  │ 1 亿条  │ ~26GB (weight+grad+m+v) │ ~53GB（vector + numpy 双份） │
  ├─────────┼─────────────────────────┼──────────────────────────────┤
  │ 10 亿条 │ ~256GB                  │ ~512GB（远超常见 DDR）       │
  └─────────┴─────────────────────────┴──────────────────────────────┘

  这也意味着 torch.save() 序列化 checkpoint 时，内存压力会翻倍。

  5. 大 block 连续分配风险

  每个 block 默认 10M slot × dim=16 × 4B = 640MB，4 个 buffer（weight/grad/m/v）同时分配就是 2.56GB/次。长期运行后内存碎片化，posix_memalign(64) 可能失败。

  ---
  三、可靠性问题

  6. int32_t slot 索引溢出

  // hash_table.h:16
  int32_t* slot_indices;  // bucket 内 slot 索引是 int32_t

  // hash_table.cpp:173
  int32_t new_slot = static_cast<int32_t>(num_entries_.fetch_add(1));

  slot 索引用 int32_t，上限 2,147,483,647。超过 21.4 亿条时溢出到负数（hash_table.cpp:66 还有个 if (slot < 0) 被当作 padding 处理，会直接丢弃数据）。

  7. grow() 无异常安全

  // hash_table.cpp:57-101
  void Bucket::grow() {
    auto* new_keys  = new int64_t[new_cap];     // ①
    auto* new_slots = new int32_t[new_cap];     // ②
    auto* new_dists = new int32_t[new_cap];     // ③
    // ... rehash ...
    delete[] keys;      // ④
    delete[] slots;     // ⑤
    delete[] dists;     // ⑥
    keys = new_keys;    // ⑦
  }

  如果 ① 成功但 ② 抛 bad_alloc（大内存时完全可能），new_keys 永久泄露。三步 new 中任何一步失败都导致之前的分配泄露。

  8. next_pow2 int32_t 溢出

  // hash_table.h:67-76
  inline int32_t next_pow2(int32_t x) { ... }

  bucket grow 几次后 capacity*2 可能超过 int32_t 最大值（2.1B），此时 next_pow2 返回 0 或负数，后续分配灾难性失败。

  ---
  四、大内存管理

  9. 无渐进式 checkpoint

  当前 state_dict 必须把所有数据从 block buffer 拷贝到 vector<float>，再拷贝到 numpy。没有任何"分批导出"或"流式写入磁盘"的机制。256GB 的 embedding 做一次 checkpoint，内存瞬间需要
   ~512GB。

  10. 无 OOM 优雅降级

  所有大分配都是裸 new[] 或 posix_memalign，bad_alloc 直接崩进程。没有 std::nothrow、没有 retry、没有释放缓存重试。

  ---
  总结优先级

  ┌────────┬─────────────────────────┬────────────────────────┬───────────────────────────────┐
  │ 优先级 │          问题           │        影响范围        │           修复难度            │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P0     │ int32_t slot 溢出       │ >2.1B 条后静默丢数据   │ 中（改 int64_t 涉及多处）     │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P0     │ step() O(n) 全遍历      │ 1 亿条时 0.5~3s/step   │ 难（需要稀疏梯度或热更新）    │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P1     │ bucket 预分配 32GB 浪费 │ 建表即占               │ 中（缩小 slack 或 lazy 分配） │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P1     │ state_dict 双倍内存     │ checkpoint 时峰值翻倍  │ 中（零拷贝导出）              │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P1     │ grow() 异常安全         │ 内存紧张时泄露         │ 低（RAII wrapper）            │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P2     │ 多 block 除法开销       │ 1 亿条 ~40ms/step 额外 │ 低（预计算偏移数组）          │
  ├────────┼─────────────────────────┼────────────────────────┼───────────────────────────────┤
  │ P2     │ next_pow2 溢出          │ 极端场景               │ 低（改 uint32_t/加检查）      │
  └────────┴─────────────────────────┴────────────────────────┴───────────────────────────────┘

  最核心的矛盾：1 亿条以上时 step() O(n) 全遍历在 CPU 上会慢到不可用，这才是真正限制规模的问题——不是内存，是时间。内存只是"贵"，时间是"没法等"。

✻ Sautéed for 45s
