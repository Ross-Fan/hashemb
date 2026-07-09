#include "embedding_table.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace hashemb {

namespace {

/// Portable barrier using std::mutex + std::condition_variable.
class Barrier {
 public:
  Barrier() : threshold_(0), count_(0), gen_(0) {}

  explicit Barrier(int count) : threshold_(count), count_(count), gen_(0) {}

  void init(int count) {
    threshold_ = count;
    count_ = count;
    gen_ = 0;
  }

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    int local_gen = gen_;
    if (--count_ == 0) {
      count_ = threshold_;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, local_gen] { return gen_ != local_gen; });
    }
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int threshold_;
  int count_;
  int gen_;
};

/// Persistent thread pool using portable Barrier for synchronization.
/// Workers are created once (lazy singleton) and reused across all calls.
class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  /// Run fn(i) for i in [0, n) in parallel (master participates too).
  /// Falls back to sequential when work is too small.
  template <typename Func>
  void parallel_for(size_t n, Func&& fn) {
    // Skip threading for tiny work: overhead > benefit.
    if (nworkers_ == 0 || n < static_cast<size_t>((nworkers_ + 1) * 4)) {
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }

    work_fn_ = [&fn](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) fn(i);
    };
    total_ = n;

    // Phase 1: release workers (master + nworkers_)
    barrier_.arrive_and_wait();

    // Master does its share
    do_chunk(nworkers_);  // master = last "worker"

    // Phase 2: wait for all workers to finish
    barrier_.arrive_and_wait();
  }

 private:
  ThreadPool() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    nworkers_ = hw > 1 ? hw - 1 : 0;  // leave 1 core for master
    if (nworkers_ > 0) {
      barrier_.init(nworkers_ + 1);
      for (int i = 0; i < nworkers_; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, i);
      }
    }
  }

  ~ThreadPool() {
    if (nworkers_ == 0) return;
    stop_.store(true, std::memory_order_release);
    barrier_.arrive_and_wait();   // wake workers from Phase 1
    barrier_.arrive_and_wait();   // let them exit via Phase 2
    for (auto& w : workers_) w.join();
  }

  void do_chunk(int tid) {
    size_t chunk = total_ / (nworkers_ + 1);
    size_t rem = total_ % (nworkers_ + 1);
    size_t start = static_cast<size_t>(tid) * chunk + std::min<size_t>(static_cast<size_t>(tid), rem);
    size_t end = start + chunk + (static_cast<size_t>(tid) < rem ? 1 : 0);
    if (start < end) work_fn_(start, end);
  }

  void worker_loop(int tid) {
    while (true) {
      // Phase 1: wait for work
      barrier_.arrive_and_wait();

      if (stop_.load(std::memory_order_acquire)) {
        barrier_.arrive_and_wait();  // Phase 2: keep barrier balanced
        break;
      }

      do_chunk(tid);

      // Phase 2: signal done
      barrier_.arrive_and_wait();
    }
  }

  int nworkers_ = 0;
  std::vector<std::thread> workers_;
  Barrier barrier_;
  std::atomic<bool> stop_{false};

  // Work descriptor (set by master before releasing workers)
  std::function<void(size_t, size_t)> work_fn_;
  size_t total_ = 0;
};

}  // namespace

// ===========================================================================
// Construction / destruction
// ===========================================================================

EmbeddingTable::EmbeddingTable(int64_t initial_capacity, int32_t embedding_dim,
                               const OptimizerConfig& opt_cfg,
                               int64_t block_size,
                               float initial_scale)
    : initial_capacity_(initial_capacity),
      embedding_dim_(embedding_dim),
      block_size_(block_size),
      opt_cfg_(opt_cfg),
      hash_table_(initial_capacity),
      initial_scale_(initial_scale) {
  if (initial_capacity <= 0 || embedding_dim <= 0) {
    throw std::invalid_argument("capacity and embedding_dim must be positive");
  }
  // No pre-allocation of embedding buffers — blocks are allocated on demand
  // via ensure_slot() during lookup_and_gather() or load_state_dict_arrays().
}

EmbeddingTable::~EmbeddingTable() {
  for (auto& b : emb_blocks_) b.deallocate();
  for (auto& b : grad_blocks_) b.deallocate();
  for (auto& b : m_blocks_) b.deallocate();
  for (auto& b : v_blocks_) b.deallocate();
}

// ===========================================================================
// Block management
// ===========================================================================

void EmbeddingTable::ensure_slot(int64_t slot_id) {
  int64_t needed = (slot_id / block_size_) + 1;
  while (static_cast<int64_t>(emb_blocks_.size()) < needed) {
    int64_t block_id = static_cast<int64_t>(emb_blocks_.size());
    emb_blocks_.emplace_back();
    emb_blocks_.back().allocate(block_size_, embedding_dim_);
    grad_blocks_.emplace_back();
    grad_blocks_.back().allocate(block_size_, embedding_dim_);
    if (opt_cfg_.type == OptimizerConfig::ADAM) {
      m_blocks_.emplace_back();
      m_blocks_.back().allocate(block_size_, embedding_dim_);
      v_blocks_.emplace_back();
      v_blocks_.back().allocate(block_size_, embedding_dim_);
    }
    // Randomly initialize embedding weights if initial_scale > 0.
    // Uses Box-Muller transform to produce normal(0, initial_scale)
    // distribution, matching nn.init.normal_(std=initial_scale).
    // grad/m/v blocks are always zero-initialized.
    if (initial_scale_ > 0.0f) {
      int64_t n = block_size_ * embedding_dim_;
      float* data = emb_blocks_.back().data;
      // Simple LCG: different seed per block for de-correlation.
      uint64_t state = 0x9d2c5680ULL + static_cast<uint64_t>(block_id) * 0x517cc1b7ULL;
      constexpr float kPi = 3.141592653589793f;
      for (int64_t i = 0; i < n; i += 2) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u1 = static_cast<float>((state >> 32) & 0xFFFFFFFF) / 4294967296.0f;
        if (u1 < 1e-10f) u1 = 1e-10f;  // guard against log(0)
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = static_cast<float>((state >> 32) & 0xFFFFFFFF) / 4294967296.0f;
        // Box-Muller: two uniforms → two independent normal(0,1)
        float r = std::sqrt(-2.0f * std::log(u1));
        data[i] = r * std::cos(kPi * 2.0f * u2) * initial_scale_;
        if (i + 1 < n) {
          data[i + 1] = r * std::sin(kPi * 2.0f * u2) * initial_scale_;
        }
      }
    }
  }
  // Grow slot_dirty_ to cover the new block's slots.
  int64_t total_slots = needed * block_size_;
  if (slot_dirty_.size() < static_cast<size_t>(total_slots)) {
    slot_dirty_.resize(total_slots, false);
  }
}

// ===========================================================================
// Lookup
// ===========================================================================

void EmbeddingTable::lookup(const int32_t* slot_indices, float* output,
                            int64_t n) const {
  int32_t D = embedding_dim_;
  int64_t bs = block_size_;

  ThreadPool::instance().parallel_for(static_cast<size_t>(n), [&](size_t i) {
    int32_t slot = slot_indices[i];
    if (slot < 0) {
      std::memset(output + i * static_cast<int64_t>(D), 0, sizeof(float) * static_cast<size_t>(D));
    } else {
      std::memcpy(output + i * static_cast<int64_t>(D), slot_ptr(emb_blocks_, slot, D, bs),
                  sizeof(float) * static_cast<size_t>(D));
    }
  });
}

void EmbeddingTable::lookup_and_gather(const int64_t* keys, float* output,
                                       int32_t* slot_indices, int64_t n) {
  hash_table_.find_or_create(keys, slot_indices, n);

  // Ensure backing blocks for the maximum slot ID assigned.
  int32_t max_slot = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (slot_indices[i] > max_slot) max_slot = slot_indices[i];
  }
  if (max_slot >= 0) ensure_slot(max_slot);

  lookup(slot_indices, output, n);
}

// ===========================================================================
// Gradient accumulation
// ===========================================================================

void EmbeddingTable::scatter_add_grad(const int32_t* slot_indices,
                                      const float* grads,
                                      int64_t n) {
  int32_t D = embedding_dim_;

  // Ensure blocks for all referenced slots.
  int32_t max_slot = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (slot_indices[i] > max_slot) max_slot = slot_indices[i];
  }
  if (max_slot >= 0) ensure_slot(max_slot);

  // Collect unique valid slots (dedup).  Reserve a reasonable lower bound.
  std::unordered_map<int32_t, bool> seen_slots;
  seen_slots.reserve(static_cast<size_t>(std::min(n, static_cast<int64_t>(200000))));

  // Direct accumulation into grad_blocks_ (no sort, no heap idx array).
  // Each occurrence of the same slot is added into the same grad buffer,
  // naturally implementing "scatter add".
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slot_indices[i];
    if (slot < 0) continue;

    seen_slots[slot] = true;

    float* grad_dst = slot_ptr(grad_blocks_, slot, D, block_size_);
    const float* g = grads + i * D;
    for (int32_t d = 0; d < D; ++d) grad_dst[d] += g[d];
  }

  // Mark dirty slots for step().
  for (const auto& p : seen_slots) {
    int32_t slot = p.first;
    if (!slot_dirty_[slot]) {
      slot_dirty_[slot] = true;
      dirty_slots_.push_back(slot);
    }
  }
}

// ===========================================================================
// Optimizer step
// ===========================================================================

void EmbeddingTable::zero_grad() {
  int32_t D = embedding_dim_;

  // Zero gradient memory for dirty slots only.
  for (int32_t slot : dirty_slots_) {
    float* g = slot_ptr(grad_blocks_, slot, D, block_size_);
    std::memset(g, 0, static_cast<size_t>(D) * sizeof(float));
    slot_dirty_[slot] = false;
  }
  dirty_slots_.clear();
}

void EmbeddingTable::step() {
  int32_t D = embedding_dim_;
  int64_t bs = block_size_;
  size_t ndirty = dirty_slots_.size();
  if (ndirty == 0) return;

  if (opt_cfg_.type == OptimizerConfig::SGD) {
    float lr = opt_cfg_.lr;
    const int32_t* slots = dirty_slots_.data();

    ThreadPool::instance().parallel_for(ndirty, [&](size_t di) {
      int32_t slot = slots[di];
      float* w = slot_ptr(emb_blocks_, slot, D, bs);
      float* g = slot_ptr(grad_blocks_, slot, D, bs);
      for (int32_t d = 0; d < D; ++d) {
        w[d] -= lr * g[d];
        g[d] = 0.0f;
      }
      slot_dirty_[slot] = false;
    });
  } else if (opt_cfg_.type == OptimizerConfig::ADAM) {
    ++t_;
    float lr = opt_cfg_.lr;
    float b1 = opt_cfg_.beta1;
    float b2 = opt_cfg_.beta2;
    float eps = opt_cfg_.eps;
    float bias_corr1 = 1.0f - std::pow(b1, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(b2, static_cast<float>(t_));

    const int32_t* slots = dirty_slots_.data();

    ThreadPool::instance().parallel_for(ndirty, [&](size_t di) {
      int32_t slot = slots[di];
      float* w = slot_ptr(emb_blocks_, slot, D, bs);
      float* g = slot_ptr(grad_blocks_, slot, D, bs);
      float* mp = slot_ptr(m_blocks_, slot, D, bs);
      float* vp = slot_ptr(v_blocks_, slot, D, bs);

      for (int32_t d = 0; d < D; ++d) {
        float gd = g[d];
        mp[d] = b1 * mp[d] + (1.0f - b1) * gd;
        vp[d] = b2 * vp[d] + (1.0f - b2) * gd * gd;
        float m_hat = mp[d] / bias_corr1;
        float v_hat = vp[d] / bias_corr2;
        w[d] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        g[d] = 0.0f;
      }
      slot_dirty_[slot] = false;
    });
  }

  dirty_slots_.clear();
  // Note: gradients zeroed in-place above — no separate zero_grad() call needed.
}

// ===========================================================================
// Serialisation
// ===========================================================================

void EmbeddingTable::state_dict_arrays(
    std::vector<int64_t>& keys,
    std::vector<int32_t>& slots,
    std::vector<float>& weight,
    std::vector<float>& grad,
    std::vector<float>& m,
    std::vector<float>& v,
    int64_t& t,
    std::string& opt_type_str) const {

  auto entries = hash_table_.dump();
  int64_t n = static_cast<int64_t>(entries.size());
  int32_t D = embedding_dim_;

  keys.resize(n);
  slots.resize(n);
  weight.resize(n * D);
  grad.resize(n * D);
  m.assign(n * D, 0.0f);
  v.assign(n * D, 0.0f);

  for (int64_t i = 0; i < n; ++i) {
    keys[i] = entries[i].first;
    int32_t slot = entries[i].second;
    slots[i] = slot;
    std::memcpy(&weight[i * D], slot_ptr(emb_blocks_, slot, D, block_size_), sizeof(float) * D);
    std::memcpy(&grad[i * D],   slot_ptr(grad_blocks_, slot, D, block_size_), sizeof(float) * D);
  }

  if (opt_cfg_.type == OptimizerConfig::ADAM) {
    opt_type_str = "adam";
    t = t_;
    for (int64_t i = 0; i < n; ++i) {
      int32_t slot = entries[i].second;
      std::memcpy(&m[i * D], slot_ptr(m_blocks_, slot, D, block_size_), sizeof(float) * D);
      std::memcpy(&v[i * D], slot_ptr(v_blocks_, slot, D, block_size_), sizeof(float) * D);
    }
  } else {
    opt_type_str = "sgd";
    t = 0;
  }
}

void EmbeddingTable::load_state_dict_arrays(
    int64_t n,
    const int64_t* keys, const int32_t* slots,
    const float* weight, const float* grad,
    const float* m, const float* v,
    int64_t t, const std::string& opt_type_str) {

  int32_t D = embedding_dim_;

  // Populate hash table.
  hash_table_.bulk_insert(keys, slots, n);

  // Ensure blocks for all referenced slots.
  if (n > 0) {
    int32_t max_slot = slots[0];
    for (int64_t i = 1; i < n; ++i) {
      if (slots[i] > max_slot) max_slot = slots[i];
    }
    ensure_slot(max_slot);
  }

  // Copy buffers.
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slots[i];
    std::memcpy(slot_ptr(emb_blocks_, slot, D, block_size_),  weight + i * D, sizeof(float) * D);
    std::memcpy(slot_ptr(grad_blocks_, slot, D, block_size_), grad   + i * D, sizeof(float) * D);
  }

  if (opt_cfg_.type == OptimizerConfig::ADAM && !opt_type_str.empty()) {
    t_ = t;
    for (int64_t i = 0; i < n; ++i) {
      int32_t slot = slots[i];
      std::memcpy(slot_ptr(m_blocks_, slot, D, block_size_), m + i * D, sizeof(float) * D);
      std::memcpy(slot_ptr(v_blocks_, slot, D, block_size_), v + i * D, sizeof(float) * D);
    }
  }
}

// ===========================================================================
// Binary save / load (bucket-by-bucket, zero extra memory allocation)
// ===========================================================================

void EmbeddingTable::save(const std::string& path) const {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("Cannot open " + path + " for writing");

  int32_t D = embedding_dim_;
  int64_t bs = block_size_;
  int64_t n_total = hash_table_.num_entries();
  bool is_adam = (opt_cfg_.type == OptimizerConfig::ADAM);

  // ── Header ─────────────────────────────────────────────────────
  // magic (8 bytes) + version (int32)
  std::fwrite("HASHEMB", 1, 8, fp);
  int32_t version = 0;
  std::fwrite(&version, sizeof(version), 1, fp);

  // num_entries, dim, opt_type (padded to 8 bytes)
  std::fwrite(&n_total, sizeof(n_total), 1, fp);
  std::fwrite(&D, sizeof(D), 1, fp);
  char opt_buf[8] = {};
  std::strncpy(opt_buf, is_adam ? "adam" : "sgd", 7);
  std::fwrite(opt_buf, 1, 8, fp);

  // Optimizer hyper-params
  std::fwrite(&opt_cfg_.lr,     sizeof(float), 1, fp);
  std::fwrite(&opt_cfg_.beta1,  sizeof(float), 1, fp);
  std::fwrite(&opt_cfg_.beta2,  sizeof(float), 1, fp);
  std::fwrite(&opt_cfg_.eps,    sizeof(float), 1, fp);
  std::fwrite(&t_, sizeof(t_), 1, fp);
  std::fwrite(&bs,  sizeof(bs), 1, fp);

  // ── Bucket sections ────────────────────────────────────────────
  // Each bucket: n_entries(int64) + bucket_id(int32)
  //   then n_entries × [key(int64) + slot(int32) + weight(float[D])
  //                     + grad(float[D]) + m(float[D]) + v(float[D])]
  for (int b = 0; b < kNumBuckets; ++b) {
    auto entries = hash_table_.dump_bucket(b);
    int64_t nb = static_cast<int64_t>(entries.size());
    std::fwrite(&nb, sizeof(nb), 1, fp);
    int32_t bid = b;
    std::fwrite(&bid, sizeof(bid), 1, fp);

    for (const auto& e : entries) {
      int64_t key = e.first;
      int32_t slot = e.second;

      std::fwrite(&key,  sizeof(key),  1, fp);
      std::fwrite(&slot, sizeof(slot), 1, fp);
      std::fwrite(slot_ptr(emb_blocks_, slot, D, bs), sizeof(float), D, fp);
      std::fwrite(slot_ptr(grad_blocks_, slot, D, bs), sizeof(float), D, fp);
      if (is_adam) {
        std::fwrite(slot_ptr(m_blocks_, slot, D, bs), sizeof(float), D, fp);
        std::fwrite(slot_ptr(v_blocks_, slot, D, bs), sizeof(float), D, fp);
      }
    }
  }

  std::fclose(fp);
}

void EmbeddingTable::load(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("Cannot open " + path + " for reading");

  // ── Header ─────────────────────────────────────────────────────
  char magic[8];
  std::fread(magic, 1, 8, fp);
  if (std::strncmp(magic, "HASHEMB", 7) != 0) {
    std::fclose(fp);
    throw std::runtime_error("Invalid hash table file: " + path);
  }

  int32_t version;
  std::fread(&version, sizeof(version), 1, fp);
  (void)version;  // reserved for future format versions

  int64_t file_n;
  int32_t file_D;
  char opt_buf[8];
  std::fread(&file_n, sizeof(file_n), 1, fp);
  std::fread(&file_D, sizeof(file_D), 1, fp);
  std::fread(opt_buf, 1, 8, fp);

  if (file_D != embedding_dim_) {
    std::fclose(fp);
    throw std::runtime_error("Embedding dimension mismatch: "
                             + std::to_string(file_D) + " vs " + std::to_string(embedding_dim_));
  }

  float file_lr, file_beta1, file_beta2, file_eps;
  int64_t file_t, file_bs;
  std::fread(&file_lr,    sizeof(float), 1, fp);
  std::fread(&file_beta1, sizeof(float), 1, fp);
  std::fread(&file_beta2, sizeof(float), 1, fp);
  std::fread(&file_eps,   sizeof(float), 1, fp);
  std::fread(&file_t,     sizeof(file_t), 1, fp);
  std::fread(&file_bs,    sizeof(file_bs), 1, fp);

  // Restore optimizer config from file.
  std::string file_opt(opt_buf);
  file_opt = std::string(file_opt.c_str());  // trim at null
  if (file_opt == "adam") {
    opt_cfg_.type = OptimizerConfig::ADAM;
    opt_cfg_.lr    = file_lr;
    opt_cfg_.beta1 = file_beta1;
    opt_cfg_.beta2 = file_beta2;
    opt_cfg_.eps   = file_eps;
  } else {
    opt_cfg_.type = OptimizerConfig::SGD;
    opt_cfg_.lr = file_lr;
  }
  t_ = file_t;

  int32_t D = embedding_dim_;
  int64_t bs = block_size_;
  bool is_adam = (opt_cfg_.type == OptimizerConfig::ADAM);

  // ── Bucket sections ────────────────────────────────────────────
  // For each entry: read key, insert via find_or_create to get the
  // correct slot for the current table state, then read weight/grad/m/v
  // directly into the block buffer (zero extra allocation).
  for (int b = 0; b < kNumBuckets; ++b) {
    int64_t nb;
    int32_t bid;
    std::fread(&nb,  sizeof(nb),  1, fp);
    std::fread(&bid, sizeof(bid), 1, fp);

    for (int64_t i = 0; i < nb; ++i) {
      int64_t key;
      int32_t saved_slot;  // saved slot ID (informational, not used)
      std::fread(&key,        sizeof(key),        1, fp);
      std::fread(&saved_slot, sizeof(saved_slot), 1, fp);

      // Insert key into hash table → get current slot.
      int32_t slot;
      hash_table_.find_or_create(&key, &slot, 1);
      ensure_slot(slot);

      // Read data directly into block buffer (zero-copy from disk).
      std::fread(slot_ptr(emb_blocks_, slot, D, bs),   sizeof(float), D, fp);
      std::fread(slot_ptr(grad_blocks_, slot, D, bs),  sizeof(float), D, fp);
      if (is_adam) {
        std::fread(slot_ptr(m_blocks_, slot, D, bs),   sizeof(float), D, fp);
        std::fread(slot_ptr(v_blocks_, slot, D, bs),   sizeof(float), D, fp);
      }
    }
  }

  std::fclose(fp);
}

}  // namespace hashemb
