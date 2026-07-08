#include "embedding_table.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hashemb {

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
  const auto& blocks = emb_blocks_;  // thread-safe const ref

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slot_indices[i];
    if (slot < 0) {
      std::memset(output + i * D, 0, sizeof(float) * D);
    } else {
      std::memcpy(output + i * D, slot_ptr(blocks, slot, D, bs),
                  sizeof(float) * D);
    }
  }
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

#ifdef HASHEMB_OMP_STEP
    #pragma omp parallel for schedule(static) \
            default(none) \
            shared(lr, slots, D, bs, ndirty, emb_blocks_, grad_blocks_, slot_dirty_)
#endif
    for (size_t di = 0; di < ndirty; ++di) {
      int32_t slot = slots[di];
      float* w = slot_ptr(emb_blocks_, slot, D, bs);
      float* g = slot_ptr(grad_blocks_, slot, D, bs);
      for (int32_t d = 0; d < D; ++d) {
        w[d] -= lr * g[d];
        g[d] = 0.0f;
      }
      slot_dirty_[slot] = false;
    }
  } else if (opt_cfg_.type == OptimizerConfig::ADAM) {
    ++t_;
    float lr = opt_cfg_.lr;
    float b1 = opt_cfg_.beta1;
    float b2 = opt_cfg_.beta2;
    float eps = opt_cfg_.eps;
    float bias_corr1 = 1.0f - std::pow(b1, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(b2, static_cast<float>(t_));

    const int32_t* slots = dirty_slots_.data();

#ifdef HASHEMB_OMP_STEP
    #pragma omp parallel for schedule(static) \
            default(none) \
            shared(lr, b1, b2, eps, bias_corr1, bias_corr2, \
                   slots, D, bs, ndirty, \
                   emb_blocks_, grad_blocks_, m_blocks_, v_blocks_, slot_dirty_)
#endif
    for (size_t di = 0; di < ndirty; ++di) {
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
    }
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
