#include "embedding_table.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace hashemb {

// ===========================================================================
// Construction / destruction
// ===========================================================================

EmbeddingTable::EmbeddingTable(int64_t initial_capacity, int32_t embedding_dim,
                               const OptimizerConfig& opt_cfg,
                               int64_t block_size)
    : initial_capacity_(initial_capacity),
      embedding_dim_(embedding_dim),
      block_size_(block_size),
      opt_cfg_(opt_cfg),
      hash_table_(initial_capacity) {
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
  }
}

// ===========================================================================
// Lookup
// ===========================================================================

void EmbeddingTable::lookup(const int32_t* slot_indices, float* output,
                            int64_t n) const {
  int32_t D = embedding_dim_;
  int64_t bs = block_size_;
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slot_indices[i];
    if (slot < 0) {
      std::memset(output + i * D, 0, sizeof(float) * D);
    } else {
      std::memcpy(output + i * D, slot_ptr(emb_blocks_, slot, D, bs),
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

  // Sort indices by slot to group duplicates.
  auto* idx = new int64_t[n];
  for (int64_t i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx, idx + n, [&](int64_t a, int64_t b) {
    return slot_indices[a] < slot_indices[b];
  });

  int64_t i = 0;
  while (i < n) {
    int32_t slot = slot_indices[idx[i]];
    if (slot < 0) { ++i; continue; }
    int64_t j = i;
    while (j < n && slot_indices[idx[j]] == slot) ++j;

    float* grad_dst = slot_ptr(grad_blocks_, slot, D, block_size_);
    for (int64_t k = i; k < j; ++k) {
      const float* g = grads + idx[k] * D;
      for (int32_t d = 0; d < D; ++d) grad_dst[d] += g[d];
    }
    i = j;
  }
  delete[] idx;
}

// ===========================================================================
// Optimizer step
// ===========================================================================

void EmbeddingTable::zero_grad() {
  int32_t D = embedding_dim_;
  int64_t n = hash_table_.num_entries();
  // Only zero the portion of the first block that has been used.
  // This avoids a massive memset on the full block_size buffer.
  int64_t bs = block_size_;
  if (!grad_blocks_.empty() && grad_blocks_[0].data && n > 0) {
    // For block 0: n entries might span into the next block.
    // Zero from 0 to min(n, bs) entries in block 0.
    int64_t zero_count = (n < bs) ? n : bs;
    std::memset(grad_blocks_[0].data, 0,
                static_cast<size_t>(zero_count) * D * sizeof(float));
    // If entries span beyond block 0, zero remaining blocks entirely.
    if (n > bs) {
      int64_t n_full_blocks = (n - 1) / bs;  // last block index
      for (int64_t b = 1; b <= n_full_blocks; ++b) {
        if (b < static_cast<int64_t>(grad_blocks_.size()) && grad_blocks_[b].data) {
          int64_t this_count = (b == n_full_blocks) ? (n - b * bs) : bs;
          std::memset(grad_blocks_[b].data, 0,
                      static_cast<size_t>(this_count) * D * sizeof(float));
        }
      }
    }
  }
}

void EmbeddingTable::step() {
  int32_t D = embedding_dim_;
  int64_t n = hash_table_.num_entries();
  if (n == 0) return;

  // Ensure all blocks up to the last slot exist.
  ensure_slot(n - 1);

  // Pre-compute base pointers to avoid slot_ptr division/call overhead.
  // All entries are in the first block when n <= block_size.
  int64_t bs = block_size_;
  float* w_base0 = emb_blocks_.empty() ? nullptr : emb_blocks_[0].data;
  float* g_base0 = grad_blocks_.empty() ? nullptr : grad_blocks_[0].data;

  if (opt_cfg_.type == OptimizerConfig::SGD) {
    float lr = opt_cfg_.lr;
    if (n <= bs && w_base0 && g_base0) {
      // Fast path: all entries in block 0, direct pointer arithmetic.
      for (int64_t slot = 0; slot < n; ++slot) {
        float* w = w_base0 + slot * D;
        float* g = g_base0 + slot * D;
        for (int32_t d = 0; d < D; ++d) {
          w[d] -= lr * g[d];
          g[d] = 0.0f;
        }
      }
    } else {
      // General path: entries may span multiple blocks.
      for (int64_t slot = 0; slot < n; ++slot) {
        float* w = slot_ptr(emb_blocks_, slot, D, bs);
        float* g = slot_ptr(grad_blocks_, slot, D, bs);
        for (int32_t d = 0; d < D; ++d) {
          w[d] -= lr * g[d];
          g[d] = 0.0f;
        }
      }
    }
  } else if (opt_cfg_.type == OptimizerConfig::ADAM) {
    ++t_;
    float lr = opt_cfg_.lr;
    float b1 = opt_cfg_.beta1;
    float b2 = opt_cfg_.beta2;
    float eps = opt_cfg_.eps;
    float bias_corr1 = 1.0f - std::pow(b1, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(b2, static_cast<float>(t_));

    float* m_base0 = m_blocks_.empty() ? nullptr : m_blocks_[0].data;
    float* v_base0 = v_blocks_.empty() ? nullptr : v_blocks_[0].data;

    if (n <= bs && w_base0 && g_base0 && m_base0 && v_base0) {
      // Fast path: all entries in block 0, direct pointer arithmetic.
      for (int64_t slot = 0; slot < n; ++slot) {
        float* w = w_base0 + slot * D;
        float* g = g_base0 + slot * D;
        float* m = m_base0 + slot * D;
        float* v = v_base0 + slot * D;

        for (int32_t d = 0; d < D; ++d) {
          float gd = g[d];
          m[d] = b1 * m[d] + (1.0f - b1) * gd;
          v[d] = b2 * v[d] + (1.0f - b2) * gd * gd;
          float m_hat = m[d] / bias_corr1;
          float v_hat = v[d] / bias_corr2;
          w[d] -= lr * m_hat / (std::sqrt(v_hat) + eps);
          g[d] = 0.0f;
        }
      }
    } else {
      // General path: entries may span multiple blocks.
      for (int64_t slot = 0; slot < n; ++slot) {
        float* w = slot_ptr(emb_blocks_, slot, D, bs);
        float* g = slot_ptr(grad_blocks_, slot, D, bs);
        float* m = slot_ptr(m_blocks_, slot, D, bs);
        float* v = slot_ptr(v_blocks_, slot, D, bs);

        for (int32_t d = 0; d < D; ++d) {
          float gd = g[d];
          m[d] = b1 * m[d] + (1.0f - b1) * gd;
          v[d] = b2 * v[d] + (1.0f - b2) * gd * gd;
          float m_hat = m[d] / bias_corr1;
          float v_hat = v[d] / bias_corr2;
          w[d] -= lr * m_hat / (std::sqrt(v_hat) + eps);
          g[d] = 0.0f;
        }
      }
    }
  }

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

}  // namespace hashemb
